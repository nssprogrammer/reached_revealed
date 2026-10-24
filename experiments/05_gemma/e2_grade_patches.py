# ---- Step G-c: grade Gemma 4's repair patches (CPU, ~1-2 h, internet off). Needs proplib.py + proplib9.py (cells above).
# Inputs: competition data, end-to-end output (propagator/: tasks/*/index.json + C.npz, repro/*.json, cfg.json),
#         Step G-a output (gemma_pack/repair.jsonl), Step G-b output (gemma_out/repair_out.jsonl).
# For every task: the reference fix and each distinct Gemma patch are run on the same selected tests:
#   HARNESS  = every test in the test files the task's test patch touches that passes with the reference fix (SWE-bench style
#              fail-to-pass + pass-to-pass);  F2P = the task's fail-to-pass tests only;
#   VISIBLE  = pre-existing tests (not added/edited by the test patch) that execute the edited functions and pass with the
#              reference fix: the tests an agent can find and run itself;
#   EXTENDED = HARNESS + VISIBLE.
import os, re, sys, glob, json, time, random, zlib, shutil, ast, textwrap
import concurrent.futures as cf
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as ssp
sys.path.insert(0, os.environ.get('PROP_WORKDIR', '/kaggle/working'))
import proplib as P; import proplib9 as P9

def find(pattern, what):
    for base in (os.environ.get('PROP_WORKDIR', '/kaggle/working'), '/kaggle/input'):
        hits = sorted(glob.glob(f'{base}/**/{pattern}', recursive=True))
        if hits: return Path(hits[0])
    raise FileNotFoundError(f'attach {what} ({pattern})')
SRC_OUT = find('propagator/tasks/*/C.npz', 'the end-to-end output').parent.parent.parent
PACK = find('gemma_pack/repair.jsonl', 'the Step G-a output').parent
GOUT = find('gemma_out/repair_out.jsonl', 'the Step G-b output').parent
SAVE = P.WORKDIR / 'gemma_eval'; SAVE.mkdir(exist_ok=True)
if SRC_OUT.resolve() != P.OUT.resolve() and (SRC_OUT / 'cfg.json').exists() and not P.CFG:
    shutil.copy(SRC_OUT / 'cfg.json', P.OUT / 'cfg.json'); P.CFG.update(json.load(open(P.OUT / 'cfg.json')))
print('end-to-end:', SRC_OUT, '| pack:', PACK, '| gemma out:', GOUT)
for repo in P.REPOS:
    info = P.build_env(repo); print(f"env {repo:9s} {len(info['installed'])} wheels, pip rc {info['pip_rc']} | cfg {P.detect_cfg(repo)}")
print('dirty_equals:', P.ensure_dirty_equals())

# ------------------------------------------------------------ turning Gemma's answer into edited files
BLOCK = re.compile(r'^#{2,4}\s*`?([A-Za-z_][\w\.]*)`?[^\n]*\n+\s*```(?:python|py)?[ \t]*\n(.*?)\n\s*```', re.M | re.S)
FENCE = re.compile(r'```(?:python|py)?[ \t]*\n(.*?)\n\s*```', re.S)
def parse_answer(text, metas):
    text = re.sub(r'<think>.*?</think>', '', text or '', flags=re.S); out = {}
    for name, code in BLOCK.findall(text):
        m = next((x for x in metas if x['qual'] == name), None) or \
            next((x for x in metas if x['qual'].endswith('.' + name) or name.endswith('.' + x['qual'].split('.')[-1])), None)
        if m and m['qual'] not in out: out[m['qual']] = code
    if not out and len(metas) == 1:
        blocks = FENCE.findall(text)
        if blocks: out[metas[0]['qual']] = blocks[0]
    return out

def edited_files(w, metas, codes):
    """replace each function's line span (decorators included) with Gemma's version, re-indented; None if not valid Python"""
    by_file = {}
    for m in metas:
        if m['qual'] in codes: by_file.setdefault(m['file'], []).append(m)
    new = {}
    for rel, ms in by_file.items():
        lines = (Path(w) / rel).read_text().split('\n')
        for m in sorted(ms, key=lambda x: -x['start']):
            body = textwrap.dedent(codes[m['qual']].expandtabs(4)).strip('\n').split('\n')
            old = lines[m['start'] - 1:m['end']]
            decos = []
            for l in old:
                if l.strip().startswith(('def ', 'async def ')): break
                decos.append(l.strip())
            if decos and not body[0].lstrip().startswith('@'): body = decos + body      # Gemma dropped the decorators: keep them
            pad = ' ' * m['indent']; lines[m['start'] - 1:m['end']] = [pad + l if l.strip() else '' for l in body]
        src = '\n'.join(lines)
        try: ast.parse(src)
        except SyntaxError: return None
        new[rel] = src
    return new

# ------------------------------------------------------------ one task
PACKROWS = {json.loads(l)['task']: json.loads(l) for l in open(PACK / 'repair.jsonl') if l.strip()}
OUTROWS = {json.loads(l)['task']: json.loads(l) for l in open(GOUT / 'repair_out.jsonl') if l.strip()}

def grade_task(tid, cap=300):
    f = SAVE / f'{tid}.json'
    if f.exists(): return json.load(open(f))
    t = P.T[tid]; repo = P.repo_of(t); metas = PACKROWS[tid]['funcs']; texts = OUTROWS[tid]['texts']
    row = dict(task=tid, repo=repo, n_samples=len(texts))
    def done(**kw):
        row.update(kw); json.dump(row, open(f, 'w'), default=str); return row
    try:
        d = SRC_OUT / 'tasks' / tid; idx = json.load(open(d / 'index.json')); keys, funcs = idx['tests'], idx['funcs']
        C = ssp.load_npz(d / 'C.npz').tocsc(); fi = {x: i for i, x in enumerate(funcs)}
        f2p_names = set(json.load(open(SRC_OUT / 'repro' / f'{tid}.json')).get('f2p', []))
        F2P = {k for k in keys if (k in f2p_names or k.split('::')[-1].split('[')[0] in f2p_names) and k not in P9.TRACER_SENSITIVE}
        w = P.fresh_copy(t, f'gc_{tid}')
        ok, _ = P.apply_patch(t['test_patch'], w)
        if not ok: return done(status='test_patch-failed')
        targets = set(P.test_targets(t, w))
        H_all = {k for k in keys if P9.key_file(w, k) in targets and k not in P9.TRACER_SENSITIVE}
        cols = [fi[m['qual']] for m in metas if m['qual'] in fi]
        tp = t['test_patch']                                   # tests the test patch adds or edits are not visible to an agent in this form
        touched = set(re.findall(r'^[+-]\s*(?:async\s+)?def\s+(test\w*)', tp, re.M)) | set(re.findall(r'^@@[^@]*@@.*?def\s+(test\w*)', tp, re.M))
        cover = sorted(k for k in {keys[r] for c in cols for r in C[:, c].nonzero()[0]}
                       if k not in F2P and k not in P9.TRACER_SENSITIVE and k.split('::')[-1].split('[')[0] not in touched)
        rng = random.Random(zlib.crc32(tid.encode()))
        if len(cover) > cap: cover = sorted(rng.sample(cover, cap))
        sel = sorted(H_all | F2P | set(cover))
        # reference fix on the same selection
        base_src = {m['file']: (Path(w) / m['file']).read_text() for m in metas}
        ok, _ = P.apply_patch(t['patch'], w)
        if not ok: return done(status='patch-failed')
        gold, _, gm = P9.run_subset(repo, w, sel, 900)
        H = {k for k in H_all if gold[k] == 'passed'}; F2Pp = {k for k in F2P if gold[k] == 'passed'}
        V = {k for k in cover if gold[k] == 'passed'}; X = H | V
        row.update(n_H=len(H), n_F2P=len(F2Pp), n_V=len(V), n_cover_all=len(cover), gold_secs=gm['secs'])
        if not F2Pp: return done(status='gold-f2p-not-passing')
        # back to base + test patch, then each distinct Gemma patch
        w = P.fresh_copy(t, f'gc_{tid}'); P.apply_patch(t['test_patch'], w)
        variants, samples = {}, []
        for i, text in enumerate(texts):
            codes = parse_answer(text, metas)
            if not codes: samples.append(dict(i=i, parsed=False)); continue
            new = edited_files(w, metas, codes)
            if new is None: samples.append(dict(i=i, parsed=True, valid=False)); continue
            if all(new[r] == base_src.get(r) for r in new): samples.append(dict(i=i, parsed=True, valid=True, noop=True)); continue
            h = zlib.crc32(json.dumps(new, sort_keys=True).encode())
            if h not in variants:
                for rel, src in new.items(): (Path(w) / rel).write_text(src)
                st, _, mm = P9.run_subset(repo, w, sel, 900)
                for rel in new: (Path(w) / rel).write_text(base_src[rel])
                ok_ = lambda S: all(st[k] == 'passed' for k in S)
                variants[h] = dict(f2p=ok_(F2Pp), harness=ok_(H), visible=ok_(V), extended=ok_(X),
                                   n_V_fail=sum(st[k] != 'passed' for k in V), timeout=bool(mm['killed']))
            samples.append(dict(i=i, parsed=True, valid=True, noop=False, **variants[h]))
        shutil.rmtree(w, ignore_errors=True)
        return done(status='graded', samples=samples, n_variants=len(variants))
    except Exception as e:
        return done(status='crash', msg=repr(e)[:300])

tids = [t for t in OUTROWS if t in PACKROWS and t in P.T]
print(f'grading {len(tids)} tasks ({sum(len(OUTROWS[t]["texts"]) for t in tids)} Gemma answers)'); t0 = time.time(); rows = []
with cf.ThreadPoolExecutor(int(os.environ.get('GC_WORKERS', 3))) as ex:
    for k, r in enumerate(ex.map(grade_task, tids)):
        rows.append(r)
        if (k + 1) % 10 == 0: print(f'   {k + 1}/{len(tids)}  {(time.time() - t0) / 60:.0f} min', flush=True)
R = pd.DataFrame(rows); R.drop(columns=['samples'], errors='ignore').to_csv(SAVE / 'gc_tasks.csv', index=False)

# ------------------------------------------------------------ report
print('\nSTATUS', R.status.value_counts().to_dict())
G = R[R.status == 'graded']
S = pd.DataFrame([dict(task=r.task, repo=r.repo, **s) for r in G.itertuples() for s in r.samples])
S.to_csv(SAVE / 'gc_samples.csv', index=False)
for c in ('valid', 'noop', 'f2p', 'harness', 'visible', 'extended', 'timeout'):
    if c not in S: S[c] = np.nan
S['score'] = S.harness.fillna(False).astype(bool) & S.valid.fillna(False).astype(bool) & ~S.noop.fillna(False).astype(bool)
print(f'\nE2 GEMMA 4 31B REPAIR, oracle localization: {len(G)} tasks, {len(S)} answers')
print(f"   parsed {S.parsed.mean():.0%} | valid Python {S.valid.fillna(False).mean():.0%} | no-op {S.noop.fillna(False).mean():.0%}")
g0 = S[S.i == 0]; smp = S[S.i > 0]
print(f"   resolve rate (harness = F2P + P2P of the touched test files): greedy {g0.score.mean():.0%} | sampled (T=0.8) {smp.score.mean():.0%} | "
      f"any of 5 (oracle) {S.groupby('task').score.any().mean():.0%}")
print(f"   F2P-only resolve (greedy): {(g0.f2p.fillna(False).astype(bool)).mean():.0%}  (gap to harness = patches broken on pass-to-pass tests)")
hp = S[S.score]
if len(hp):
    print(f"\n   FALSE-POSITIVE REWARD on real Gemma patches: of {len(hp)} harness-passing answers, {(~hp.visible.astype(bool)).mean():.1%} fail a "
          f"pre-existing test that executes the edit ({hp[~hp.visible.astype(bool)].task.nunique()} tasks)")
# selection policies over the 5 answers of each task
pol = []
for tid, q in S.groupby('task'):
    q = q.sort_values('i'); greedy = q.iloc[0]
    vis = q[q.valid.fillna(False).astype(bool) & ~q.noop.fillna(False).astype(bool) & q.visible.fillna(False).astype(bool)]
    pick = vis.iloc[0] if len(vis) else greedy
    rnd = q.iloc[random.Random(zlib.crc32(tid.encode())).randrange(len(q))]
    pol.append(dict(task=tid, greedy=bool(greedy.score), visible_select=bool(pick.score), random=bool(rnd.score), oracle=bool(q.score.any())))
Pl = pd.DataFrame(pol)
print('\n   SELECTION among the 5 answers (resolve rate under the harness):')
print('   ' + ' | '.join(f'{c} {Pl[c].mean():.0%}' for c in ('greedy', 'random', 'visible_select', 'oracle')))
from scipy.stats import binomtest
a = int((Pl.visible_select & ~Pl.greedy).sum()); b = int((~Pl.visible_select & Pl.greedy).sum())
print(f'   visible-test selection vs greedy: wins {a}, losses {b}, sign-test p = {binomtest(a, a + b).pvalue if a + b else 1:.3g}')
print(f"   tasks with a pre-existing test executing the edit (VISIBLE non-empty): {(G.n_V > 0).mean():.0%}, median {G.n_V.median():.0f} tests")
Pl.to_csv(SAVE / 'gc_selection.csv', index=False)
shutil.make_archive(str(P.WORKDIR / 'gemma_eval'), 'gztar', P.WORKDIR, 'gemma_eval'); print('saved', P.WORKDIR / 'gemma_eval.tar.gz')
