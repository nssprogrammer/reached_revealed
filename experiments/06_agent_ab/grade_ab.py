# ---- A/B grading (CPU, internet off): resolve each run's final patch with the harness; McNemar per seed and a task-level
# permutation test over all seeds. Inputs: competition data + the A/B output (ab_out/<prompt>[_s<seed>]/<task>.json).
import os, sys, glob, json, time, shutil
import concurrent.futures as cf
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import binomtest
sys.path.insert(0, os.environ.get('PROP_WORKDIR', '/kaggle/working'))
import proplib as P

def find_ab():
    for base in (os.environ.get('PROP_WORKDIR', '/kaggle/working'), '/kaggle/input'):
        hits = sorted(glob.glob(f'{base}/**/ab_out/*/*.json', recursive=True))
        if hits: return Path(hits[0]).parent.parent
    raise FileNotFoundError('attach the A/B run output (ab_out/)')
AB = find_ab(); SAVE = P.WORKDIR / 'ab_grade'; SAVE.mkdir(exist_ok=True)
REC = {}
for f in glob.glob(str(AB / '*' / '*.json')):
    r = json.load(open(f)); REC[(r['task'], r['arm'])] = r
def split_tag(tag):
    p, _, s = tag.partition('_s'); return p, int(s or 0)
TAGS = sorted({x for _, x in REC}); PROMPTS_ = sorted({split_tag(x)[0] for x in TAGS})
assert len(PROMPTS_) == 2, PROMPTS_
A_, B_ = PROMPTS_                                            # 'v2', 'v32'
SEEDS = sorted(s for s in {split_tag(x)[1] for x in TAGS} if all((f'{p}_s{s}' if s else p) in TAGS for p in PROMPTS_))
tag = lambda p, s: f'{p}_s{s}' if s else p
TIDS = sorted({t for t, _ in REC if all((t, tag(p, s)) in REC for p in PROMPTS_ for s in SEEDS)})
print('A/B output:', AB, '| prompts', PROMPTS_, '| seeds', SEEDS, '| tasks with every run', len(TIDS))
if SEEDS == [0]: print('NOTE: only the first run (seed 0) is in this ab_out; attach the ab_out.tgz written by the second-seed run to analyse both.')
for repo in P.REPOS:
    P.build_env(repo); P.detect_cfg(repo)
print('dirty_equals:', P.ensure_dirty_equals())

def grade_one(t, repo, targets, H, f2p, r):
    d = r.get('diff') or ''
    if r.get('stop') == 'skipped-budget': return None, 'not run'
    if not d.strip(): return False, 'crash, no patch' if r.get('stop') == 'crash' else 'empty patch'
    if r.get('touched_tests'): return False, 'modified tests'
    w = P.fresh_copy(t, f"abg_{t['instance_id']}_{r['arm']}"); P.apply_patch(t['test_patch'], w); ok, _ = P.apply_patch(d, w)
    try:
        if not ok: return False, 'patch did not apply'
        res, _ = P.suite_run(repo, w, 'arm', timeout=1500, targets=targets)
        fails = [k for k in H if res.get(k, ('missing',))[0] != 'passed']
        if fails and not any(k in f2p for k in fails):          # only pass-to-pass failures: re-run once to rule out flaky tests
            res, _ = P.suite_run(repo, w, 'arm2', timeout=1500, targets=targets)
            fails = [k for k in H if res.get(k, ('missing',))[0] != 'passed']
        return (not fails), ('resolved' if not fails else 'f2p' if any(k in f2p for k in fails) else 'p2p only')
    finally:
        shutil.rmtree(w, ignore_errors=True)

def grade_task(tid):
    f = SAVE / f'{tid}.json'; out = json.load(open(f)) if f.exists() else None
    t = P.T[tid]; repo = P.repo_of(t)
    try:
        if out is None or out.get('v') != 2:                    # reference fix once per task (cached)
            out = dict(task=tid, repo=repo)
            w = P.fresh_copy(t, f'abg_{tid}'); ok, _ = P.apply_patch(t['test_patch'], w)
            if not ok: out.update(status='test_patch-failed'); return out
            targets = P.test_targets(t, w); ok, _ = P.apply_patch(t['patch'], w)
            if not targets or not ok: out.update(status='no-targets' if not targets else 'patch-failed'); return out
            gold, _ = P.suite_run(repo, w, 'g', timeout=1500, targets=targets)
            gold2, _ = P.suite_run(repo, w, 'g2', timeout=1500, targets=targets); shutil.rmtree(w, ignore_errors=True)
            H = sorted(k for k, (s, _) in gold.items() if s == 'passed' and gold2.get(k, ('missing',))[0] == 'passed')   # stable on the fix
            out['n_flaky_gold'] = sum(1 for k, (s, _) in gold.items() if s == 'passed') - len(H); req, _ = P.required_tests(t)
            f2p = [k for k in H if k.split('::')[-1].split('[')[0] in req]
            out.update(targets=targets, H=H, f2p=f2p, v=2)
            if not H or not f2p: out.update(status='gold-not-passing'); return out
            out['status'] = 'graded'
        if out.get('status') != 'graded': return out
        for x in TAGS:                                          # grade only runs not graded before
            if (tid, x) in REC and x not in out:
                out[x], out[f'{x}_why'] = grade_one(t, repo, out['targets'], out['H'], out['f2p'], REC[(tid, x)])
        return out
    except Exception as e:
        out = out or dict(task=tid, repo=repo); out.update(status='crash', msg=repr(e)[:300]); return out
    finally:
        json.dump(out, open(f, 'w'), default=str)

t0 = time.time(); G = []
with cf.ThreadPoolExecutor(int(os.environ.get('AB_GRADE_WORKERS', 3))) as ex:
    for k, r in enumerate(ex.map(grade_task, TIDS)):
        G.append(r)
        if (k + 1) % 20 == 0: print(f'   graded {k + 1}/{len(TIDS)}  {(time.time() - t0) / 60:.0f} min', flush=True)
G = pd.DataFrame(G); G.drop(columns=['H', 'f2p', 'targets'], errors='ignore').to_csv(SAVE / 'ab_tasks.csv', index=False)
print('\nGRADING STATUS', G.status.value_counts().to_dict(), '| tests flaky on the reference fix (excluded):', int(G.get('n_flaky_gold', pd.Series(dtype=float)).fillna(0).sum()))

def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h
cols = [tag(p, s) for p in PROMPTS_ for s in SEEDS]
Q = G[(G.status == 'graded')].dropna(subset=cols).copy()
for c in cols: Q[c] = Q[c].astype(bool)
n = len(Q); summary = dict(n_tasks=n, seeds=SEEDS, per_seed={})
print(f'\nDEV-SET A/B, {n} paired tasks x {len(SEEDS)} seed(s) (same model, tools, limits and sampling; only the system prompt differs)')
for s in SEEDS:
    a, b = Q[tag(A_, s)], Q[tag(B_, s)]; n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    p = binomtest(n01, n01 + n10).pvalue if n01 + n10 else 1.0
    print(f'   seed {s}: {A_} {int(a.sum())}/{n} = {a.mean():.1%} | {B_} {int(b.sum())}/{n} = {b.mean():.1%} | '
          f'McNemar {B_} only {n01}, {A_} only {n10}, p = {p:.3g}')
    summary['per_seed'][s] = dict(a=float(a.mean()), b=float(b.mean()), b_only=n01, a_only=n10, p=float(p))
SA = Q[[tag(A_, s) for s in SEEDS]].mean(axis=1).values; SB = Q[[tag(B_, s) for s in SEEDS]].mean(axis=1).values; D = SB - SA
ka, kb = int(round(SA.sum() * len(SEEDS))), int(round(SB.sum() * len(SEEDS))); N = n * len(SEEDS)
la, ha = wilson(ka, N); lb, hb = wilson(kb, N)
print(f'\n   ALL SEEDS: {A_} {ka}/{N} = {ka / N:.1%} [{la:.1%}-{ha:.1%}] | {B_} {kb}/{N} = {kb / N:.1%} [{lb:.1%}-{hb:.1%}]')
nz = D[D != 0]; rng = np.random.default_rng(0)                  # task-level sign-flip permutation test (task = unit)
if len(nz):
    flips = rng.choice([-1.0, 1.0], size=(200000, len(nz))); null = (flips * nz).mean(axis=1) * len(nz) / n
    p_perm = float((np.abs(null) >= abs(D.mean()) - 1e-12).mean())
else: p_perm = 1.0
boot = [rng.choice(D, n).mean() for _ in range(20000)]; lo, hi = np.percentile(boot, [2.5, 97.5])
wins, losses = int((D > 0).sum()), int((D < 0).sum())
print(f'   difference {B_} - {A_}: {D.mean():+.1%} of tasks [95% CI {lo:+.1%}, {hi:+.1%}] (bootstrap over tasks)')
print(f'   tasks where {B_} resolves more often: {wins}, {A_} more often: {losses}; task-level permutation p = {p_perm:.4f}; '
      f'sign test p = {binomtest(wins, wins + losses).pvalue if wins + losses else 1:.4f}')
summary.update(all=dict(a=ka / N, b=kb / N, diff=float(D.mean()), diff_ci=[float(lo), float(hi)], wins=wins, losses=losses, p_perm=p_perm))
print('   by repository:', Q.assign(a=SA, b=SB).groupby('repo')[['a', 'b']].mean().round(2).to_dict('index'))
why = {p: pd.Series([Q.loc[i, f'{tag(p, s)}_why'] for i in Q.index for s in SEEDS]).value_counts().to_dict() for p in PROMPTS_}
print('   outcomes over all runs:', why)
for p in PROMPTS_:
    w = why[p]; edits = w.get('resolved', 0) + w.get('f2p', 0) + w.get('p2p only', 0) + w.get('modified tests', 0) + w.get('patch did not apply', 0)
    print(f"   {p}: patches with an edit {edits}, correct {w.get('resolved', 0)} ({w.get('resolved', 0) / max(edits, 1):.0%})")
B = pd.DataFrame([dict(prompt=split_tag(x)[0], **{k: REC[(t, x)].get(k) for k in ('stop', 'calls', 'vtime', 'completion_tokens', 'probe', 'submitted')})
                  for t in Q.task for x in cols])
print('\nBEHAVIOUR (all runs)')
print(B.groupby('prompt').agg(submitted=('submitted', 'mean'), tool_calls=('calls', 'mean'), virtual_min=('vtime', lambda v: v.mean() / 60),
                              hit_call_cap=('stop', lambda s: (s == 'tool_limit').mean()), used_probe=('probe', 'mean'),
                              tokens=('completion_tokens', 'mean')).round(3).to_string())
json.dump(summary, open(SAVE / 'ab_summary.json', 'w'), indent=1, default=str)
shutil.make_archive(str(P.WORKDIR / 'ab_grade'), 'gztar', P.WORKDIR, 'ab_grade'); print('saved', P.WORKDIR / 'ab_grade.tar.gz')
