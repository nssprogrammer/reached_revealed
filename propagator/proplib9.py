# proplib9.py - Step 9: empirical Green's function G from mutants (+ tracer-fidelity recheck).
# Requires proplib.py (Cell A) and the Step 8 outputs in /kaggle/working/propagator/tasks/<task>/.
import os, re, ast, json, time, random, zlib, shutil
import concurrent.futures as cf
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as ssp
import proplib as P

SEL_PLUGIN = r'''
import os, json
import pytest

def _key(nid):
    p = nid.split("::"); f = p[0][:-3].replace("/", ".") if p[0].endswith(".py") else p[0]
    return ".".join([f] + p[1:-1]) + "::" + p[-1]

def pytest_collection_modifyitems(session, config, items):
    path = os.environ.get("PROP_SELECT")
    if not path:
        return
    with open(path) as fh:
        want = set(fh.read().split("\n"))
    keep, drop = [], []
    for it in items:
        (keep if _key(it.nodeid) in want else drop).append(it)
    items[:] = keep
    if drop:
        config.hook.pytest_deselected(items=drop)

_MF = os.path.realpath(os.environ.get("PROP_MUT_FILE", "/nonexistent"))
_MQ = os.environ.get("PROP_MUT_QUAL", "")
_ROOT = os.path.realpath(os.environ.get("PROP_ROOT", ".")) + os.sep
_OUT = os.environ.get("PROP_DEPTH_OUT")

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    if not (_OUT and rep.failed and call.excinfo is not None):
        return
    frames, tb = [], call.excinfo.tb
    while tb is not None:
        frames.append(tb.tb_frame); tb = tb.tb_next
    tcode = getattr(getattr(item, "function", None), "__code__", None)
    ti = next((i for i, f in enumerate(frames) if f.f_code is tcode), 0)
    mi = next((i for i, f in enumerate(frames)
               if os.path.realpath(f.f_code.co_filename) == _MF and f.f_code.co_qualname.replace("<locals>.", "") == _MQ), None)
    d_all = d_pkg = -1
    if mi is not None and mi >= ti:
        d_all = mi - ti
        d_pkg = sum(1 for f in frames[ti + 1:mi + 1] if os.path.realpath(f.f_code.co_filename).startswith(_ROOT))
    with open(_OUT, "a") as fh:
        fh.write(json.dumps([_key(item.nodeid), call.when, d_all, d_pkg, type(call.excinfo.value).__name__]) + "\n")
'''
(P.PLUG / 'prop_sel.py').write_text(SEL_PLUGIN)

# ------------------------------------------------------------ locating functions and making mutants
FLIP = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt, ast.Eq: ast.NotEq, ast.NotEq: ast.Eq,
        ast.Is: ast.IsNot, ast.IsNot: ast.Is, ast.In: ast.NotIn, ast.NotIn: ast.In}
DEFS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)

def locate(root, name):
    root, parts = Path(root), name.split('.')
    for k in range(len(parts) - 1, 0, -1):
        rel = '/'.join(parts[:k])
        for base in (root, root / 'src'):
            for cand in (base / f'{rel}.py', base / rel / '__init__.py'):
                if cand.exists(): return cand, '.'.join(parts[k:])
    return None, None

def find_def(tree, qual):
    parts = qual.split('.')
    def rec(node, i):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if ch.name == parts[i]:
                    if i == len(parts) - 1: return ch
                    r = rec(ch, i + 1)
                    if r is not None: return r
            elif not isinstance(ch, ast.Lambda):
                r = rec(ch, i)
                if r is not None: return r
        return None
    return rec(tree, 0)

def mutation_points(fn):
    pts = []
    def visit(n):
        if isinstance(n, ast.Compare) and all(type(o) in FLIP for o in n.ops): pts.append(('cmp', n))
        elif isinstance(n, ast.BoolOp): pts.append(('bool', n))
        elif isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Not): pts.append(('not', n))
        elif isinstance(n, ast.Constant) and type(n.value) is int: pts.append(('const', n))
        elif isinstance(n, ast.Return) and n.value is not None and not (isinstance(n.value, ast.Constant) and n.value.value is None):
            pts.append(('ret', n))
        if isinstance(n, (ast.If, ast.While, ast.IfExp)): pts.append(('cond', n.test))
        for ch in ast.iter_child_nodes(n):
            if not isinstance(ch, DEFS): visit(ch)
    for stmt in fn.body:
        if not isinstance(stmt, DEFS): visit(stmt)
    return [(k, n) for k, n in pts if getattr(n, 'end_lineno', None) is not None]

def _offset(lines, lineno, col):              # ast columns are UTF-8 byte offsets
    return sum(len(l) for l in lines[:lineno - 1]) + len(lines[lineno - 1].encode()[:col].decode(errors='ignore'))

def apply_mutation(src, kind, node):
    lines = src.splitlines(keepends=True)
    a = _offset(lines, node.lineno, node.col_offset); b = _offset(lines, node.end_lineno, node.end_col_offset)
    seg = src[a:b]
    if kind == 'cmp':
        new = ast.Compare(left=node.left, ops=[FLIP[type(o)]() for o in node.ops], comparators=node.comparators)
        rep = '(' + ast.unparse(new) + ')'
    elif kind == 'bool':
        rep = '(' + ast.unparse(ast.BoolOp(op=ast.Or() if isinstance(node.op, ast.And) else ast.And(), values=node.values)) + ')'
    elif kind == 'not':
        rep = '(' + ast.unparse(node.operand) + ')'
    elif kind == 'const':
        rep = str(node.value + 1)
    elif kind == 'ret':
        rep = 'return None'
    elif kind == 'cond':
        rep = '(not (' + seg + '))'
    else:
        return None
    out = src[:a] + rep + src[b:]
    try:
        compile(out, '<mutant>', 'exec')
    except SyntaxError:
        return None
    return out

# ------------------------------------------------------------ choosing mutation sites on the runtime graph
def choose_sites(funcs, edges, fstar_idx, w, rng, n_f=3, n_each=4, n_rand=6):
    succ, pred = defaultdict(set), defaultdict(set)
    for a, b in edges: succ[a].add(b); pred[b].add(a)
    def ok(i):
        f, q = locate(w, funcs[i])
        return f is not None and not q.endswith('<module>')
    F = [i for i in fstar_idx if ok(i)][:n_f]
    callers = sorted({p for i in F for p in pred[i]} - set(F))
    callees = sorted({s for i in F for s in succ[i]} - set(F))
    d1 = set(F) | set(callers) | set(callees)
    d2 = sorted({x for i in callers + callees for x in pred[i] | succ[i]} - d1)
    pick = lambda xs, k: rng.sample([x for x in xs if ok(x)], min(k, len([x for x in xs if ok(x)])))
    sites = [(i, 'Fstar') for i in F] + [(i, 'caller') for i in pick(callers, n_each)] + \
            [(i, 'callee') for i in pick(callees, n_each)] + [(i, 'd2') for i in pick(d2, n_each)]
    used = {i for i, _ in sites}
    rest = [i for i in range(len(funcs)) if i not in used and not funcs[i].endswith('<module>')]
    sites += [(i, 'rand') for i in pick(rest, n_rand)]
    return sites

def key_file(w, key):
    parts = key.split('::')[0].split('.')
    for k in range(len(parts), 0, -1):
        cand = Path(w) / ('/'.join(parts[:k]) + '.py')
        if cand.exists(): return '/'.join(parts[:k]) + '.py'
    return None

# ------------------------------------------------------------ running a selected subset (v2: fixed rootdir/config)
TRACER_SENSITIVE = {'tests.test_syntax::test_from_path', 'tests.test_syntax::test_syntax_guess_lexer'}

def ini_for(w):
    """the config file pytest picks when run with no paths from the repo root (what the full-suite runs used)"""
    w = Path(w)
    for name in ('pytest.ini', '.pytest.ini', 'pyproject.toml', 'tox.ini', 'setup.cfg'):
        p = w / name
        if not p.exists(): continue
        txt = p.read_text(errors='replace')
        if name == 'pyproject.toml' and '[tool.pytest' not in txt: continue
        if name == 'tox.ini' and '[pytest]' not in txt: continue
        if name == 'setup.cfg' and '[tool:pytest]' not in txt: continue
        return p
    return None

def run_subset(repo, w, sel, timeout, mut_path=None, qual=''):
    w = Path(w); files = sorted({f for f in (key_file(w, k) for k in sel) if f})
    (w / '.sel.txt').write_text('\n'.join(sel)); dep = w / '.depth.jsonl'; dep.unlink(missing_ok=True)
    xml = w / '.junit_sub.xml'; xml.unlink(missing_ok=True); ini = ini_for(w)
    env = P.env_for(repo, w, dict(PROP_SELECT=str(w / '.sel.txt'), PROP_MUT_FILE=str(mut_path.resolve()) if mut_path else '/nonexistent',
                                  PROP_MUT_QUAL=qual, PROP_DEPTH_OUT=str(dep), PROP_ROOT=str(w)))
    cmd = P.cmd_for(repo) + ['--rootdir', str(w)] + (['-c', str(ini)] if ini else []) + files + ['-p', 'prop_sel', f'--junitxml={xml}']
    rc, out, secs, peak, killed_by = P.sp(cmd, w, timeout, env)
    res = P.junit_cases(xml); st = {k: res.get(k, ('missing',))[0] for k in sel}
    depths = {}
    if dep.exists():
        for line in open(dep):
            k, when, da, dp, exc = json.loads(line); depths[k] = [da, dp, exc]
    return st, depths, dict(rc=rc, secs=round(secs, 1), killed=killed_by, tail=out[-300:])

# ------------------------------------------------------------ one task
def g_task(t, cap_tests=300, fid_recheck=False, mut_timeout=300):
    tid, repo = t['instance_id'], P.repo_of(t); d = P.TD / tid
    if (d / 'G.json').exists(): return json.load(open(d / 'G.json'))['summary']
    if not (d / 'index.json').exists(): return dict(task=tid, repo=repo, error='no step-8 data')
    idx = json.load(open(d / 'index.json')); C = ssp.load_npz(d / 'C.npz').tocsc()
    keys, funcs, edges = idx['tests'], idx['funcs'], idx['edges']; fi = {f: i for i, f in enumerate(funcs)}
    w = P.fresh_copy(t, f'g_{tid}'); summ = dict(task=tid, repo=repo, version=2)
    try:
        if not (P.apply_patch(t['patch'], w)[0] and P.apply_patch(t['test_patch'], w)[0]):
            return {**summ, 'error': 'patch'}
        base, mb = P.suite_run(repo, w, 'gbase')                       # healthy reference state: gold + test patch
        if mb['killed']: return {**summ, 'error': 'baseline ' + str(mb['killed'])}
        passed = {k for k, (s, _) in base.items() if s == 'passed' and k not in TRACER_SENSITIVE}
        if fid_recheck:                                                # identical code + tests, traced vs untraced
            tr, mt = P.suite_run(repo, w, 'gtr', traced=True)
            mism = sorted(k for k in set(base) | set(tr) if base.get(k, ('missing',))[0] != tr.get(k, ('missing',))[0])
            summ.update(fid2=len(mism), fid2_list=mism[:6])
            for f in ('.trace_tests.jsonl', '.trace_meta.json'): (w / f).unlink(missing_ok=True)
        rng = random.Random(zlib.crc32(tid.encode()))
        Fs = P.patch_functions(P.extract_lite(t), t['patch'])[0]
        fstar_idx = [fi[f] for f in sorted(Fs) if f in fi and not f.endswith('<module>')]
        rec = json.load(open(P.RD / f'{tid}.json')); f2p_names = set(rec.get('f2p', []))
        f2p_rows = {i for i, k in enumerate(keys) if k in f2p_names or k.split('::')[-1].split('[')[0] in f2p_names}
        sites = choose_sites(funcs, edges, fstar_idx, w, rng)
        # identity control: union of all selections, no mutation; anything that fails here is unstable in isolation
        union = sorted({r for s, _ in sites for r in C[:, s].nonzero()[0].tolist()} | f2p_rows)
        union = [r for r in union if keys[r] in passed]
        if len(union) > 2 * cap_tests: union = sorted(rng.sample(union, 2 * cap_tests))
        unstable = set()
        if union:
            st0, _, m0 = run_subset(repo, w, [keys[r] for r in union], mut_timeout)
            unstable = {k for k, v in st0.items() if v != 'passed'}
            summ.update(ctrl_n=len(union), ctrl_bad=len(unstable), ctrl_bad_list=sorted(unstable)[:5], ctrl_rc=m0['rc'])
        mutants = []
        for site, cls in sites:
            path, qual = locate(w, funcs[site]); src0 = path.read_text()
            try:
                fn = find_def(ast.parse(src0), qual)
            except SyntaxError:
                fn = None
            if fn is None or isinstance(fn, ast.ClassDef): continue
            pts = mutation_points(fn); rng.shuffle(pts)
            n_mut = 2 if cls == 'Fstar' else 1; made = 0
            cover = set(C[:, site].nonzero()[0].tolist())
            for kind, node in pts:
                if made >= n_mut: break
                src1 = apply_mutation(src0, kind, node)
                if src1 is None: continue
                made += 1
                sel_rows = sorted(r for r in cover | f2p_rows if keys[r] in passed and keys[r] not in unstable)
                if len(sel_rows) > cap_tests: sel_rows = sorted(rng.sample(sel_rows, cap_tests))
                m = dict(site=funcs[site], cls=cls, op=kind, line=node.lineno, n_cover=len(cover), n_sel=len(sel_rows))
                if not sel_rows:
                    mutants.append({**m, 'killed': [], 'skip': 'no covering test'}); continue
                sel = [keys[r] for r in sel_rows]
                path.write_text(src1)
                try:
                    st, depths, meta = run_subset(repo, w, sel, mut_timeout, mut_path=path, qual=qual)
                finally:
                    path.write_text(src0)
                kill = [r for r, k in zip(sel_rows, sel) if st[k] in ('failed', 'error')]
                m.update(killed=kill, n_missing=sum(v == 'missing' for v in st.values()), secs=meta['secs'], rc=meta['rc'],
                         timeout=bool(meta['killed']), depth={str(r): depths.get(keys[r]) for r in kill})
                mutants.append(m)
        ks = [m for m in mutants if 'skip' not in m and not m.get('timeout')]
        summ.update(n_mut=len(mutants), n_run=len(ks), n_timeout=sum(bool(m.get('timeout')) for m in mutants),
                    n_killed=sum(bool(m['killed']) for m in ks), baseline_pass=len(passed),
                    n_missing=sum(m.get('n_missing', 0) for m in ks))
        json.dump(dict(summary=summ, keys_n=len(keys), mutants=mutants), open(d / 'G.json', 'w'))
        return summ
    except Exception as e:
        return {**summ, 'error': repr(e)[:300]}
    finally:
        shutil.rmtree(w, ignore_errors=True)

def run_g(task_ids, fid_ids=(), workers=3, log_every=10):
    rows, t0 = [], time.time()
    with cf.ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(g_task, P.T[x], fid_recheck=(x in set(fid_ids))): x for x in task_ids}
        for i, fu in enumerate(cf.as_completed(futs)):
            try:
                rows.append(fu.result())
            except Exception as e:
                rows.append(dict(task=futs[fu], error=repr(e)[:300]))
            if (i + 1) % log_every == 0: print(f'   G {i + 1}/{len(task_ids)}  {time.time() - t0:.0f}s', flush=True)
    return pd.DataFrame(rows)

def collect_g():
    rows = []
    for f in sorted(P.TD.glob('*/G.json')):
        g = json.load(open(f)); tid = f.parent.name
        for m in g['mutants']:
            if 'skip' in m or m.get('timeout'): continue
            ds = [v for v in m['depth'].values() if v]
            rows.append(dict(task=tid, repo=tid.split('_')[0], site=m['site'], cls=m['cls'], op=m['op'], n_cover=m['n_cover'],
                             n_sel=m['n_sel'], b=len(m['killed']), killed=m['n_sel'] > 0 and len(m['killed']) > 0,
                             n_missing=m.get('n_missing', 0), in_tb=sum(v[0] >= 0 for v in ds), n_tb=len(ds),
                             depth_pkg=[v[1] for v in ds if v[1] >= 0]))
    return pd.DataFrame(rows)

def report_g(M, R):
    err = R[R['error'].notna()] if 'error' in R else R.iloc[:0]
    print(f'tasks: {len(R)}  errors: {len(err)}')
    for _, r in err.iterrows(): print('   ERR', r.task, str(r.error)[:160])
    if 'fid2' in R:
        F = R[R.fid2.notna()]
        print(f"\nFIDELITY RECHECK (same code + tests, traced vs untraced): {(F.fid2 == 0).sum()}/{len(F)} tasks identical, "
              f"total mismatches {int(F.fid2.sum())}")
        for _, r in F[F.fid2 > 0].iterrows(): print(f'   {r.task}: {int(r.fid2)} {r.fid2_list[:3]}')
    if 'ctrl_bad' in R:
        Cq = R[R.ctrl_bad.notna()]
        print(f"\nIDENTITY CONTROL (selection run with no mutation): {(Cq.ctrl_bad == 0).sum()}/{len(Cq)} tasks clean, "
              f"{int(Cq.ctrl_bad.sum())} unstable tests excluded in total")
        for _, r in Cq[Cq.ctrl_bad > 0].head(8).iterrows(): print(f'   {r.task}: {int(r.ctrl_bad)} {r.ctrl_bad_list[:2]}')
    if M.empty: return
    print('missing-test rows by repo:', M.groupby('repo').n_missing.sum().to_dict())
    print(f'\nMUTANTS run: {len(M)} on {M.task.nunique()} tasks | missing-test rows: {int(M.n_missing.sum())}')
    print('by site class (b = number of selected tests the mutant breaks):')
    g = M.groupby('cls').agg(n=('b', 'size'), kill_rate=('killed', 'mean'), b_median=('b', 'median'), b_mean=('b', 'mean'),
                             cover_median=('n_cover', 'median'))
    print(g.reindex(['Fstar', 'caller', 'callee', 'd2', 'rand']).round(3).to_string())
    print('\nby repo:'); print(M.groupby('repo').agg(n=('b', 'size'), kill_rate=('killed', 'mean'), b_median=('b', 'median')).round(3).to_string())
    print('\nby operator:'); print(M.groupby('op').agg(n=('b', 'size'), kill_rate=('killed', 'mean')).round(3).to_string())
    K = M[M.killed]
    tb = K.in_tb.sum() / max(K.n_tb.sum(), 1)
    dep = Counter(d for l in K.depth_pkg for d in l)
    print(f'\nfailures whose traceback contains the mutated frame: {tb:.2%}  (rest = wrong value propagated to an assertion)')
    print('package-frame depth from test to mutated frame:', dict(sorted(dep.items())[:12]))
    print('\ncovered-but-unkilled (silent) mutants:', int(((M.n_cover > 0) & ~M.killed).sum()), 'of', int((M.n_cover > 0).sum()))

def retry_step8(tid, timeout=3600):
    """re-run Step 8 for one task with a longer per-suite timeout"""
    orig = P.suite_run
    P.suite_run = lambda repo, w, tag, traced=False, timeout=timeout, targets=(): orig(repo, w, tag, traced=traced, timeout=timeout, targets=targets)
    try:
        return P.task_run(P.T[tid], json.load(open(P.RD / f'{tid}.json')).get('f2p', []))
    finally:
        P.suite_run = orig


def archive_g(repos):
    """move existing G.json of the given repos aside (G_v1.json) so they are recomputed"""
    n = 0
    for r in repos:
        for f in P.TD.glob(f'{r}_*/G.json'):
            f.rename(f.with_name('G_v1.json')); n += 1
    return n
