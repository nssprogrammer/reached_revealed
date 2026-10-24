# ---- Step 10b: coverage-free targets, within-repo task level, and the depth profile of detection (CPU, minutes) ----
# Standalone: reads the 'propagator' folder (attached input or /kaggle/working) and Step 10's tables in /kaggle/working/step10.
import os, glob, json, time, warnings
from collections import defaultdict
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as ssp
from scipy.sparse.csgraph import shortest_path
from scipy.stats import spearmanr, wilcoxon
warnings.filterwarnings('ignore')

def find_out():
    for base in (os.environ.get('PROP_WORKDIR', '/kaggle/working'), '/kaggle/input'):
        for h in sorted(glob.glob(f'{base}/**/propagator/tasks', recursive=True)):
            if glob.glob(f'{h}/*/G.json'): return Path(h).parent
    raise FileNotFoundError('attach the end-to-end output (propagator/tasks/*/G.json)')
OUT = find_out(); TD = OUT / 'tasks'
SAVE = Path(os.environ.get('PROP_WORKDIR', '/kaggle/working')) / 'step10'; SAVE.mkdir(parents=True, exist_ok=True)
SENSITIVE = {'tests.test_syntax::test_from_path', 'tests.test_syntax::test_syntax_guess_lexer'}
print('using', OUT)

# ------------------------------------------------------------ A. coverage-free blast radius targets
S = pd.read_csv(SAVE / 'step10_sites.csv') if (SAVE / 'step10_sites.csv').exists() else None
PRED = ['n_cover', 'indeg', 'degree', 'pagerank', 'betweenness', 'ancestors', 'Katz_1.0', 'Gamma_1.0']
def within(S, target, preds, min_sites=6):
    rows, rng = [], np.random.default_rng(0)
    for p in preds:
        v = []
        for _, g in S.groupby('task'):
            if len(g) < min_sites or g[target].nunique() < 3 or g[p].nunique() < 2: continue
            v.append(spearmanr(g[p], g[target]).correlation)
        v = np.array([x for x in v if np.isfinite(x)])
        if len(v):
            bs = [rng.choice(v, len(v)).mean() for _ in range(2000)]
            rows.append(dict(predictor=p, tasks=len(v), mean_rho=v.mean(), ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5)))
    return pd.DataFrame(rows).sort_values('mean_rho', ascending=False) if rows else pd.DataFrame(columns=['predictor', 'tasks', 'mean_rho'])
if S is not None:
    print('\nA) bhat multiplies by n_cover, so part of coverage\'s lead is mechanical. Coverage-free targets:')
    print('   target = killed FRACTION of the covering tests (how strongly a covering test observes the fault)')
    print(within(S, 'frac', PRED).round(3).to_string(index=False))
    print('   target = RAW number of broken tests among those run (selection capped at 300)')
    print(within(S, 'b', PRED).round(3).to_string(index=False))
    Gt = pd.read_csv(SAVE / 'step10_graphs.csv')
    print('\n   task level WITHIN repo (Spearman vs mean blast radius; tasks of one repo share most of their graph):')
    for repo, g in Gt.groupby('repo'):
        if len(g) >= 8:
            print(f"   {repo:8s} n={len(g):3d}  amp_1.0 {spearmanr(g['amp_1.0'], g['mean_bhat']).correlation:+.3f}  "
                  f"departure {spearmanr(g['departure'], g['mean_bhat']).correlation:+.3f}  size m {spearmanr(g['m'], g['mean_bhat']).correlation:+.3f}")

# ------------------------------------------------------------ B. the depth profile: does a covering test's chance of failing decay with call depth?
def roots_per_test(C, R):
    """functions a test executes that none of the test's other executed functions call (its entry points into the package)"""
    out = []
    for i in range(C.shape[0]):
        S_ = C.indices[C.indptr[i]:C.indptr[i + 1]]
        if len(S_) == 0: out.append(S_); continue
        has_caller = np.asarray(R[S_][:, S_].sum(axis=0)).ravel() > 0
        out.append(S_[~has_caller])
    return out

t0 = time.time(); pairs = []
for d in sorted(p.parent for p in TD.glob('*/G.json')):
    g = json.load(open(d / 'G.json'))
    if g.get('summary', {}).get('version') != 2: continue
    idx = json.load(open(d / 'index.json')); keys, funcs = idx['tests'], idx['funcs']; n = len(funcs)
    fi = {f: i for i, f in enumerate(funcs)}
    E = np.array(idx['edges'], dtype=np.int64).reshape(-1, 2)
    R = ssp.csr_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(n, n)); R.data[:] = 1
    C = ssp.load_npz(d / 'C.npz').tocsr(); Cc = C.tocsc(); roots = roots_per_test(C, R); RT = R.T.tocsr()
    for m in g['mutants']:
        if 'skip' in m or m.get('timeout') or m['n_sel'] == 0 or m['n_cover'] > 300 or m['site'] not in fi: continue
        v = fi[m['site']]; cover = Cc[:, v].nonzero()[0]
        if len(cover) == 0 or m['n_sel'] < 0.8 * len(cover): continue          # selection must be (almost) the full cover
        dist = shortest_path(RT, indices=v, unweighted=True, directed=True)    # hops from v up to each caller
        killed = set(m['killed'])
        for r in cover:
            if keys[r] in SENSITIVE: continue
            rt = roots[r]; dep = dist[rt].min() if len(rt) else np.inf
            pairs.append((d.name, d.name.split('_')[0], m['cls'], m['op'], m['site'], m['line'], float(dep), r in killed))
P = pd.DataFrame(pairs, columns=['task', 'repo', 'cls', 'op', 'site', 'line', 'depth', 'killed'])
P['mutant'] = P.task + '|' + P.site + '|' + P.line.astype(str) + '|' + P.op
P.to_csv(SAVE / 'step10b_pairs.csv', index=False)
print(f'\nB) DEPTH PROFILE: {len(P)} (mutant, covering test) pairs from {P.mutant.nunique()} uncapped mutants in {time.time() - t0:.0f}s')
P['dbin'] = P.depth.where(np.isfinite(P.depth), -1).clip(upper=8).astype(int)
tab = P.groupby('dbin').agg(pairs=('killed', 'size'), p_fail=('killed', 'mean'))
tab.index = [('unreached' if i < 0 else ('8+' if i == 8 else str(i))) for i in tab.index]
print('   P(covering test fails | hops from the test\'s entry point down to the mutated function):')
print(tab.round(3).to_string())
for repo, q in P[np.isfinite(P.depth)].groupby('repo'):
    if len(q) > 500:
        t = q.groupby(q.depth.clip(upper=8)).killed.mean()
        print(f'   {repo:8s}', ' '.join(f'{int(k)}:{v:.2f}' for k, v in t.items()))

# within-mutant test: among the covering tests of the SAME mutant, are the failing ones closer to it?
diffs = []
for mid, q in P[np.isfinite(P.depth)].groupby('mutant'):
    if q.killed.any() and (~q.killed).any():
        diffs.append(q.loc[q.killed, 'depth'].mean() - q.loc[~q.killed, 'depth'].mean())
diffs = np.array(diffs)
if len(diffs) > 10:
    print(f'\n   within-mutant: {len(diffs)} mutants with both failing and passing covering tests')
    print(f'   mean depth(failing) - depth(passing) = {diffs.mean():+.3f} hops; failing tests are closer in {(diffs < 0).mean():.0%} '
          f'of mutants; Wilcoxon p = {wilcoxon(diffs).pvalue:.2e}')
print('\n   by site class, P(fail) among covering tests:', P.groupby('cls').killed.mean().round(3).to_dict())
