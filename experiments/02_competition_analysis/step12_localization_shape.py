# ---- Step 12: is there a localization "light cone"? front vs smooth decay, and does the front move with reach? (CPU, minutes) ----
# Inputs: end-to-end output (propagator/...) and the competition data (problem statements). Agent-free: the localizer is
# Ochiai on the failing tests' coverage, diffused over the call graph for k hops (or personalized PageRank with damping alpha).
import os, glob, json, re, time, warnings
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as ssp
from scipy.sparse.csgraph import shortest_path
from scipy.sparse.linalg import spsolve
from scipy.optimize import minimize
from scipy.stats import spearmanr
warnings.filterwarnings('ignore')

def find_out():
    for base in (os.environ.get('PROP_WORKDIR', '/kaggle/working'), '/kaggle/input'):
        for h in sorted(glob.glob(f'{base}/**/propagator/tasks', recursive=True)):
            if glob.glob(f'{h}/*/G.json'): return Path(h).parent
    raise FileNotFoundError('attach the end-to-end output (propagator/tasks/*/G.json)')
OUT = find_out(); TD, RD = OUT / 'tasks', OUT / 'repro'
SAVE = Path(os.environ.get('PROP_WORKDIR', '/kaggle/working')) / 'step12'; SAVE.mkdir(parents=True, exist_ok=True)
TASKS = {}
for f in sorted(glob.glob(os.environ.get('PROP_DATA', '/kaggle/input') + '/**/*.jsonl', recursive=True)):
    with open(f) as fh: first = fh.readline()
    if '"instance_id"' in first and '"patch"' in first:
        TASKS = {json.loads(l)['instance_id']: json.loads(l) for l in open(f) if l.strip()}; break
SENSITIVE = {'tests.test_syntax::test_from_path', 'tests.test_syntax::test_syntax_guess_lexer'}
IDENT = re.compile(r'[A-Za-z_][A-Za-z0-9_]{2,}')
def mentioned(func, text):
    toks = set(IDENT.findall(text or '')); p = func.split('.')
    return p[-1] in toks or (len(p) > 1 and p[-2] in toks and p[-1] != '<module>')

def ochiai(C, fail):
    ef = np.asarray(C[fail].sum(axis=0)).ravel(); tot = np.asarray(C.sum(axis=0)).ravel()
    den = np.sqrt(len(fail) * tot); den[den == 0] = 1; return ef / den
def best_rank(s, targets):
    order = (-s).argsort(kind='stable'); pos = np.empty(len(s), int); pos[order] = np.arange(1, len(s) + 1)
    return int(min(pos[t] for t in targets))

KS = [0, 1, 2, 4, 8, 16]; ALPHAS = [0.3, 0.5, 0.7, 0.85, 0.95]; TOP = 5
rows, t0 = [], time.time()
for d in sorted(p.parent for p in TD.glob('*/G.json')):
    tid = d.name
    try:
        idx = json.load(open(d / 'index.json')); keys, funcs = idx['tests'], idx['funcs']; fi = {f: i for i, f in enumerate(funcs)}
        n = len(funcs); E = np.array(idx['edges'], dtype=np.int64).reshape(-1, 2)
        R = ssp.csr_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(n, n)); R.data[:] = 1
        C = ssp.load_npz(d / 'C.npz').tocsr().astype(float)
        f2p = set(json.load(open(RD / f'{tid}.json')).get('f2p', []))
        fail = [i for i, k in enumerate(keys) if (k in f2p or k.split('::')[-1].split('[')[0] in f2p) and k not in SENSITIVE]
        g = json.load(open(d / 'G.json')); F = sorted({m['site'] for m in g['mutants'] if m['cls'] == 'Fstar'} & set(fi))
        F_idx = [fi[f] for f in F]
        if not fail or not F_idx: continue
        Au = ((R + R.T) > 0).astype(float); deg = np.asarray(Au.sum(axis=1)).ravel(); deg[deg == 0] = 1
        PT = (ssp.diags(1 / deg) @ Au).T.tocsr()                              # random-walk transition, transposed
        roots = set()
        for i in fail:
            S_ = C.indices[C.indptr[i]:C.indptr[i + 1]]
            if len(S_): roots |= set(S_[~(np.asarray(R[S_][:, S_].sum(axis=0)).ravel() > 0)].tolist())
        dist = shortest_path(Au, indices=F_idx, unweighted=True, directed=False)
        d_sf = float(dist[:, sorted(roots)].min()) if roots else np.inf
        s0 = ochiai(C, fail); row = dict(task=tid, repo=tid.split('_')[0], d=d_sf, n_funcs=n, nF=len(F_idx), n_fail=len(fail),
                                         mentioned=any(mentioned(funcs[v], TASKS.get(tid, {}).get('problem_statement', '')) for v in F_idx),
                                         indeg_F=float(np.asarray(R[:, F_idx].sum(axis=0)).max()))
        acc, cur = s0.copy(), s0.copy()                                       # k-round diffusion: sum of the first k hops
        for k in range(max(KS) + 1):
            if k in KS: row[f'rank_k{k}'] = best_rank(acc, F_idx)
            cur = PT @ cur; acc = acc + cur
        sn = s0 / s0.sum() if s0.sum() > 0 else s0
        for a in ALPHAS:
            row[f'rank_a{a}'] = best_rank((1 - a) * spsolve((ssp.identity(n) - a * PT).tocsc(), sn), F_idx)
        rows.append(row)
    except Exception as e:
        print('skip', tid, repr(e)[:120])
L = pd.DataFrame(rows); L.to_csv(SAVE / 'step12_tasks.csv', index=False)
fin = L[np.isfinite(L.d)].copy()
print(f'using {OUT} | {len(L)} tasks ({len(fin)} with finite symptom-to-fix distance) in {time.time() - t0:.0f}s')

# ------------------------------------------------------------ shape: a front (step) or a smooth decay (logistic)?
def nll_logit(w, x, y):
    z = w[0] + w[1] * x; p = 1 / (1 + np.exp(-z)); p = p.clip(1e-9, 1 - 1e-9)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p)).sum()
def fit_shapes(x, y):
    lg = minimize(nll_logit, [0.0, -0.5], args=(x, y), method='Nelder-Mead')
    best = (np.inf, None)
    for tau in np.unique(x)[:-1]:                                             # step: P = p1 if d <= tau else p2
        a, b = y[x <= tau], y[x > tau]
        p1, p2 = a.mean().clip(1e-9, 1 - 1e-9), b.mean().clip(1e-9, 1 - 1e-9)
        nll = -(a * np.log(p1) + (1 - a) * np.log(1 - p1)).sum() - (b * np.log(p2) + (1 - b) * np.log(1 - p2)).sum()
        if nll < best[0]: best = (nll, (tau, a.mean(), b.mean()))
    aic_l, aic_s = 2 * 2 + 2 * lg.fun, 2 * 3 + 2 * best[0]
    return dict(tau=best[1][0] if best[1] else np.nan, p_in=best[1][1] if best[1] else np.nan, p_out=best[1][2] if best[1] else np.nan,
                dAIC=aic_l - aic_s, slope=lg.x[1])

print('\n1) SHAPE of top-5 localization vs distance d, per localizer reach')
print('   dAIC > 2 favours a FRONT (step at tau) over smooth logistic decay; p_front = share of 300 label shuffles with a dAIC at least as large')
out = []
for col in [f'rank_k{k}' for k in KS] + [f'rank_a{a}' for a in ALPHAS]:
    for label, q in (('all', fin), ('unmentioned', fin[~fin.mentioned])):
        x, y = q.d.values.astype(float), (q[col] <= TOP).astype(float).values
        if len(q) < 15 or y.min() == y.max(): continue
        r = fit_shapes(x, y); rng_p = np.random.default_rng(1)
        null = [fit_shapes(rng_p.permutation(x), y)['dAIC'] for _ in range(300)]       # how often a shuffled d gives as clean a front
        r.update(localizer=col.replace('rank_', ''), subset=label, n=len(q), hit=y.mean(),
                 rho=spearmanr(q.d, q[col]).correlation, p_front=(np.array(null) >= r['dAIC']).mean()); out.append(r)
Sh = pd.DataFrame(out, columns=['localizer', 'subset', 'n', 'hit', 'rho', 'tau', 'p_in', 'p_out', 'dAIC', 'p_front', 'slope'])
print(Sh.round(3).to_string(index=False)); Sh.to_csv(SAVE / 'step12_shapes.csv', index=False)

# ------------------------------------------------------------ velocity: does the knee move with the number of diffusion rounds k?
print('\n2) VELOCITY: knee position tau (best step threshold) against diffusion rounds k, unmentioned subset')
v = Sh[(Sh.subset == 'unmentioned') & Sh.localizer.str.startswith('k')]
for _, r in v.iterrows(): print(f"   k = {r.localizer[1:]:>2s}  knee tau = {r.tau:.0f} hops  top-5 inside {r.p_in:.2f} / outside {r.p_out:.2f}  dAIC {r.dAIC:+.1f}  p_front {r.p_front:.3f}")

# ------------------------------------------------------------ controls: does d survive size/centrality, and a label shuffle?
def rres(y, X):
    ry = pd.Series(y).rank().values; Xr = np.column_stack([pd.Series(c).rank().values for c in X.T] + [np.ones(len(y))])
    return ry - Xr @ np.linalg.lstsq(Xr, ry, rcond=None)[0]
col = 'rank_a0.85'
if len(fin) < 10: raise SystemExit('too few tasks for controls')
X = fin[['n_funcs', 'nF', 'n_fail', 'indeg_F']].values.astype(float)
pr = spearmanr(rres(fin.d.values, X), rres(fin[col].values, X)).correlation
rng = np.random.default_rng(0); obs = spearmanr(fin.d, fin[col]).correlation
null = [spearmanr(rng.permutation(fin.d.values), fin[col]).correlation for _ in range(2000)]
print(f'\n3) CONTROLS ({col}): Spearman(d, rank) = {obs:+.3f}; partial, given graph size, |F*|, #failing tests, in-degree of F*: {pr:+.3f}; '
      f'label-shuffle p = {(np.abs(null) >= abs(obs)).mean():.4f}')
for c in ('n_funcs', 'nF', 'n_fail', 'indeg_F'):
    print(f'   {c:8s} Spearman with rank {spearmanr(fin[c], fin[col]).correlation:+.3f}')
print('\n   top-5 rate by localizer:', {c.replace('rank_', ''): f"{(L[c] <= TOP).mean():.0%}" for c in L.columns if c.startswith('rank_')})
