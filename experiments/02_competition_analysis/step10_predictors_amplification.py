# ---- Step 10: does the propagator's transient structure predict the measured blast radius? (no tests run; minutes) ----
# Standalone: needs only the 'propagator' output folder of the end-to-end notebook (in /kaggle/working, or attached via Add Input).
import os, glob, json, math, time, warnings
from collections import defaultdict
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as ssp, networkx as nx
from scipy.sparse.linalg import expm_multiply, spsolve
from scipy.sparse.csgraph import connected_components, breadth_first_order
from scipy.stats import spearmanr, mannwhitneyu
warnings.filterwarnings('ignore')

def find_out():
    for base in (os.environ.get('PROP_WORKDIR', '/kaggle/working'), '/kaggle/input'):
        hits = sorted(glob.glob(f'{base}/**/propagator/tasks', recursive=True)) or \
               ([f'{base}/propagator/tasks'] if os.path.isdir(f'{base}/propagator/tasks') else [])
        for h in hits:
            if glob.glob(f'{h}/*/G.json'): return Path(h).parent
    raise FileNotFoundError('No propagator/tasks/*/G.json found: run the end-to-end notebook first or attach its output.')
OUT = find_out(); TD = OUT / 'tasks'; print('using', OUT)
SAVE = Path(os.environ.get('PROP_WORKDIR', '/kaggle/working')) / 'step10'; SAVE.mkdir(parents=True, exist_ok=True)   # inputs are read-only
_vers = [json.load(open(f)).get('summary', {}).get('version') for f in sorted(TD.glob('*/G.json'))]
if sum(v == 2 for v in _vers) < 0.9 * len(_vers):
    raise RuntimeError(f'{sum(v != 2 for v in _vers)} of {len(_vers)} G.json files are from the OLD (buggy) Step 9. '
                       'Attach the output of the committed end-to-end run instead of the older dataset.')
GAMMA0 = (0.5, 1.0, 2.0)
SENSITIVE = {'tests.test_syntax::test_from_path', 'tests.test_syntax::test_syntax_guess_lexer'}

# ------------------------------------------------------------ graph objects
def load_graph(d):
    idx = json.load(open(d / 'index.json')); n = len(idx['funcs'])
    E = np.array(idx['edges'], dtype=np.int64).reshape(-1, 2)
    R = ssp.csr_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(n, n)) if len(E) else ssp.csr_matrix((n, n))
    R.data[:] = 1.0                       # R[u, v] = 1 if u calls v; a fault at v reaches its callers: R e_v
    return idx, R

def census(R):
    n = R.shape[0]; m = R.nnz
    recip = R.multiply(R.T).nnz / max(m, 1)
    ncomp, lab = connected_components(R, directed=True, connection='strong')
    sizes = np.bincount(lab); big = np.where(sizes > 1)[0]
    rho, eig2 = 0.0, 0.0
    for c in big:                          # spectrum of a near-DAG = union of its strongly connected blocks
        ix = np.where(lab == c)[0]; ev = np.linalg.eigvals(R[ix][:, ix].toarray())
        rho = max(rho, float(np.abs(ev).max())); eig2 += float((np.abs(ev) ** 2).sum())
    fro2 = float(m); dep = math.sqrt(max(fro2 - eig2, 0)) / math.sqrt(fro2) if m else 0.0
    return dict(n=n, m=m, reciprocity=recip, frac_in_cycles=float(sizes[big].sum()) / n if n else 0.0,
                largest_scc=int(sizes.max()) if n else 0, rho=rho, departure=dep)

def tgrid(gamma, num=60): return np.linspace(0, 12.0 / gamma, num)

def transient_sites(R, sites, gamma):
    """|| e^{t(R - gamma I)} e_v ||_1 on a time grid for each site v (all entries are non-negative)"""
    n = R.shape[0]; M = (R - gamma * ssp.identity(n, format='csr')).tocsc(); t = tgrid(gamma)
    B = np.zeros((n, len(sites))); B[sites, np.arange(len(sites))] = 1.0
    Y = expm_multiply(M, B, start=t[0], stop=t[-1], num=len(t), endpoint=True)      # (T, n, k)
    norms = np.abs(Y).sum(axis=1)                                                    # (T, k)
    return t, norms

def graph_amplification(R, gamma):
    """|| e^{tM} ||_1 = max column sum (non-negative operator): the whole graph's transient growth"""
    n = R.shape[0]; M = (R.T - gamma * ssp.identity(n, format='csr')).tocsc(); t = tgrid(gamma)
    Y = expm_multiply(M, np.ones(n), start=t[0], stop=t[-1], num=len(t), endpoint=True)  # rows = column sums of e^{tM}
    eta = Y.max(axis=1); k = int(eta.argmax())
    return float(eta[k]), float(t[k])

# ------------------------------------------------------------ per task: measured blast radius + predictors per site
def task_rows(d):
    g = json.load(open(d / 'G.json')); idx, R = load_graph(d); funcs = idx['funcs']; fi = {f: i for i, f in enumerate(funcs)}
    keys = idx['tests']; n = len(funcs)
    ms = [m for m in g['mutants'] if 'skip' not in m and not m.get('timeout') and m['n_sel'] > 0]
    if not ms: return None, None
    per = defaultdict(list)
    for m in ms:
        kill = [r for r in m['killed'] if keys[r] not in SENSITIVE]
        frac = len(kill) / m['n_sel']
        per[m['site']].append(dict(b=len(kill), frac=frac, bhat=frac * m['n_cover'], killed=len(kill) > 0,
                                   n_cover=m['n_cover'], cls=m['cls']))
    sites = [s for s in per if s in fi]; six = np.array([fi[s] for s in sites])
    if len(six) == 0: return None, None
    c = census(R); rho = c['rho']
    G = nx.DiGraph(); G.add_nodes_from(range(n)); G.add_edges_from(map(tuple, idx['edges']))
    pr = nx.pagerank(G, alpha=0.85) if G.number_of_edges() else {i: 1 / n for i in range(n)}
    bt = nx.betweenness_centrality(G, k=min(200, n), seed=0) if G.number_of_edges() else {i: 0 for i in range(n)}
    indeg = np.asarray(R.sum(axis=0)).ravel(); outdeg = np.asarray(R.sum(axis=1)).ravel()
    RT = R.T.tocsr()
    anc = np.array([len(breadth_first_order(RT, int(v), directed=True, return_predecessors=False)) - 1 for v in six])
    pred = {}
    for g0 in GAMMA0:
        gam = rho + g0; t, norms = transient_sites(R, list(six), gam)
        pred[f'Gamma_{g0}'] = norms.max(axis=0); pred[f'tstar_{g0}'] = t[norms.argmax(axis=0)]
        A = (gam * ssp.identity(n, format='csc') - R.tocsc())
        pred[f'Katz_{g0}'] = np.array([gam * spsolve(A, np.eye(1, n, int(v)).ravel()).sum() for v in six])
        amp, tpk = graph_amplification(R, gam); c[f'amp_{g0}'] = amp; c[f'tpeak_{g0}'] = tpk
    rows = []
    for j, s in enumerate(sites):
        L = per[s]; v = int(six[j])
        rows.append(dict(task=d.name, repo=d.name.split('_')[0], site=s, cls=L[0]['cls'], n_mut=len(L),
                         b=np.mean([x['b'] for x in L]), frac=np.mean([x['frac'] for x in L]), bhat=np.mean([x['bhat'] for x in L]),
                         killed=np.mean([x['killed'] for x in L]), n_cover=L[0]['n_cover'],
                         indeg=indeg[v], outdeg=outdeg[v], degree=indeg[v] + outdeg[v], pagerank=pr[v], betweenness=bt[v],
                         ancestors=anc[j], **{k: float(a[j]) for k, a in pred.items()}))
    c.update(task=d.name, repo=d.name.split('_')[0], n_sites=len(rows),
             mean_bhat=float(np.mean([r['bhat'] for r in rows])), kill_rate=float(np.mean([x['killed'] for L in per.values() for x in L])))
    return rows, c

t0 = time.time(); site_rows, graph_rows = [], []
for d in sorted(p.parent for p in TD.glob('*/G.json')):
    try:
        r, c = task_rows(d)
    except Exception as e:
        print('skip', d.name, repr(e)[:120]); continue
    if r: site_rows += r; graph_rows.append(c)
S = pd.DataFrame(site_rows); Gt = pd.DataFrame(graph_rows)
S.to_csv(SAVE / 'step10_sites.csv', index=False); Gt.to_csv(SAVE / 'step10_graphs.csv', index=False)
print(f'{len(S)} sites on {S.task.nunique()} tasks in {time.time() - t0:.0f}s')

# ------------------------------------------------------------ 1. the eigenvalue view vs the transient view (Part I premise)
print('\n1) RUNTIME CALL GRAPHS (per repo, median over tasks)')
print(Gt.groupby('repo')[['n', 'm', 'reciprocity', 'frac_in_cycles', 'largest_scc', 'rho', 'departure', 'amp_1.0', 'tpeak_1.0']]
      .median().round(3).to_string())
print('   eigenvalues predict ||e^{tM}|| <= 1 and decaying for all t (rho - gamma < 0); measured peak amplification '
      f"||e^(tM)||_1 at gamma = rho + 1: median {Gt['amp_1.0'].median():.2f}, max {Gt['amp_1.0'].max():.1f}")

# ------------------------------------------------------------ 2. which predictor ranks sites by measured blast radius?
PRED = ['n_cover', 'indeg', 'degree', 'pagerank', 'betweenness', 'ancestors', 'Katz_1.0', 'Gamma_0.5', 'Gamma_1.0', 'Gamma_2.0', 'Katz_0.5', 'Katz_2.0']
GRAPH_ONLY = [p for p in PRED if p != 'n_cover']

def rank_resid(y, x):
    ry, rx = pd.Series(y).rank().values, pd.Series(x).rank().values
    if np.std(rx) == 0: return ry - ry.mean()
    b = np.polyfit(rx, ry, 1); return ry - np.polyval(b, rx)

def within_task(S, target, preds, control=None, min_sites=6, min_unique=3):
    out = {p: [] for p in preds}
    for _, g in S.groupby('task'):
        if len(g) < min_sites or g[target].nunique() < min_unique: continue
        y = rank_resid(g[target], g[control]) if control else g[target].values
        for p in preds:
            x = rank_resid(g[p], g[control]) if control else g[p].values
            if np.std(x) == 0 or np.std(y) == 0: continue
            out[p].append(spearmanr(x, y).correlation)
    rng = np.random.default_rng(0); rows = []
    for p, v in out.items():
        v = np.array([x for x in v if np.isfinite(x)])
        if len(v) == 0: continue
        bs = [rng.choice(v, len(v)).mean() for _ in range(2000)]
        rows.append(dict(predictor=p, tasks=len(v), mean_rho=v.mean(), ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5),
                         frac_pos=(v > 0).mean()))
    return pd.DataFrame(rows).sort_values('mean_rho', ascending=False) if rows else pd.DataFrame(columns=['predictor', 'tasks', 'mean_rho'])

def pooled_auc(S, target, preds, q=0.75):
    rows = []
    for p in preds:
        pos, neg = [], []
        for _, g in S.groupby('task'):
            if len(g) < 6 or g[target].nunique() < 3: continue
            thr = g[target].quantile(q); x = g[p].rank(pct=True).values
            pos += list(x[g[target].values >= thr]); neg += list(x[g[target].values < thr])
        if pos and neg:
            u = mannwhitneyu(pos, neg).statistic; rows.append(dict(predictor=p, auc=u / (len(pos) * len(neg))))
    return pd.DataFrame(rows).sort_values('auc', ascending=False) if rows else pd.DataFrame(columns=['predictor', 'auc'])

print('\n2) WITHIN-TASK SPEARMAN: predictor vs measured blast radius (bhat = killed fraction x tests covering the site)')
print(within_task(S, 'bhat', PRED).round(3).to_string(index=False))
print('\n   same, CONTROLLING FOR COVERAGE (partial rank correlation given n_cover): does graph structure add anything?')
print(within_task(S, 'bhat', GRAPH_ONLY, control='n_cover').round(3).to_string(index=False))
print('\n   AUC for the top-quartile blast-radius sites within each task (pooled):')
print(pooled_auc(S, 'bhat', PRED).round(3).to_string(index=False))
print('\n   killed-at-all (any covering test fails) vs predictors, within-task Spearman:')
print(within_task(S, 'killed', ['n_cover', 'ancestors', 'pagerank', 'Gamma_1.0', 'Katz_1.0'], min_unique=2).round(3).to_string(index=False))

# ------------------------------------------------------------ 3. silent displacement among covered mutants
cov = S[S.n_cover > 0]
print(f"\n3) SILENT (covered but never killed) sites: {(cov.killed == 0).sum()}/{len(cov)}")
for p in ['n_cover', 'ancestors', 'Gamma_1.0', 'pagerank']:
    a, b = cov.loc[cov.killed == 0, p], cov.loc[cov.killed > 0, p]
    if len(a) and len(b): print(f'   {p:12s} median silent {a.median():9.3f}  vs killed {b.median():9.3f}  AUC(killed) {1 - mannwhitneyu(a, b).statistic / (len(a) * len(b)):.3f}')

# ------------------------------------------------------------ 4. task level: does graph-wide amplification track mean blast radius?
print('\n4) TASK LEVEL (Spearman across tasks)')
for p in ['amp_0.5', 'amp_1.0', 'amp_2.0', 'departure', 'rho', 'n', 'm']:
    r = spearmanr(Gt[p], Gt['mean_bhat']).correlation; k = spearmanr(Gt[p], Gt['kill_rate']).correlation
    print(f'   {p:10s} vs mean blast radius {r:+.3f} | vs kill rate {k:+.3f}')
print('\nby site class (median measured blast radius):'); print(S.groupby('cls')[['bhat', 'frac', 'killed', 'n_cover']].median().round(3).to_string())
