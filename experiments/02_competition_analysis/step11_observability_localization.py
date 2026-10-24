# ---- Step 11: Part III (localization difficulty and ranking fragility) + fix-site observability (CPU, minutes) ----
# Inputs (Add Input): the end-to-end output (propagator/...) AND the competition data (for problem statements).
import os, glob, json, re, time, warnings
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as ssp
from scipy.sparse.csgraph import shortest_path
from scipy.sparse.linalg import spsolve
from scipy.stats import spearmanr, wilcoxon, mannwhitneyu
warnings.filterwarnings('ignore')

def find_out():
    for base in (os.environ.get('PROP_WORKDIR', '/kaggle/working'), '/kaggle/input'):
        for h in sorted(glob.glob(f'{base}/**/propagator/tasks', recursive=True)):
            if glob.glob(f'{h}/*/G.json'): return Path(h).parent
    raise FileNotFoundError('attach the end-to-end output (propagator/tasks/*/G.json)')
OUT = find_out(); TD, RD = OUT / 'tasks', OUT / 'repro'
SAVE = Path(os.environ.get('PROP_WORKDIR', '/kaggle/working')) / 'step11'; SAVE.mkdir(parents=True, exist_ok=True)
TASKS = {}
for f in sorted(glob.glob(os.environ.get('PROP_DATA', '/kaggle/input') + '/**/*.jsonl', recursive=True)):
    with open(f) as fh: first = fh.readline()
    if '"instance_id"' in first and '"patch"' in first:
        TASKS = {json.loads(l)['instance_id']: json.loads(l) for l in open(f) if l.strip()}; break
print('using', OUT, '| tasks with problem statements:', sum('problem_statement' in t for t in TASKS.values()))
SENSITIVE = {'tests.test_syntax::test_from_path', 'tests.test_syntax::test_syntax_guess_lexer'}

# ------------------------------------------------------------ A. are faults at the real fix site less observable?
S10 = Path(os.environ.get('PROP_WORKDIR', '/kaggle/working')) / 'step10' / 'step10_sites.csv'
if S10.exists():
    S = pd.read_csv(S10); pairs = []
    for tid, g in S.groupby('task'):
        f, o = g[g.cls == 'Fstar'], g[g.cls.isin(['caller', 'callee', 'd2'])]
        if len(f) and len(o): pairs.append((tid, f.frac.mean(), o.frac.mean(), f.n_cover.median(), o.n_cover.median()))
    A = pd.DataFrame(pairs, columns=['task', 'frac_F', 'frac_nbr', 'cover_F', 'cover_nbr'])
    d = A.frac_F - A.frac_nbr
    print(f"\nA) FIX-SITE OBSERVABILITY (paired within task, {len(A)} tasks): fraction of covering tests that catch a planted fault")
    print(f"   at F*: median {A.frac_F.median():.3f} | at its graph neighbours: median {A.frac_nbr.median():.3f} | "
          f"F* lower in {(d < 0).mean():.0%} of tasks, Wilcoxon p = {wilcoxon(d).pvalue:.2e}")
    print(f"   (coverage: F* median {A.cover_F.median():.0f} tests, neighbours {A.cover_nbr.median():.0f})")

# ------------------------------------------------------------ B. per-task localization from the failing tests (agent-free)
def graph(idx):
    n = len(idx['funcs']); E = np.array(idx['edges'], dtype=np.int64).reshape(-1, 2)
    R = ssp.csr_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(n, n)); R.data[:] = 1
    return R

def ppr(R, s, alpha=0.85):
    """personalized PageRank of a seed vector s on the symmetrised call graph (PRFL-style smoothing)"""
    Au = ((R + R.T) > 0).astype(float); deg = np.asarray(Au.sum(axis=1)).ravel(); deg[deg == 0] = 1
    P = ssp.diags(1 / deg) @ Au; s = s / s.sum() if s.sum() > 0 else s
    return (1 - alpha) * spsolve((ssp.identity(R.shape[0]) - alpha * P.T).tocsc(), s)

def ochiai(C, fail_rows):
    n_fail = len(fail_rows); ef = np.asarray(C[fail_rows].sum(axis=0)).ravel(); tot = np.asarray(C.sum(axis=0)).ravel()
    ep = tot - ef; den = np.sqrt(n_fail * (ef + ep)); den[den == 0] = 1
    return ef / den

def rank_of(score, targets):
    order = (-score).argsort(kind='stable'); pos = np.empty(len(score), int); pos[order] = np.arange(1, len(score) + 1)
    return int(min(pos[t] for t in targets)) if targets else None

IDENT = re.compile(r'[A-Za-z_][A-Za-z0-9_]{2,}')
def mentioned(func, text):
    toks = set(IDENT.findall(text or ''))
    parts = func.split('.'); return parts[-1] in toks or (len(parts) > 1 and parts[-2] in toks and parts[-1] != '<module>')

K = 5; NREP = 32; rows = []; t0 = time.time()
for d in sorted(p.parent for p in TD.glob('*/G.json')):
    tid = d.name; t = TASKS.get(tid, {})
    try:
        idx = json.load(open(d / 'index.json')); keys, funcs = idx['tests'], idx['funcs']; fi = {f: i for i, f in enumerate(funcs)}
        C = ssp.load_npz(d / 'C.npz').tocsr().astype(float); R = graph(idx); n = len(funcs)
        rec = json.load(open(RD / f'{tid}.json')); f2p = set(rec.get('f2p', []))
        fail = [i for i, k in enumerate(keys) if (k in f2p or k.split('::')[-1].split('[')[0] in f2p) and k not in SENSITIVE]
        # F* = the executed fix functions (the Fstar mutation sites recorded in Step 9)
        g = json.load(open(d / 'G.json')); Fset = sorted({m['site'] for m in g['mutants'] if m['cls'] == 'Fstar'} & set(fi))
        F_idx = [fi[f] for f in Fset]
        if not fail or not F_idx: continue
        s_och = ochiai(C, fail); s_ppr = ppr(R, s_och) if s_och.sum() > 0 else s_och
        r_och, r_ppr = rank_of(s_och, F_idx), rank_of(s_ppr, F_idx)
        # symptom-to-fix distance: hops (undirected) from the failing tests' entry functions to the nearest F*
        Au = ((R + R.T) > 0).astype(float)
        roots = set()
        for i in fail:
            Sx = C.indices[C.indptr[i]:C.indptr[i + 1]]
            if len(Sx): roots |= set(Sx[~(np.asarray(R[Sx][:, Sx].sum(axis=0)).ravel() > 0)].tolist())
        dist = shortest_path(Au, indices=F_idx, unweighted=True, directed=False)
        d_sf = float(dist[:, sorted(roots)].min()) if roots else np.inf
        # fragility of the top-K under resampling of the passing tests + 10% multiplicative noise
        rng = np.random.default_rng(0); pas = np.setdiff1d(np.arange(C.shape[0]), fail); tops, top1 = [], []
        base_top = set((-s_ppr).argsort()[:K].tolist())
        for _ in range(NREP):
            boot = rng.choice(pas, len(pas)) if len(pas) else pas
            Cb = ssp.vstack([C[fail], C[boot]]).tocsr()
            s = ochiai(Cb, list(range(len(fail)))) * (1 + 0.1 * rng.standard_normal(n)).clip(0)
            s = ppr(R, s) if s.sum() > 0 else s
            o = (-s).argsort(); tops.append(set(o[:K].tolist())); top1.append(int(o[0]))
        q = np.mean([len(a & b) / K for i, a in enumerate(tops) for b in tops[i + 1:]])
        frozen = set.intersection(*tops); sv = np.sort(s_ppr)[::-1]
        gap = (sv[K - 1] - sv[K]) / sv[0] if sv[0] > 0 and len(sv) > K else 0.0
        stab = np.mean([len(tp & base_top) / K for tp in tops])
        text = t.get('problem_statement', '')
        rows.append(dict(task=tid, repo=tid.split('_')[0], n_funcs=n, n_fail=len(fail), nF=len(F_idx),
                         rank_och=r_och, rank_ppr=r_ppr, d_sf=d_sf, mentioned=any(mentioned(funcs[v], text) for v in F_idx),
                         q_overlap=q, basins=len(set(top1)), frozen=len(frozen), F_in_frozen=bool(frozen & set(F_idx)),
                         gap_k=gap, stab_k=stab, files=len({f.rsplit('.', 1)[0] for f in Fset})))
    except Exception as e:
        print('skip', tid, repr(e)[:120])
L = pd.DataFrame(rows); L.to_csv(SAVE / 'step11_localization.csv', index=False)
print(f'\nB) LOCALIZATION from failing-test coverage (Ochiai, then PPR smoothing on the call graph): {len(L)} tasks in {time.time() - t0:.0f}s')
for col in ('rank_och', 'rank_ppr'):
    r = L[col]; print(f'   {col}: top-1 {(r <= 1).mean():.0%}  top-5 {(r <= 5).mean():.0%}  top-10 {(r <= 10).mean():.0%}  MRR {(1 / r).mean():.3f}  median rank {r.median():.0f}')
print('   by repo (top-5 after PPR):', L.groupby('repo').rank_ppr.apply(lambda r: f'{(r <= 5).mean():.0%} of {len(r)}').to_dict())

# ------------------------------------------------------------ C. symptom-to-fix distance, lexical mention, and the "light cone" shape
print(f"\nC) SYMPTOM-TO-FIX DISTANCE d (hops from failing tests' entry points to F*): median {L.d_sf[np.isfinite(L.d_sf)].median():.0f}, "
      f"unreached {(~np.isfinite(L.d_sf)).mean():.0%}; F* named in the problem statement: {L.mentioned.mean():.0%}")
L['dbin'] = L.d_sf.where(np.isfinite(L.d_sf), 99).clip(upper=5).astype(int)
tab = L.groupby(['dbin', 'mentioned']).rank_ppr.apply(lambda r: f'{(r <= 5).mean():.0%} (n={len(r)})').unstack()
tab.index = [('unreached' if i == 99 else ('5+' if i == 5 else str(i))) for i in tab.index]
print('   top-5 localization rate by d (columns: F* mentioned in the issue?)'); print(tab.to_string())
fin = L[np.isfinite(L.d_sf)]
print(f"   Spearman(d, rank) = {spearmanr(fin.d_sf, fin.rank_ppr).correlation:+.3f} | unmentioned only: "
      f"{spearmanr(fin[~fin.mentioned].d_sf, fin[~fin.mentioned].rank_ppr).correlation:+.3f}")

# ------------------------------------------------------------ D. ranking fragility predicts difficulty?
print(f'\nD) FRAGILITY of the top-{K} ({NREP} perturbed replicas per task)')
print(f"   mean overlap q: median {L.q_overlap.median():.2f} | distinct top-1 'basins': median {L.basins.median():.0f} | "
      f"F* inside the frozen core: {L.F_in_frozen.mean():.0%}")
for p in ('q_overlap', 'stab_k', 'gap_k', 'basins', 'frozen', 'n_fail', 'nF', 'd_sf'):
    ok = L[np.isfinite(L[p])]
    print(f'   {p:10s} Spearman with F* rank {spearmanr(ok[p], ok.rank_ppr).correlation:+.3f}')
hit = L.rank_ppr <= K
for p in ('q_overlap', 'gap_k', 'basins'):
    a, b = L.loc[hit, p], L.loc[~hit, p]
    if len(a) and len(b): print(f'   AUC of {p} for "F* in top-{K}": {mannwhitneyu(a, b).statistic / (len(a) * len(b)):.3f}')
print('\n   frozen core vs hit:', pd.crosstab(L.F_in_frozen, hit, rownames=['F* in frozen core'], colnames=[f'F* in top-{K}']).to_dict())

# ------------------------------------------------------------ E. effective dimension (precondition of distance-exponent routing)
print('\nE) EFFECTIVE DIMENSION: log ball volume vs log radius on the undirected call graph (median R^2 of a straight line)')
r2s = []
for d in sorted(p.parent for p in TD.glob('*/G.json'))[:40]:
    idx = json.load(open(d / 'index.json')); R = graph(idx); Au = ((R + R.T) > 0).astype(float); n = R.shape[0]
    src = np.random.default_rng(0).choice(n, min(30, n), replace=False)
    D = shortest_path(Au, indices=src, unweighted=True, directed=False)
    rmax = int(np.nanmax(D[np.isfinite(D)])) if np.isfinite(D).any() else 0
    if rmax < 3: continue
    rad = np.arange(1, rmax + 1); vol = np.array([(D <= r).sum(axis=1).mean() for r in rad])
    x, y = np.log(rad), np.log(vol); c = np.polyfit(x, y, 1); r2 = 1 - ((y - np.polyval(c, x)) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    r2s.append((d.name, rmax, c[0], r2))
E = pd.DataFrame(r2s, columns=['task', 'diameter', 'slope', 'R2'])
print(f'   {len(E)} graphs: diameter median {E.diameter.median():.0f}, slope median {E.slope.median():.2f}, R^2 median {E.R2.median():.3f} '
      '(a usable dimension needs a straight line over a wide range of radii)')
