# ---- Step 13: (A) the competition's provided code graph vs our runtime graph for localization; (B) how much the
# graders observe at the fix site. CPU, minutes. Inputs: end-to-end output (propagator/...) + competition data. ----
import os, glob, json, re, time, warnings
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as ssp
from scipy.sparse.linalg import spsolve
warnings.filterwarnings('ignore')

def find_out():
    for base in (os.environ.get('PROP_WORKDIR', '/kaggle/working'), '/kaggle/input'):
        for h in sorted(glob.glob(f'{base}/**/propagator/tasks', recursive=True)):
            if glob.glob(f'{h}/*/G.json'): return Path(h).parent
    raise FileNotFoundError('attach the end-to-end output (propagator/tasks/*/G.json)')
OUT = find_out(); TD, RD = OUT / 'tasks', OUT / 'repro'
DATA_ROOT = os.environ.get('PROP_DATA', '/kaggle/input')
SAVE = Path(os.environ.get('PROP_WORKDIR', '/kaggle/working')) / 'step13'; SAVE.mkdir(parents=True, exist_ok=True)
SENSITIVE = {'tests.test_syntax::test_from_path', 'tests.test_syntax::test_syntax_guess_lexer'}
GRAPH_FILES = sorted(p for p in glob.glob(f'{DATA_ROOT}/**/graphs/*.json', recursive=True))
TASKS = {}
for f in sorted(glob.glob(f'{DATA_ROOT}/**/*.jsonl', recursive=True)):
    with open(f) as fh: first = fh.readline()
    if '"instance_id"' in first and '"patch"' in first:
        TASKS = {json.loads(l)['instance_id']: json.loads(l) for l in open(f) if l.strip()}; break
SHA = re.compile(r'^[0-9a-f]{40}$')
def graph_file_for(tid):
    t = TASKS.get(tid, {}); repo = tid.split('_')[0]
    shas = [v for k, v in t.items() if isinstance(v, str) and SHA.match(v)]
    for sha in shas:
        for p in GRAPH_FILES:
            if Path(p).stem == f'{repo}_{sha}': return p, 'commit'
    return None, 'no matching commit (fields with a sha: ' + ','.join(k for k, v in t.items() if isinstance(v, str) and SHA.match(v)) + ')'
print('using', OUT, '| provided graph files:', [Path(p).name for p in GRAPH_FILES])

# ------------------------------------------------------------ loading the provided graph (NetworkX JSON, several layouts)
def load_provided(path):
    data = json.load(open(path))
    if isinstance(data, dict) and 'nodes' in data:
        nodes = [n['id'] if isinstance(n, dict) else n for n in data['nodes']]
        E = data.get('links') or data.get('edges') or []
        def ends(e):
            for a, b in (('source', 'target'), ('from', 'to'), ('u', 'v')):
                if a in e: return e[a], e[b]
            return e[0], e[1]
        edges = [(*ends(e), (e.get('type') or e.get('edge_type') or e.get('label') or e.get('key') or '?') if isinstance(e, dict) else '?')
                 for e in E]
        if nodes and all(isinstance(x, int) for x in nodes[:5]) and edges and isinstance(edges[0][0], int) and 'name' in data['nodes'][0]:
            names = [n['name'] for n in data['nodes']]; nodes = names; edges = [(names[a], names[b], t) for a, b, t in edges]
        return nodes, edges
    if isinstance(data, dict):                                                 # adjacency dict {node: [nbrs] or {nbr: attrs}}
        nodes = list(data); edges = []
        for u, nb in data.items():
            for v in (nb if isinstance(nb, list) else list(nb)): edges.append((u, v if not isinstance(v, dict) else v.get('id'), '?'))
        return nodes, edges
    raise ValueError(f'unrecognised graph layout in {path}: top-level {type(data).__name__}')

def norm_keys(name):
    s = re.sub(r'\.py(?=[.:/;]|$)', '', str(name)); s = re.sub(r'[/:;]+', '.', s).strip('.')
    if s.startswith('src.'): s = s[4:]
    keys = {s}
    parts = s.split('.')
    if '__init__' in parts[:-1]:                                               # package __init__ module segment (not a method)
        i = parts.index('__init__'); keys.add('.'.join(parts[:i] + parts[i + 1:]))
    return keys

GRAPHS = {}
def provided_for(path):
    if path in GRAPHS: return GRAPHS[path]
    nodes, edges = load_provided(path); ix = {n: i for i, n in enumerate(nodes)}
    E = [(ix[a], ix[b], t) for a, b, t in edges if a in ix and b in ix and a != b]
    n = len(nodes); A = ssp.csr_matrix((np.ones(len(E)), ([e[0] for e in E], [e[1] for e in E])), shape=(n, n)); A.data[:] = 1
    types = Counter(t for _, _, t in E)
    calls = [(a, b) for a, b, t in E if 'call' in str(t).lower()]
    Ac = ssp.csr_matrix((np.ones(len(calls)), ([e[0] for e in calls], [e[1] for e in calls])), shape=(n, n)) if calls else None
    if Ac is not None: Ac.data[:] = 1
    lookup = defaultdict(set)
    for i, nm in enumerate(nodes):
        for k in norm_keys(nm): lookup[k].add(i)
    contain = sum(1 for a, b, _ in E if str(nodes[b]).startswith(str(nodes[a]) + '.') or str(nodes[a]).startswith(str(nodes[b]) + '.'))
    GRAPHS[path] = dict(nodes=nodes, A=A, Ac=Ac, types=types, lookup=lookup, path=path, containment=contain / max(len(E), 1),
                        reciprocity=A.multiply(A.T).nnz / max(A.nnz, 1))
    return GRAPHS[path]

def map_funcs(funcs, P):
    """runtime function name -> provided node index (exact normalised match, else unique suffix match)"""
    out, suffix = {}, defaultdict(set)
    for k, ids in P['lookup'].items():
        for depth in (2, 3):
            suffix['.'.join(k.split('.')[-depth:])] |= ids
    for j, f in enumerate(funcs):
        hit = next((P['lookup'][k] for k in norm_keys(f) if k in P['lookup']), None)
        if not hit:
            for depth in (3, 2):
                cand = suffix.get('.'.join(f.split('.')[-depth:]))
                if cand and len(cand) == 1: hit = cand; break
        if hit and len(hit) == 1: out[j] = next(iter(hit))
    return out

# ------------------------------------------------------------ localizers
def ochiai(C, fail):
    ef = np.asarray(C[fail].sum(axis=0)).ravel(); tot = np.asarray(C.sum(axis=0)).ravel()
    den = np.sqrt(len(fail) * tot); den[den == 0] = 1; return ef / den
def transition_T(A):
    Au = ((A + A.T) > 0).astype(float); deg = np.asarray(Au.sum(axis=1)).ravel(); deg[deg == 0] = 1
    return (ssp.diags(1 / deg) @ Au).T.tocsr()
def diffuse(PT, s, mode):
    if mode == 'k1': return s + PT @ s
    sn = s / s.sum() if s.sum() > 0 else s
    return 0.5 * spsolve((ssp.identity(PT.shape[0]) - 0.5 * PT).tocsc(), sn)       # PPR, alpha = 0.5
def on_other_graph(PT, s0, fmap, n_other, mode):
    """seed the provided graph with the runtime scores of mapped functions, diffuse there, read back; unmapped keep s0"""
    s = np.zeros(n_other)
    for j, i in fmap.items(): s[i] += s0[j]
    z = diffuse(PT, s, mode); out = s0.copy()
    for j, i in fmap.items(): out[j] = z[i]
    return out
def best_rank(s, targets):
    order = (-s).argsort(kind='stable'); pos = np.empty(len(s), int); pos[order] = np.arange(1, len(s) + 1)
    return int(min(pos[t] for t in targets))

rows, obs, t0 = [], [], time.time()
for d in sorted(p.parent for p in TD.glob('*/G.json')):
    tid = d.name; repo = tid.split('_')[0]
    try:
        idx = json.load(open(d / 'index.json')); keys, funcs = idx['tests'], idx['funcs']; fi = {f: i for i, f in enumerate(funcs)}
        n = len(funcs); E = np.array(idx['edges'], dtype=np.int64).reshape(-1, 2)
        R = ssp.csr_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(n, n)); R.data[:] = 1
        C = ssp.load_npz(d / 'C.npz').tocsr().astype(float)
        f2p = set(json.load(open(RD / f'{tid}.json')).get('f2p', []))
        fail = [i for i, k in enumerate(keys) if (k in f2p or k.split('::')[-1].split('[')[0] in f2p) and k not in SENSITIVE]
        g = json.load(open(d / 'G.json')); F = sorted({m['site'] for m in g['mutants'] if m['cls'] == 'Fstar'} & set(fi))
        F_idx = [fi[f] for f in F]
        # ---- (B) grader observation at the fix site: do the task's own fail-to-pass tests catch planted faults at F*?
        fail_set = set(fail)
        for m in g['mutants']:
            if m['cls'] != 'Fstar' or 'skip' in m or m.get('timeout') or m['n_sel'] == 0: continue
            k = set(m['killed']) - {i for i, kk in enumerate(keys) if kk in SENSITIVE}
            obs.append(dict(task=tid, repo=repo, any_test=bool(k), grader=bool(k & fail_set), n_cover=m['n_cover']))
        if not fail or not F_idx: continue
        # ---- (A) localization on three graphs, same seeds, same candidate set
        s0 = ochiai(C, fail); row = dict(task=tid, repo=repo, n_funcs=n, rank_coverage_only=best_rank(s0, F_idx))
        PTr = transition_T(R)
        for mode in ('k1', 'ppr'): row[f'rank_runtime_{mode}'] = best_rank(diffuse(PTr, s0, mode), F_idx)
        gpath, how = graph_file_for(tid); row['graph'] = Path(gpath).name if gpath else how
        P = provided_for(gpath) if gpath else None
        if P is not None:
            fmap = map_funcs(funcs, P); row.update(mapped_funcs=len(fmap) / n, F_mapped=np.mean([j in fmap for j in F_idx]))
            PTp = transition_T(P['A'])
            for mode in ('k1', 'ppr'): row[f'rank_provided_{mode}'] = best_rank(on_other_graph(PTp, s0, fmap, len(P['nodes']), mode), F_idx)
            # FAIR comparison: rank only among functions present in both graphs (no mixing of propagated and raw scores)
            mask = np.full(n, -np.inf); keep = np.array(sorted(fmap)); Fm = [j for j in F_idx if j in fmap]
            if len(Fm):
                fair = lambda sc: best_rank(np.where(np.isin(np.arange(n), keep), sc, -np.inf), Fm)
                row['fair_coverage_only'] = fair(s0)
                for mode in ('k1', 'ppr'):
                    row[f'fair_runtime_{mode}'] = fair(diffuse(PTr, s0, mode))
                    sp_ = np.zeros(len(P['nodes']))
                    for j, i in fmap.items(): sp_[i] += s0[j]
                    z = diffuse(PTp, sp_, mode); zz = np.full(n, -np.inf)
                    for j, i in fmap.items(): zz[j] = z[i]
                    row[f'fair_provided_{mode}'] = fair(zz)
            if P['Ac'] is not None:
                PTc = transition_T(P['Ac'])
                for mode in ('k1', 'ppr'): row[f'rank_providedcalls_{mode}'] = best_rank(on_other_graph(PTc, s0, fmap, len(P['nodes']), mode), F_idx)
        rows.append(row)
    except Exception as e:
        print('skip', tid, repr(e)[:150])
L = pd.DataFrame(rows); L.to_csv(SAVE / 'step13_localization.csv', index=False)
O = pd.DataFrame(obs); O.to_csv(SAVE / 'step13_grader_observation.csv', index=False)
print(f'{len(L)} tasks in {time.time() - t0:.0f}s')

print('\nPROVIDED GRAPHS (one per task commit):')
print(f"   tasks matched to their own commit's graph: {L.graph.str.endswith('.json').sum()}/{len(L)}"
      + ('' if L.graph.str.endswith('.json').all() else f" | unmatched e.g.: {L.loc[~L.graph.str.endswith('.json'), 'graph'].iloc[0]}"))
G = pd.DataFrame([dict(repo=Path(p).stem.split('_')[0], nodes=len(P['nodes']), edges=P['A'].nnz, reciprocity=P['reciprocity'],
                       containment=P['containment'], call_typed=sum(v for k, v in P['types'].items() if 'call' in str(k).lower()) / max(sum(P['types'].values()), 1))
                  for p, P in GRAPHS.items() if P])
if len(G): print(G.groupby('repo').median().round(3).to_string())
if 'mapped_funcs' in L:
    print(f"   runtime functions found in the provided graph: median {L.mapped_funcs.median():.0%}; fix functions found: {L.F_mapped.mean():.0%}")

print('\nA) LOCALIZATION OF THE FIX (same failing-test seeds; candidates = executed functions)')
res = []
for c in [c for c in L.columns if c.startswith('rank_')]:
    r = L[c].dropna()
    res.append(dict(localizer=c[5:], tasks=len(r), top1=(r <= 1).mean(), top5=(r <= 5).mean(), top10=(r <= 10).mean(), MRR=(1 / r).mean(),
                    median_rank=r.median()))
print(pd.DataFrame(res).round(3).to_string(index=False))
print('\n   FAIR version: candidates restricted to functions present in BOTH graphs (same candidate set for every localizer)')
res = []
for c in [c for c in L.columns if c.startswith('fair_')]:
    r = L[c].dropna()
    res.append(dict(localizer=c[5:], tasks=len(r), top1=(r <= 1).mean(), top5=(r <= 5).mean(), top10=(r <= 10).mean(), MRR=(1 / r).mean(),
                    median_rank=r.median()))
if res: print(pd.DataFrame(res).round(3).to_string(index=False))
if 'fair_provided_k1' in L:
    from scipy.stats import wilcoxon
    fb = L.dropna(subset=['fair_provided_k1', 'fair_runtime_k1'])
    dl = np.log(fb.fair_provided_k1) - np.log(fb.fair_runtime_k1)
    print(f'   fair paired, one-hop: runtime better in {(dl > 0).mean():.0%}, worse in {(dl < 0).mean():.0%} '
          f'(Wilcoxon p = {wilcoxon(dl[dl != 0]).pvalue if (dl != 0).sum() > 5 else float("nan"):.3g})')
    print('   fair, by repo, top-5 one-hop runtime vs provided:',
          {r: f"{(g.fair_runtime_k1 <= 5).mean():.0%} vs {(g.fair_provided_k1 <= 5).mean():.0%} (n={len(g)})" for r, g in fb.groupby('repo')})
both = L.dropna(subset=['rank_provided_k1']) if 'rank_provided_k1' in L else L.iloc[:0]
if len(both):
    from scipy.stats import wilcoxon
    dlt = np.log(both.rank_provided_k1) - np.log(both.rank_runtime_k1)
    print(f'   paired, one-hop: runtime graph ranks the fix better in {(dlt > 0).mean():.0%} of tasks, worse in {(dlt < 0).mean():.0%} '
          f'(Wilcoxon p = {wilcoxon(dlt[dlt != 0]).pvalue if (dlt != 0).sum() > 5 else float("nan"):.3g})')
    print('   by repo, top-5 one-hop runtime vs provided:',
          {r: f"{(g.rank_runtime_k1 <= 5).mean():.0%} vs {(g.rank_provided_k1 <= 5).mean():.0%} (n={len(g)})" for r, g in both.groupby('repo')})

print('\nB) DO THE GRADERS OBSERVE THE FIX SITE? planted faults at F* (2 per fix function), caught by ...')
if len(O):
    print(f"   any covering test: {O.any_test.mean():.0%} | the task's own fail-to-pass tests: {O.grader.mean():.0%} "
          f"({len(O)} mutants, {O.task.nunique()} tasks)")
    per = O.groupby('task').grader.mean()
    print(f"   tasks where the grader catches NONE of the planted faults at the fix site: {(per == 0).mean():.0%}; all of them: {(per == 1).mean():.0%}")
    print('   by repo (grader catch rate):', O.groupby('repo').grader.mean().round(2).to_dict())
