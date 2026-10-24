# ---- Step G-a: build the Gemma prompt pack (CPU, minutes, no internet).
# Inputs: the competition data (tasks .jsonl, snapshots/*.tgz, graphs/*.json) + the end-to-end output (propagator/tasks, propagator/repro).
# Output: /kaggle/working/gemma_pack/{loc.jsonl, repair.jsonl, meta.json} -> attach to the GPU notebook (Step G-b).
#   E1 localization: Gemma picks the function to edit from K candidates; conditions differ ONLY in who chose the candidates:
#      bm25 = issue text only (no execution) | cov = failing tests' runtime coverage (Ochiai) | rt1 = cov + one hop on the runtime call graph
#      | pv1 = cov + one hop on the competition's provided graph. Same prompt template, candidates shown in a seeded random order.
#   E2 repair: Gemma rewrites the functions the reference fix touches (oracle localization); the patches are graded in Step G-c by the
#      harness (fail-to-pass + pass-to-pass of the touched test files) vs the extended oracle (+ every old test that executes the edit).
import os, glob, json, re, ast, math, random, tarfile, time, zlib, warnings
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as ssp
warnings.filterwarnings('ignore')
K = 15; MAX_SNIP_LINES = 25; MAX_SNIP_CHARS = 1400; MAX_ISSUE = 6000; MAX_REPAIR_LINES = 400; MAX_REPAIR_FUNCS = 5

def find_out():
    for base in (os.environ.get('PROP_WORKDIR', '/kaggle/working'), '/kaggle/input'):
        for h in sorted(glob.glob(f'{base}/**/propagator/tasks', recursive=True)):
            if glob.glob(f'{h}/*/G.json'): return Path(h).parent
    raise FileNotFoundError('attach the end-to-end output (propagator/tasks/*/G.json)')
OUT = find_out(); TD, RD = OUT / 'tasks', OUT / 'repro'
DATA_ROOT = os.environ.get('PROP_DATA', '/kaggle/input')
SAVE = Path(os.environ.get('PROP_WORKDIR', '/kaggle/working')) / 'gemma_pack'; SAVE.mkdir(parents=True, exist_ok=True)
SNAP_DIR = next((Path(p) for p in sorted(glob.glob(f'{DATA_ROOT}/**/snapshots', recursive=True)) if glob.glob(f'{p}/*.tgz')), None)
GRAPH_FILES = sorted(glob.glob(f'{DATA_ROOT}/**/graphs/*.json', recursive=True))
TASKS = {}
for f in sorted(glob.glob(f'{DATA_ROOT}/**/*.jsonl', recursive=True)):
    with open(f) as fh: first = fh.readline()
    if '"instance_id"' in first and '"patch"' in first:
        TASKS = {json.loads(l)['instance_id']: json.loads(l) for l in open(f) if l.strip()}; break
SENSITIVE = {'tests.test_syntax::test_from_path', 'tests.test_syntax::test_syntax_guess_lexer'}
TESTFILE = re.compile(r'(^|/)(tests?|testing)/.*\.py$|(^|/)test_[^/]*\.py$|_test\.py$')
NONPKG = {'tests', 'test', 'testing', 'docs', 'docs_src', 'scripts', 'examples', 'benchmarks', 'site', 'build', 'dist'}
print('end-to-end output:', OUT, '| snapshots:', SNAP_DIR, '| tasks:', len(TASKS), '| graph files:', len(GRAPH_FILES))

# ------------------------------------------------------------ the base (buggy) source, straight from the snapshot tarball
def snapshot_py(tid):
    files = {}
    with tarfile.open(SNAP_DIR / f'{tid}.tgz') as tf:
        for m in tf.getmembers():
            name = m.name[2:] if m.name.startswith('./') else m.name
            if m.isfile() and name.endswith('.py') and '/.git/' not in '/' + name: files[name] = tf.extractfile(m).read().decode('utf-8', 'replace')
    tops = {p.split('/')[0] for p in files}
    if len(tops) == 1 and not any('/' not in p for p in files):             # single top folder: strip it
        top = next(iter(tops)) + '/'
        if not any(p == top + 'pyproject.toml' for p in files): files = {p[len(top):]: s for p, s in files.items()}
    return files

def modname(rel):
    parts = rel[:-3].split('/')
    if parts and parts[0] == 'src': parts = parts[1:]
    if parts and parts[-1] == '__init__': parts = parts[:-1]
    return '.'.join(parts)

def catalogue(files):
    """qualified function name -> (file, first line incl. decorators, last line, source) for package code"""
    pk = {p.split('/')[1] if p.startswith('src/') else p.split('/')[0] for p in files if p.endswith('__init__.py')} - NONPKG
    cat = {}
    for rel, src in files.items():
        if TESTFILE.search(rel) or modname(rel).split('.')[0] not in pk: continue
        try: tree = ast.parse(src)
        except SyntaxError: continue
        lines = src.split('\n'); mod = modname(rel)
        def walk(node, pre):
            for ch in ast.iter_child_nodes(node):
                if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    q = pre + ch.name; s = min([ch.lineno] + [d.lineno for d in ch.decorator_list]); e = ch.end_lineno
                    if not isinstance(ch, ast.ClassDef):
                        cat[f'{mod}.{q}'] = (rel, s, e, '\n'.join(lines[s - 1:e]))
                    walk(ch, q + '.')
        walk(tree, '')
    return cat

def changed_old_lines(patch):
    out, cur, ol = defaultdict(set), None, None
    for line in patch.replace('\r\n', '\n').split('\n'):
        if line.startswith('--- '): cur = line[6:] if line.startswith('--- a/') else None; continue
        if line.startswith('+++ '): continue
        if line.startswith('@@'):
            m = re.match(r'@@ -(\d+)', line); ol = int(m.group(1)) if m else None; continue
        if cur is None or ol is None or line.startswith('\\'): continue
        if line.startswith('-'): out[cur].add(ol); ol += 1
        elif line.startswith('+'): out[cur].add(max(ol - 1, 1)); out[cur].add(ol)
        else: ol += 1
    return out

def patch_funcs_base(cat, patch):
    """functions of the BASE code the reference fix edits (innermost function containing each changed old-side line)"""
    by_file = defaultdict(list)
    for q, (rel, s, e, _) in cat.items(): by_file[rel].append((s, e, q))
    F = set()
    for rel, lines in changed_old_lines(patch).items():
        for ln in lines:
            inside = [x for x in by_file.get(rel, []) if x[0] <= ln <= x[1]]
            if inside: F.add(max(inside)[2])
    return sorted(F)

# ------------------------------------------------------------ retrieval: BM25 over the issue text (the no-execution baseline)
TOK = re.compile(r'[A-Za-z_][A-Za-z0-9_]+')
def toks(text):
    out = []
    for w in TOK.findall(text):
        parts = [p for p in re.split(r'_|(?<=[a-z])(?=[A-Z])', w) if len(p) > 1]
        out += [w.lower()] + [p.lower() for p in parts if p.lower() != w.lower()]
    return out
def bm25_rank(query, docs, k1=1.2, b=0.75):
    D = [Counter(toks(d)) for d in docs]; N = len(D); dl = np.array([sum(c.values()) for c in D], float); avg = dl.mean() if N else 1
    df = Counter(t for c in D for t in c); q = set(toks(query)); sc = np.zeros(N)
    for t in q:
        if t not in df: continue
        idf = math.log(1 + (N - df[t] + 0.5) / (df[t] + 0.5))
        tf = np.array([c.get(t, 0) for c in D], float); sc += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avg))
    return sc

# ------------------------------------------------------------ the provided graph (as in Step 13)
SHA = re.compile(r'^[0-9a-f]{40}$')
def graph_file_for(tid):
    t = TASKS.get(tid, {}); repo = tid.split('_')[0]
    for sha in [v for v in t.values() if isinstance(v, str) and SHA.match(v)]:
        for p in GRAPH_FILES:
            if Path(p).stem == f'{repo}_{sha}': return p
    return None
def norm_keys(name):
    s = re.sub(r'\.py(?=[.:/;]|$)', '', str(name)); s = re.sub(r'[/:;]+', '.', s).strip('.')
    if s.startswith('src.'): s = s[4:]
    keys = {s}; parts = s.split('.')
    if '__init__' in parts[:-1]: i = parts.index('__init__'); keys.add('.'.join(parts[:i] + parts[i + 1:]))
    return keys
_GR = {}
def provided(path):
    if path in _GR: return _GR[path]
    data = json.load(open(path)); nodes = [n['id'] if isinstance(n, dict) else n for n in data['nodes']]
    E = data.get('links') or data.get('edges') or []; ix = {n: i for i, n in enumerate(nodes)}
    pairs = [(ix[e['source']], ix[e['target']]) for e in E if isinstance(e, dict) and e.get('source') in ix and e.get('target') in ix]
    n = len(nodes); A = ssp.csr_matrix((np.ones(len(pairs)), ([a for a, _ in pairs], [b for _, b in pairs])), shape=(n, n)); A.data[:] = 1
    lookup = defaultdict(set)
    for i, nm in enumerate(nodes):
        for k in norm_keys(nm): lookup[k].add(i)
    _GR[path] = (nodes, A, lookup); return _GR[path]
def map_funcs(funcs, lookup):
    out, suffix = {}, defaultdict(set)
    for k, ids in lookup.items():
        for depth in (2, 3): suffix['.'.join(k.split('.')[-depth:])] |= ids
    for j, f in enumerate(funcs):
        hit = next((lookup[k] for k in norm_keys(f) if k in lookup), None)
        if not hit:
            for depth in (3, 2):
                c = suffix.get('.'.join(f.split('.')[-depth:]))
                if c and len(c) == 1: hit = c; break
        if hit and len(hit) == 1: out[j] = next(iter(hit))
    return out
def transition_T(A):
    Au = ((A + A.T) > 0).astype(float); deg = np.asarray(Au.sum(axis=1)).ravel(); deg[deg == 0] = 1
    return (ssp.diags(1 / deg) @ Au).T.tocsr()
def ochiai(C, fail):
    ef = np.asarray(C[fail].sum(axis=0)).ravel(); tot = np.asarray(C.sum(axis=0)).ravel()
    den = np.sqrt(len(fail) * tot); den[den == 0] = 1; return ef / den

# ------------------------------------------------------------ prompts
class _Slim(ast.NodeTransformer):
    """drop docstrings, comments and Doc(...)-style annotation payloads so a snippet shows the signature and the logic"""
    def _strip(self, node):
        b = node.body
        if b and isinstance(b[0], ast.Expr) and isinstance(getattr(b[0], 'value', None), ast.Constant) and isinstance(b[0].value.value, str):
            node.body = b[1:] or [ast.Pass()]
        self.generic_visit(node); return node
    visit_FunctionDef = visit_AsyncFunctionDef = visit_ClassDef = _strip
    def visit_arg(self, node):
        if node.annotation is not None and len(ast.unparse(node.annotation)) > 60:
            a = node.annotation                                     # Annotated[T, Doc("...")] -> T
            node.annotation = a.slice.elts[0] if isinstance(a, ast.Subscript) and isinstance(a.slice, ast.Tuple) and a.slice.elts else None
        return node

def snippet(src):
    body, _ = dedent_block(src)
    try:
        tree = _Slim().visit(ast.parse(body)); body = ast.unparse(tree)
        fn = tree.body[0] if tree.body and isinstance(tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef)) else None
        if fn is not None and max(len(l) for l in body.split('\n')[:len(fn.decorator_list) + 1]) > 300:          # very long signature: keep parameter names only
            a = fn.args; names = [x.arg for x in a.posonlyargs + a.args] + ([f'*{a.vararg.arg}'] if a.vararg else (['*'] if a.kwonlyargs else []))
            names += [x.arg for x in a.kwonlyargs] + ([f'**{a.kwarg.arg}'] if a.kwarg else [])
            head = ''.join(f'@{ast.unparse(d)}\n' for d in fn.decorator_list)
            head += ('async ' if isinstance(fn, ast.AsyncFunctionDef) else '') + f"def {fn.name}({', '.join(names)}):  # annotations/defaults omitted"
            body = head + '\n' + '\n'.join('    ' + l for st in fn.body for l in ast.unparse(st).split('\n'))
    except Exception: pass
    lines = body.split('\n'); s = '\n'.join(lines[:MAX_SNIP_LINES])
    if len(lines) > MAX_SNIP_LINES: s += f'\n    ... ({len(lines) - MAX_SNIP_LINES} more lines)'
    return s[:MAX_SNIP_CHARS]
LOC_TMPL = """You are a software engineer fixing a GitHub issue in the Python repository {repo}.

<issue>
{issue}
</issue>

Below are {k} candidate functions from the repository (shown in no particular order; long bodies are truncated).

{cands}

Which function(s) must be edited to resolve the issue? Answer with a JSON object and nothing else:
{{"functions": ["<qualified name>", ...]}}
List at most 3 names, most likely first, copied exactly from the candidate headers."""
REPAIR_TMPL = """You are a software engineer fixing a GitHub issue in the Python repository {repo}.

<issue>
{issue}
</issue>

The fix belongs in the following function(s) of the current (buggy) code:

{funcs}

Write the fix. For EVERY function you change, output its complete new source, starting at column 0 (no class-level indentation), as:

### <qualified name>
```python
<complete function source, including decorators>
```

Only output functions you change, use exactly the qualified names above, keep their signatures unless the issue requires otherwise, and do not add new top-level imports."""

def dedent_block(src):
    lines = src.split('\n'); ind = min((len(l) - len(l.lstrip()) for l in lines if l.strip()), default=0)
    return '\n'.join(l[ind:] for l in lines), ind

loc_rows, rep_rows, stats, t0 = [], [], [], time.time()
for d in sorted(p.parent for p in TD.glob('*/G.json')):
    tid = d.name; t = TASKS.get(tid)
    if t is None: continue
    try:
        idx = json.load(open(d / 'index.json')); keys, funcs = idx['tests'], idx['funcs']; fi = {f: i for i, f in enumerate(funcs)}
        n = len(funcs); E = np.array(idx['edges'], dtype=np.int64).reshape(-1, 2)
        R = ssp.csr_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(n, n)); R.data[:] = 1
        C = ssp.load_npz(d / 'C.npz').tocsr().astype(float)
        f2p = set(json.load(open(RD / f'{tid}.json')).get('f2p', []))
        fail = [i for i, k in enumerate(keys) if (k in f2p or k.split('::')[-1].split('[')[0] in f2p) and k not in SENSITIVE]
        files = snapshot_py(tid); cat = catalogue(files)
        Fb = patch_funcs_base(cat, t['patch'])                               # oracle: base-side functions the fix edits
        st = dict(task=tid, repo=tid.split('_')[0], n_cat=len(cat), nF=len(Fb), n_fail=len(fail))
        issue = (t.get('problem_statement') or '')[:MAX_ISSUE]; repo = t.get('repo', tid.split('_')[0])
        if Fb and fail:
            names = list(cat); s_bm = bm25_rank(issue, [q.replace('.', ' ') + '\n' + cat[q][3] for q in names])
            scores = {'bm25': dict(zip(names, s_bm))}
            s0 = ochiai(C, fail); scores['cov'] = dict(zip(funcs, s0))
            scores['rt1'] = dict(zip(funcs, s0 + transition_T(R) @ s0))
            gp = graph_file_for(tid)
            if gp:
                nodes, A, lookup = provided(gp); fmap = map_funcs(funcs, lookup); sp_ = np.zeros(len(nodes))
                for j, i in fmap.items(): sp_[i] += s0[j]
                z = sp_ + transition_T(A) @ sp_; out = s0.copy()
                for j, i in fmap.items(): out[j] = z[i]
                scores['pv1'] = dict(zip(funcs, out))
            rng = random.Random(zlib.crc32(tid.encode()))
            for cond, sc in scores.items():
                ranked = [q for q, v in sorted(sc.items(), key=lambda kv: -kv[1]) if q in cat and v > 0][:K]
                st[f'recall_{cond}'] = any(q in Fb for q in ranked); st[f'n_{cond}'] = len(ranked)
                if not ranked: continue
                shown = ranked[:]; rng.shuffle(shown)
                cands = '\n\n'.join(f'### {q}   ({cat[q][0]})\n```python\n{dedent_block(snippet(cat[q][3]))[0]}\n```' for q in shown)
                loc_rows.append(dict(id=f'{tid}|{cond}', task=tid, cond=cond, candidates=shown, fstar=Fb,
                                     prompt=LOC_TMPL.format(repo=repo, issue=issue, k=len(shown), cands=cands)))
        if Fb and all(cat[q][2] - cat[q][1] < MAX_REPAIR_LINES for q in Fb) and len(Fb) <= MAX_REPAIR_FUNCS:
            blocks, meta = [], []
            for q in Fb:
                rel, s, e, src = cat[q]; body, ind = dedent_block(src)
                blocks.append(f'### {q}   ({rel}, lines {s}-{e})\n```python\n{body}\n```'); meta.append(dict(qual=q, file=rel, start=s, end=e, indent=ind))
            rep_rows.append(dict(id=tid, task=tid, funcs=meta, prompt=REPAIR_TMPL.format(repo=repo, issue=issue, funcs='\n\n'.join(blocks))))
            st['repair'] = True
        stats.append(st)
    except Exception as ex:
        print('skip', tid, repr(ex)[:150])

with open(SAVE / 'loc.jsonl', 'w') as fh:
    for r in loc_rows: fh.write(json.dumps(r) + '\n')
with open(SAVE / 'repair.jsonl', 'w') as fh:
    for r in rep_rows: fh.write(json.dumps(r) + '\n')
S = pd.DataFrame(stats); S.to_csv(SAVE / 'pack_stats.csv', index=False)
json.dump(dict(K=K, created=time.strftime('%Y-%m-%d %H:%M'), n_loc=len(loc_rows), n_repair=len(rep_rows)), open(SAVE / 'meta.json', 'w'))
print(f'\n{len(S)} tasks scanned in {time.time() - t0:.0f}s -> {len(loc_rows)} localization prompts, {len(rep_rows)} repair prompts')
print('localization prompts per condition:', Counter(r['cond'] for r in loc_rows))
print(f'retrieval recall@{K} (is a fix function among the candidates Gemma will see?):')
for c in ('bm25', 'cov', 'rt1', 'pv1'):
    if f'recall_{c}' in S: q = S[f'recall_{c}'].dropna(); print(f'   {c:5s} {q.mean():.0%} of {len(q)} tasks')
L = [len(r['prompt']) for r in loc_rows]; P = [len(r['prompt']) for r in rep_rows]
print(f'prompt length (chars): localization median {np.median(L):.0f} max {max(L)} | repair median {np.median(P):.0f} max {max(P)}' if L and P else '')
print('example localization prompt (truncated):\n' + (loc_rows[0]['prompt'][:1500] if loc_rows else '-'))
