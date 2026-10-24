# genlib.py - generalization beyond the competition: the measured propagator on open-source Python libraries.
# Needs internet (git clone + pip). CPU only. Self-contained: no competition data required.
import os, re, sys, json, time, shutil, subprocess, ast, gc, tempfile, tomllib, random, zlib, math
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as ssp, networkx as nx
from scipy.sparse.linalg import expm_multiply, spsolve
from scipy.sparse.csgraph import connected_components, breadth_first_order, shortest_path
from scipy.stats import spearmanr, wilcoxon

WORKDIR = Path(os.environ.get('PROP_WORKDIR', '/kaggle/working'))
GEN = WORKDIR / 'gen'; ENVS, PLUG, REPOS, WORK, OUTG = GEN / 'envs', GEN / 'plug', GEN / 'repos', GEN / 'work', GEN / 'out'
for _p in (ENVS, PLUG, REPOS, WORK, OUTG): _p.mkdir(parents=True, exist_ok=True)
MEM_CAP_MB = 6000; PY = sys.executable
LIBS = [dict(name='click', url='https://github.com/pallets/click', tag='8.1.7'),
        dict(name='more_itertools', url='https://github.com/more-itertools/more-itertools', tag='v10.2.0'),
        dict(name='toolz', url='https://github.com/pytoolz/toolz', tag='0.12.1'),
        dict(name='jinja2', url='https://github.com/pallets/jinja', tag='3.1.3'),
        dict(name='itsdangerous', url='https://github.com/pallets/itsdangerous', tag='2.1.2'),
        dict(name='packaging', url='https://github.com/pypa/packaging', tag='24.0'),
        dict(name='marshmallow', url='https://github.com/marshmallow-code/marshmallow', tag='3.21.0'),
        dict(name='boltons', url='https://github.com/mahmoud/boltons', tag='23.1.1')]
EXCLUDE_REQ = re.compile(r'^(pytest|pytest-cov|pytest-xdist|pytest-sugar|coverage|tox|nox|mypy|pre-commit|sphinx.*|furo|flake8.*|black|ruff|isort|pip-tools|twine|build|setuptools|wheel)$', re.I)

TESTFILE = re.compile(r'(^|/)(tests?|testing)/.*\.py$|(^|/)test_[^/]*\.py$|_test\.py$')

NONPKG = {'tests', 'test', 'testing', 'docs', 'docs_src', 'scripts', 'examples', 'benchmarks', 'site', 'build', 'dist'}

def _rss_mb(pid):
    try:
        with open(f'/proc/{pid}/status') as fh:
            for l in fh:
                if l.startswith('VmRSS:'): return int(l.split()[1]) // 1024
    except OSError:
        pass
    return 0

def sp(cmd, cwd, timeout, env=None, inp=None, cap_mb=None):
    """run a command; returns rc, output, secs, peak_rss_mb, killed_reason (None if not killed)"""
    cap = cap_mb or MEM_CAP_MB; t0 = time.time(); peak = 0; killed = None
    with tempfile.TemporaryFile('w+') as fo:
        p = subprocess.Popen([str(c) for c in cmd], cwd=str(cwd), env=env, text=True, stdout=fo, stderr=subprocess.STDOUT,
                             stdin=subprocess.PIPE if inp is not None else subprocess.DEVNULL, start_new_session=True)
        if inp is not None:
            try:
                p.stdin.write(inp); p.stdin.close()
            except BrokenPipeError:
                pass
        while p.poll() is None:
            r = _rss_mb(p.pid); peak = max(peak, r)
            if r > cap or time.time() - t0 > timeout:
                killed = f'rss>{cap}MB' if r > cap else 'timeout'
                try: os.killpg(p.pid, 9)
                except OSError: pass
            time.sleep(0.1)
        p.wait(); fo.seek(0); out = fo.read()
    return (124 if killed else p.returncode), out, time.time() - t0, peak, killed

STUB = '''import os
def pytest_addoption(parser):
    for name in os.environ.get("PROP_STUB_INI", "").split(","):
        if name: parser.addini(name, "stub: plugin unavailable offline", default="")
def pytest_configure(config):
    for m in os.environ.get("PROP_STUB_MARKERS", "").split(","):
        if m: config.addinivalue_line("markers", m + ": stub marker (plugin unavailable offline)")
'''

TRACER = r'''
import os, sys, json
import pytest
_mon = sys.monitoring
_ROOT = os.path.realpath(os.environ.get("PROP_TRACE_ROOT", os.getcwd())) + os.sep
_PKGS = set(filter(None, os.environ.get("PROP_TRACE_PKG", "").split(",")))
_DIR = os.environ.get("PROP_TRACE_DIR", ".")
_TOOL = next(i for i in (2, 3, 4, 5) if _mon.get_tool(i) is None)
_ids, _nid, _edges, _collect, _fcache = {}, {}, set(), set(), {}
_cur = _collect
_fh = open(os.path.join(_DIR, ".trace_tests.jsonl"), "w")

def _modparts(fname):
    r = _fcache.get(fname, 0)
    if r != 0:
        return r
    r = None
    if fname.endswith(".py"):
        rf = os.path.realpath(fname)
        if rf.startswith(_ROOT):
            parts = rf[len(_ROOT):-3].split(os.sep)
            if parts and parts[0] == "src": parts = parts[1:]
            if parts and parts[-1] == "__init__": parts = parts[:-1]
            if parts and parts[0] in _PKGS: r = ".".join(parts)
    _fcache[fname] = r
    return r

def _start(code, offset):
    i = _ids.get(code)
    if i is None:
        mod = _modparts(code.co_filename)
        if mod is None:
            return _mon.DISABLE
        q = code.co_qualname.replace("<locals>.", "")
        q = "<module>" if q.startswith("<") else q.split(".<")[0]
        i = _ids[code] = _nid.setdefault(mod + "." + q, len(_nid))
    _cur.add(i)
    f = sys._getframe(1).f_back
    for _ in range(6):
        if f is None: break
        j = _ids.get(f.f_code)
        if j is not None:
            if j != i: _edges.add((j, i))
            break
        f = f.f_back

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    global _cur
    _cur = set()
    try:
        yield
    finally:
        _fh.write(json.dumps([item.nodeid, sorted(_cur)]) + "\n")
        _cur = _collect

def pytest_sessionfinish(session, exitstatus):
    _mon.set_events(_TOOL, 0)
    _fh.close()
    nodes = [None] * len(_nid)
    for n, i in _nid.items(): nodes[i] = n
    with open(os.path.join(_DIR, ".trace_meta.json"), "w") as fh:
        json.dump({"nodes": nodes, "collect": sorted(_collect), "edges": sorted(_edges)}, fh)

_mon.use_tool_id(_TOOL, "prop_trace2")
_mon.register_callback(_TOOL, _mon.events.PY_START, _start)
_mon.set_events(_TOOL, _mon.events.PY_START)
'''

def junit_cases(xml):
    out = {}
    try:
        root = ET.parse(xml).getroot()
    except Exception:
        return out
    for tc in root.iter('testcase'):
        tags = {k.tag: k for k in tc}
        st = 'failed' if 'failure' in tags else 'error' if 'error' in tags else 'skipped' if 'skipped' in tags else 'passed'
        msg = ''
        for k in ('failure', 'error'):
            if k in tags:
                body = (tags[k].text or '').strip().splitlines()
                msg = ((tags[k].get('message') or '') + ' | ' + (body[-1] if body else ''))[:240]
        out[f"{tc.get('classname')}::{tc.get('name')}"] = (st, msg)
    return out

def nid2key(nid):
    p = nid.split('::'); f = p[0][:-3].replace('/', '.') if p[0].endswith('.py') else p[0]
    return '.'.join([f] + p[1:-1]) + '::' + p[-1]

def pkgs_of(root):
    root = Path(root)
    return sorted({d.name for b in (root, root / 'src') if b.is_dir() for d in b.iterdir()
                   if d.is_dir() and (d / '__init__.py').exists() and d.name not in NONPKG})

def load_trace(w):
    w = Path(w); meta = json.load(open(w / '.trace_meta.json')); keys, R, Cc = [], [], []
    with open(w / '.trace_tests.jsonl') as fh:
        for i, line in enumerate(fh):
            nid, ids = json.loads(line); keys.append(nid2key(nid))
            if ids: R.append(np.full(len(ids), i, np.int32)); Cc.append(np.asarray(ids, np.int32))
    r = np.concatenate(R) if R else np.zeros(0, np.int32); c = np.concatenate(Cc) if Cc else np.zeros(0, np.int32)
    C = ssp.csr_matrix((np.ones(len(r), np.int8), (r, c)), shape=(len(keys), len(meta['nodes'])))
    C.data[:] = 1
    return keys, meta['nodes'], C, meta

def _setup_py_reqs(path):
    out = []
    try:
        tree = ast.parse(path.read_text(errors='replace'))
    except SyntaxError:
        return out
    def strs(node):
        if isinstance(node, (ast.List, ast.Tuple)): return [e.value for e in node.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)]
        if isinstance(node, ast.Dict): return [s for v in node.values for s in strs(v)]
        return []
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and 'require' in t.id.lower() for t in n.targets): out += strs(n.value)
        if isinstance(n, ast.keyword) and n.arg and 'require' in n.arg: out += strs(n.value)
    return out

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

def key_file(w, key):
    parts = key.split('::')[0].split('.')
    for k in range(len(parts), 0, -1):
        cand = Path(w) / ('/'.join(parts[:k]) + '.py')
        if cand.exists(): return '/'.join(parts[:k]) + '.py'
    return None

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

def roots_per_test(C, R):
    """functions a test executes that none of the test's other executed functions call (its entry points into the package)"""
    out = []
    for i in range(C.shape[0]):
        S_ = C.indices[C.indptr[i]:C.indptr[i + 1]]
        if len(S_) == 0: out.append(S_); continue
        has_caller = np.asarray(R[S_][:, S_].sum(axis=0)).ravel() > 0
        out.append(S_[~has_caller])
    return out

(PLUG / 'prop_stub.py').write_text(STUB); (PLUG / 'prop_trace2.py').write_text(TRACER); (PLUG / 'prop_sel.py').write_text(SEL_PLUGIN)

# ============================== repositories and environments ==============================
def clone(lib):
    dst = REPOS / lib['name']
    if not (dst / '.git').exists():
        shutil.rmtree(dst, ignore_errors=True)
        rc, out, *_ = sp(['git', 'clone', '-q', '--depth', '1', '--branch', lib['tag'], lib['url'], dst], '/tmp', 600)
        if rc != 0: raise RuntimeError(f"clone {lib['name']}: {out[-300:]}")
    return dst

def test_requirements(root):
    root = Path(root); raw = []
    pp = root / 'pyproject.toml'
    if pp.exists():
        d = tomllib.loads(pp.read_text(errors='replace')); proj = d.get('project', {})
        raw += proj.get('dependencies', [])
        for k, v in proj.get('optional-dependencies', {}).items():
            if 'test' in k.lower(): raw += v
        for k, v in d.get('dependency-groups', {}).items():
            if 'test' in k.lower(): raw += [x for x in v if isinstance(x, str)]
    for f in list(root.glob('requirements*/*test*.*')) + list(root.glob('*requirements*test*.txt')) + list(root.glob('requirements/tests*.in')) \
             + list(root.glob('tests/requirements*.txt')) + list(root.glob('requirements-dev.txt')):
        for line in f.read_text(errors='replace').splitlines():
            line = line.split('#')[0].strip()
            if line and not line.startswith(('-', 'git+', 'http', '.')): raw.append(line)
    if (root / 'setup.py').exists(): raw += _setup_py_reqs(root / 'setup.py')
    own = {p.lower().replace('_', '-') for p in pkgs_of(root)}
    out = []
    for s in raw:
        name = re.split(r'[\s\[<>=!~;]', s.strip(), 1)[0]
        if name and not EXCLUDE_REQ.match(name) and name.lower().replace('_', '-') not in own: out.append(s)
    return sorted(set(out))

def build_env(lib):
    d = ENVS / lib['name']; done = d / '.done.json'
    if done.exists(): return json.load(open(done))
    shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True)
    reqs = test_requirements(REPOS / lib['name']); rc, out = 0, ''
    if reqs:
        rc, out, *_ = sp([PY, '-m', 'pip', 'install', '-q', '--disable-pip-version-check', '--target', d] + reqs, '/tmp', 1800, cap_mb=20000)
    for pat in ('pytest', '_pytest', 'pluggy', 'iniconfig', 'py.py', 'pytest-*', 'pytest_*', 'pluggy-*', 'iniconfig-*', 'bin'):   # keep the system pytest
        for q in d.glob(pat):
            shutil.rmtree(q, ignore_errors=True) if q.is_dir() else q.unlink(missing_ok=True)
    info = dict(name=lib['name'], reqs=reqs, pip_rc=rc, pip_tail=out[-500:]); json.dump(info, open(done, 'w'), indent=1)
    return info

CFG = {}
def env_for(name, w, extra=None):
    w = Path(w); c = CFG.get(name, {})
    env = {k: v for k, v in os.environ.items() if k not in ('COLUMNS', 'PYTHONPATH', 'MPLBACKEND', 'PYTHONSTARTUP')}
    pp = [str(w)] + ([str(w / 'src')] if (w / 'src').is_dir() else []) + [str(PLUG), str(ENVS / name)]
    env.update(PYTHONPATH=os.pathsep.join(pp), PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', PYTHONDONTWRITEBYTECODE='1',
               PROP_STUB_INI=','.join(c.get('ini_stubs') or []), PROP_STUB_MARKERS=','.join(c.get('markers') or []),
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    env.update(extra or {}); return env

def cmd_for(name):
    c = CFG.get(name, {})
    return ([PY, '-m', 'pytest', '-p', 'no:cacheprovider', '-q', '--continue-on-collection-errors', '-p', 'prop_stub']
            + (['-p', 'anyio.pytest_plugin'] if c.get('anyio') else []) + list(c.get('wflags') or []))

def detect_cfg(name, w):
    txt = ' '.join(p.read_text(errors='replace') for p in Path(w).rglob('test*.py'))
    c = dict(anyio=bool(re.search(r'pytest\.mark\.anyio|anyio_backend', txt)), ini_stubs=[], markers=[], wflags=[]); CFG[name] = c
    for _ in range(6):
        rc, out, *_ = sp(cmd_for(name) + ['--co'], w, 900, env_for(name, w))
        ini = set(re.findall(r'Unknown config option: ([\w\-\.]+)', out)) - set(c['ini_stubs'])
        mk = set(re.findall(r"'([\w\.\-]+)' not found in `markers`", out)) - set(c['markers'])
        bad_args = 'unrecognized arguments' in out and ['-o', 'addopts='] != c['wflags'][-2:]
        if not (ini or mk or bad_args): break
        c['ini_stubs'] += sorted(ini); c['markers'] += sorted(mk)
        if bad_args: c['wflags'] += ['-o', 'addopts=']
    return c

def full_run(name, w, tag, traced=False, timeout=1800):
    w = Path(w); xml = w / f'.junit_{tag}.xml'; xml.unlink(missing_ok=True); env = env_for(name, w)
    cmd = cmd_for(name) + [f'--junitxml={xml}']
    if traced:
        cmd += ['-p', 'prop_trace2']; env.update(PROP_TRACE_DIR=str(w), PROP_TRACE_ROOT=str(w), PROP_TRACE_PKG=','.join(pkgs_of(w)))
    rc, out, secs, peak, killed = sp(cmd, w, timeout, env)
    return junit_cases(xml), dict(rc=rc, secs=round(secs, 1), peak_mb=peak, killed=killed, tail=out[-400:])

def run_subset(name, w, sel, timeout, mut_path=None, qual=''):
    w = Path(w); files = sorted({f for f in (key_file(w, k) for k in sel) if f})
    (w / '.sel.txt').write_text('\n'.join(sel)); dep = w / '.depth.jsonl'; dep.unlink(missing_ok=True)
    xml = w / '.junit_sub.xml'; xml.unlink(missing_ok=True); ini = ini_for(w)
    env = env_for(name, w, dict(PROP_SELECT=str(w / '.sel.txt'), PROP_MUT_FILE=str(mut_path.resolve()) if mut_path else '/nonexistent',
                                PROP_MUT_QUAL=qual, PROP_DEPTH_OUT=str(dep), PROP_ROOT=str(w)))
    cmd = cmd_for(name) + ['--rootdir', str(w)] + (['-c', str(ini)] if ini else []) + files + ['-p', 'prop_sel', f'--junitxml={xml}']
    rc, out, secs, peak, killed_by = sp(cmd, w, timeout, env)
    res = junit_cases(xml); st = {k: res.get(k, ('missing',))[0] for k in sel}
    depths = {}
    if dep.exists():
        for line in open(dep):
            k, when, da, dp, exc = json.loads(line); depths[k] = [da, dp, exc]
    return st, depths, dict(rc=rc, secs=round(secs, 1), killed=killed_by, tail=out[-300:])

# ============================== one library: baseline, trace, planted-bug campaign ==============================
def campaign(lib, n_rand=40, n_hub=10, cap=300, timeout=300, log=print):
    name = lib['name']; d = OUTG / name; d.mkdir(exist_ok=True)
    if (d / 'G.json').exists(): return json.load(open(d / 'G.json'))['summary']
    src = clone(lib); info = build_env(lib)
    w = WORK / name; shutil.rmtree(w, ignore_errors=True); shutil.copytree(src, w, symlinks=True, ignore=shutil.ignore_patterns('.git'))
    cfg = detect_cfg(name, w)
    base, mb = full_run(name, w, 'base'); n = Counter(s for s, _ in base.values())
    summ = dict(name=name, tag=lib['tag'], pip_rc=info['pip_rc'], cfg=cfg, passed=n['passed'], failed=n['failed'], error=n['error'],
                skipped=n['skipped'], base_secs=mb['secs'])
    log(f"   {name}: suite {dict(n)} in {mb['secs']}s | cfg {cfg}")
    if n['passed'] < 20: summ['error_msg'] = 'suite did not run: ' + mb['tail']; json.dump(dict(summary=summ), open(d / 'G.json', 'w')); return summ
    tr, mt = full_run(name, w, 'traced', traced=True)
    if not (w / '.trace_meta.json').exists(): summ['error_msg'] = 'no trace: ' + mt['tail']; json.dump(dict(summary=summ), open(d / 'G.json', 'w')); return summ
    keys, funcs, C, meta = load_trace(w); fi = {f: i for i, f in enumerate(funcs)}
    summ.update(fidelity_mismatch=sum(base.get(k, ('missing',))[0] != tr.get(k, ('missing',))[0] for k in set(base) | set(tr)),
                n_tests=len(keys), n_funcs=len(funcs), n_edges=len(meta['edges']))
    ssp.save_npz(d / 'C.npz', C); json.dump({'tests': keys, 'funcs': funcs, 'edges': meta['edges']}, open(d / 'index.json', 'w'))
    passed = {k for k, (s, _) in base.items() if s == 'passed'}; Cc = C.tocsc()
    rng = random.Random(zlib.crc32(name.encode()))
    def ok(i):
        f, q = locate(w, funcs[i])
        return f is not None and not q.endswith('<module>')
    cand = [i for i in range(len(funcs)) if not funcs[i].endswith('<module>') and Cc[:, i].nnz > 0 and ok(i)]
    indeg = Counter(b for a, b in meta['edges'])
    hubs = sorted(cand, key=lambda i: -indeg.get(i, 0))[:n_hub]
    rest = [i for i in cand if i not in set(hubs)]
    sites = [(i, 'hub') for i in hubs] + [(i, 'rand') for i in rng.sample(rest, min(n_rand, len(rest)))]
    union = sorted({r for s, _ in sites for r in Cc[:, s].nonzero()[0].tolist()})
    union = [r for r in union if keys[r] in passed]
    if len(union) > 2 * cap: union = sorted(rng.sample(union, 2 * cap))
    st0, _, _ = run_subset(name, w, [keys[r] for r in union], timeout) if union else ({}, None, None)
    unstable = {k for k, v in st0.items() if v != 'passed'}; summ.update(ctrl_n=len(union), ctrl_bad=len(unstable))
    mutants = []
    for site, cls in sites:
        path, qual = locate(w, funcs[site]); src0 = path.read_text()
        try:
            fn = find_def(ast.parse(src0), qual)
        except SyntaxError:
            fn = None
        if fn is None or isinstance(fn, ast.ClassDef): continue
        pts = mutation_points(fn); rng.shuffle(pts); cover = set(Cc[:, site].nonzero()[0].tolist())
        for kind, node in pts:
            src1 = apply_mutation(src0, kind, node)
            if src1 is None: continue
            sel_rows = sorted(r for r in cover if keys[r] in passed and keys[r] not in unstable)
            if len(sel_rows) > cap: sel_rows = sorted(rng.sample(sel_rows, cap))
            m = dict(site=funcs[site], cls=cls, op=kind, line=node.lineno, n_cover=len(cover), n_sel=len(sel_rows))
            if not sel_rows: mutants.append({**m, 'killed': [], 'skip': 'no covering test'}); break
            sel = [keys[r] for r in sel_rows]; path.write_text(src1)
            try:
                st, depths, mm = run_subset(name, w, sel, timeout, mut_path=path, qual=qual)
            finally:
                path.write_text(src0)
            kill = [r for r, k in zip(sel_rows, sel) if st[k] in ('failed', 'error')]
            m.update(killed=kill, n_missing=sum(v == 'missing' for v in st.values()), secs=mm['secs'], timeout=bool(mm['killed']),
                     depth={str(r): depths.get(keys[r]) for r in kill})
            mutants.append(m); break
    ks = [m for m in mutants if 'skip' not in m and not m.get('timeout')]
    summ.update(n_mut=len(ks), kill_rate=float(np.mean([bool(m['killed']) for m in ks])) if ks else None,
                n_missing=sum(m.get('n_missing', 0) for m in ks))
    json.dump(dict(summary=summ, mutants=mutants), open(d / 'G.json', 'w')); shutil.rmtree(w, ignore_errors=True)
    log(f"   {name}: {len(ks)} mutants, kill rate {summ['kill_rate']}, control unstable {len(unstable)}, missing {summ['n_missing']}")
    return summ

# ============================== analysis (same measures as Steps 10 and 10b) ==============================
SENSITIVE = set()
def analyse_lib(name):
    d = OUTG / name; g = json.load(open(d / 'G.json'))
    if 'mutants' not in g: return None, None, None
    idx = json.load(open(d / 'index.json')); funcs, keys = idx['funcs'], idx['tests']; n = len(funcs); fi = {f: i for i, f in enumerate(funcs)}
    E = np.array(idx['edges'], dtype=np.int64).reshape(-1, 2)
    R = ssp.csr_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(n, n)) if len(E) else ssp.csr_matrix((n, n)); R.data[:] = 1
    c = census(R); gam = c['rho'] + 1.0; amp, tpk = graph_amplification(R, gam); c.update(name=name, amp=amp, tpeak=tpk)
    ms = [m for m in g['mutants'] if 'skip' not in m and not m.get('timeout') and m['n_sel'] > 0 and m['site'] in fi]
    six = [fi[m['site']] for m in ms]
    G = nx.DiGraph(); G.add_nodes_from(range(n)); G.add_edges_from(map(tuple, idx['edges']))
    pr = nx.pagerank(G) if G.number_of_edges() else {i: 0 for i in range(n)}
    bt = nx.betweenness_centrality(G, k=min(200, n), seed=0) if G.number_of_edges() else {i: 0 for i in range(n)}
    indeg = np.asarray(R.sum(axis=0)).ravel(); outdeg = np.asarray(R.sum(axis=1)).ravel(); RT = R.T.tocsr()
    t, norms = transient_sites(R, six, gam)
    A = (gam * ssp.identity(n, format='csc') - R.tocsc())
    rows = []
    for j, (m, v) in enumerate(zip(ms, six)):
        rows.append(dict(name=name, site=m['site'], cls=m['cls'], b=len(m['killed']), frac=len(m['killed']) / m['n_sel'],
                         killed=len(m['killed']) > 0, n_cover=m['n_cover'], indeg=indeg[v], degree=indeg[v] + outdeg[v],
                         pagerank=pr[v], betweenness=bt[v],
                         ancestors=len(breadth_first_order(RT, v, directed=True, return_predecessors=False)) - 1,
                         Gamma=float(norms[:, j].max()), Katz=float(gam * spsolve(A, np.eye(1, n, v).ravel()).sum())))
    # depth profile for fully-run covers
    C = ssp.load_npz(d / 'C.npz').tocsr(); Cc = C.tocsc(); roots = roots_per_test(C, R); pairs = []
    for m, v in zip(ms, six):
        cover = Cc[:, v].nonzero()[0]
        if m['n_cover'] > 300 or len(cover) == 0 or m['n_sel'] < 0.8 * len(cover): continue
        dist = shortest_path(RT, indices=v, unweighted=True, directed=True); killed = set(m['killed'])
        for r in cover:
            rt = roots[r]; dep = dist[rt].min() if len(rt) else np.inf
            pairs.append(dict(name=name, mutant=f"{m['site']}|{m['line']}", depth=float(dep), killed=r in killed))
    return c, pd.DataFrame(rows), pd.DataFrame(pairs)

def report(names, log=print):
    cs, S, P = [], [], []
    for nm in names:
        c, s, p = analyse_lib(nm)
        if c is None: continue
        cs.append(c); S.append(s); P.append(p)
    Gt = pd.DataFrame(cs); S = pd.concat(S, ignore_index=True) if S else pd.DataFrame(); P = pd.concat(P, ignore_index=True) if P else pd.DataFrame()
    Gt.to_csv(OUTG / 'gen_graphs.csv', index=False); S.to_csv(OUTG / 'gen_sites.csv', index=False); P.to_csv(OUTG / 'gen_pairs.csv', index=False)
    summ = pd.DataFrame([json.load(open(OUTG / nm / 'G.json'))['summary'] for nm in names if (OUTG / nm / 'G.json').exists()])
    cols = [c for c in ['name', 'tag', 'passed', 'failed', 'error', 'n_tests', 'n_funcs', 'fidelity_mismatch', 'ctrl_bad', 'n_mut', 'kill_rate', 'n_missing', 'error_msg'] if c in summ]
    log('\nLIBRARIES'); log(summ[cols].to_string(index=False))
    if Gt.empty: return
    log('\n1) RUNTIME CALL GRAPHS'); log(Gt[['name', 'n', 'm', 'reciprocity', 'frac_in_cycles', 'rho', 'departure', 'amp', 'tpeak']].round(3).to_string(index=False))
    log(f"   peak transient amplification ||e^(tM)||_1 at gamma = rho + 1: median {Gt.amp.median():.2f} (eigenvalues predict <= 1)")
    log('\n2) PREDICTORS vs broken tests, Spearman across sites within each library (mean over libraries):')
    out = []
    for p in ['n_cover', 'indeg', 'Katz', 'degree', 'pagerank', 'Gamma', 'ancestors', 'betweenness']:
        rb = [spearmanr(g[p], g.b).correlation for _, g in S.groupby('name') if len(g) >= 10 and g[p].nunique() > 1]
        rf = [spearmanr(g[p], g.frac).correlation for _, g in S.groupby('name') if len(g) >= 10 and g[p].nunique() > 1]
        def part(g):
            ry, rx, rc = g.b.rank().values, g[p].rank().values, g.n_cover.rank().values
            res = lambda y: y - np.polyval(np.polyfit(rc, y, 1), rc)
            return spearmanr(res(rx), res(ry)).correlation
        rp = [part(g) for _, g in S.groupby('name') if len(g) >= 10 and g[p].nunique() > 1 and p != 'n_cover']
        out.append(dict(predictor=p, libs=len(rb), rho_broken=np.nanmean(rb), rho_fraction=np.nanmean(rf),
                        rho_given_coverage=np.nanmean(rp) if rp else np.nan))
    log(pd.DataFrame(out).round(3).to_string(index=False))
    cov = S[S.n_cover > 0]
    log(f"\n3) SILENT covered mutants: {(~cov.killed).sum()}/{len(cov)} ({(~cov.killed).mean():.0%}); kill rate hub {S[S.cls == 'hub'].killed.mean():.2f} vs random {S[S.cls == 'rand'].killed.mean():.2f}")
    if len(P):
        fin = P[np.isfinite(P.depth)]
        t = fin.groupby(fin.depth.clip(upper=6)).killed.agg(['size', 'mean'])
        log('\n4) DEPTH PROFILE  P(covering test fails | hops from its entry point):  ' + '  '.join(f"{int(k)}:{v:.2f}(n={s})" for k, (s, v) in t.iterrows()))
        diffs = [q.loc[q.killed, 'depth'].mean() - q.loc[~q.killed, 'depth'].mean() for _, q in fin.groupby(['name', 'mutant'])
                 if q.killed.any() and (~q.killed).any()]
        if len(diffs) > 10:
            diffs = np.array(diffs)
            log(f"   within-mutant: failing tests {diffs.mean():+.2f} hops vs passing ({len(diffs)} mutants), closer in {(diffs < 0).mean():.0%}, "
                f"Wilcoxon p = {wilcoxon(diffs[diffs != 0]).pvalue:.2e}")
