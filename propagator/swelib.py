# swelib.py - Option 2: the real-bug measurements on independent repositories (SWE-rebench tasks).
# Needs internet (Hugging Face dataset, GitHub clones, PyPI, uv-managed Python builds). CPU only. No competition data.
import os, re, sys, shlex, json, time, shutil, subprocess, ast, gc, tempfile, random, zlib
import xml.etree.ElementTree as ET
import concurrent.futures as cf
from collections import defaultdict
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as ssp
from scipy.sparse.csgraph import shortest_path

WORKDIR = Path(os.environ.get('PROP_WORKDIR', '/kaggle/working'))
SW = WORKDIR / 'swe'; REPOS, ENVS, PLUG, OUTS = SW / 'repos', SW / 'envs', SW / 'plug', SW / 'out'
for _p in (REPOS, ENVS, PLUG, OUTS): _p.mkdir(parents=True, exist_ok=True)
MEM_CAP_MB = 6000; PY = sys.executable
SUITE_TIMEOUT = int(os.environ.get('SWE_SUITE_TIMEOUT', 1200)); VERSION = 3

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

def changed_lines(patch):
    files = defaultdict(set); cur = old = None
    for line in patch.replace('\r\n', '\n').split('\n'):
        if line.startswith('diff --git'):
            m = re.match(r'diff --git a/(\S+) b/(\S+)', line); cur = m.group(2) if m else None; old = None; continue
        if line.startswith('--- '):
            m = re.match(r'--- (?:a/)?(\S+)', line); path = m.group(1) if m else None
            cur = None if path in (None, '/dev/null') else path; old = None; continue
        if line.startswith('+++ '):
            m = re.match(r'\+\+\+ (?:b/)?(\S+)', line); path = m.group(1) if m else None
            if cur is None and path not in (None, '/dev/null'): cur = path
            continue
        if line.startswith('@@'):
            m = re.match(r'@@ -(\d+)', line); old = int(m.group(1)) if m else None; continue
        if cur is None or old is None or line.startswith('\\'): continue
        if line.startswith('-'): files[cur].add(old); old += 1
        elif line.startswith('+'): files[cur].update({max(old - 1, 1), old})
        else: old += 1
    return files

def _spans(src):
    out = []
    def walk(node, prefix):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                q = prefix + ch.name
                out.append((min([ch.lineno] + [d.lineno for d in ch.decorator_list]), ch.end_lineno, q))
                walk(ch, q + '.')
            else:
                walk(ch, prefix)
    walk(ast.parse(src), '')
    return out

def modname(rel):
    parts = rel[:-3].split('/')
    if parts and parts[0] == 'src': parts = parts[1:]
    if parts and parts[-1] == '__init__': parts = parts[:-1]
    return '.'.join(parts)


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

def _isq(code):
    q = getattr(code, "co_qualname", None)
    if q is None: return code.co_name == _MQ.split(".")[-1]          # Python < 3.11
    return q.replace("<locals>.", "") == _MQ

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
               if os.path.realpath(f.f_code.co_filename) == _MF and _isq(f.f_code)), None)
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


def roots_per_test(C, R):
    """functions a test executes that none of the test's other executed functions call (its entry points into the package)"""
    out = []
    for i in range(C.shape[0]):
        S_ = C.indices[C.indptr[i]:C.indptr[i + 1]]
        if len(S_) == 0: out.append(S_); continue
        has_caller = np.asarray(R[S_][:, S_].sum(axis=0)).ravel() > 0
        out.append(S_[~has_caller])
    return out


# tracer: sys.monitoring on Python >= 3.12, sys.setprofile fallback on 3.11 (both need co_qualname)
TRACER = r'''
import os, sys, json, threading, ast
import pytest
_ROOT = os.path.realpath(os.environ.get("PROP_TRACE_ROOT", os.getcwd())) + os.sep
_PKGS = set(filter(None, os.environ.get("PROP_TRACE_PKG", "").split(",")))
_DIR = os.environ.get("PROP_TRACE_DIR", ".")
_ids, _nid, _edges, _collect, _fcache, _neg = {}, {}, set(), set(), {}, set()
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

_spanc = {}
def _spanmap(fname):
    sp = _spanc.get(fname)
    if sp is None:
        sp = []
        try:
            tree = ast.parse(open(fname, encoding="utf-8", errors="replace").read())
        except Exception:
            tree = None
        def walk(node, pre):
            for ch in ast.iter_child_nodes(node):
                if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    q = pre + ch.name
                    start = min([ch.lineno] + [d.lineno for d in ch.decorator_list])
                    sp.append((start, getattr(ch, "end_lineno", ch.lineno), q)); walk(ch, q + ".")
                else:
                    walk(ch, pre)
        if tree is not None: walk(tree, "")
        _spanc[fname] = sp
    return sp

def _qualname(code):
    q = getattr(code, "co_qualname", None)
    if q is not None:
        return q
    ln, best = code.co_firstlineno, None                 # Python < 3.11: innermost def/class containing the code object
    for s, e, qn in _spanmap(code.co_filename):
        if s <= ln <= e and (best is None or s >= best[0]): best = (s, qn)
    return best[1] if best else "<module>"

def _ident(code):
    i = _ids.get(code)
    if i is None:
        mod = _modparts(code.co_filename)
        if mod is None:
            return None
        q = _qualname(code).replace("<locals>.", "")
        q = "<module>" if q.startswith("<") else q.split(".<")[0]
        i = _ids[code] = _nid.setdefault(mod + "." + q, len(_nid))
    return i

def _record(i, f):
    _cur.add(i)
    for _ in range(6):
        if f is None: break
        j = _ids.get(f.f_code)
        if j is not None:
            if j != i: _edges.add((j, i))
            break
        f = f.f_back

if hasattr(sys, "monitoring"):
    _mon = sys.monitoring
    _TOOL = next(i for i in (2, 3, 4, 5) if _mon.get_tool(i) is None)
    def _start(code, offset):
        i = _ident(code)
        if i is None:
            return _mon.DISABLE
        _record(i, sys._getframe(1).f_back)
    _mon.use_tool_id(_TOOL, "prop_trace2")
    _mon.register_callback(_TOOL, _mon.events.PY_START, _start)
    _mon.set_events(_TOOL, _mon.events.PY_START)
    def _stop(): _mon.set_events(_TOOL, 0)
else:
    def _prof(frame, event, arg):
        if event != "call": return
        code = frame.f_code
        if code in _neg: return
        i = _ident(code)
        if i is None:
            _neg.add(code); return
        _record(i, frame.f_back)
    sys.setprofile(_prof); threading.setprofile(_prof)
    def _stop(): sys.setprofile(None); threading.setprofile(None)

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
    _stop(); _fh.close()
    nodes = [None] * len(_nid)
    for n, i in _nid.items(): nodes[i] = n
    with open(os.path.join(_DIR, ".trace_meta.json"), "w") as fh:
        json.dump({"nodes": nodes, "collect": sorted(_collect), "edges": sorted(_edges)}, fh)
'''



# outcome recorder: test outcomes keyed by the SAME node-id key as selection and tracing (no junit name join)
OUT_PLUGIN = r'''
import os, json
_P = os.environ.get("PROP_OUTCOMES"); _st = {}
def _key(nid):
    p = nid.split("::"); f = p[0][:-3].replace("/", ".") if p[0].endswith(".py") else p[0]
    return ".".join([f] + p[1:-1]) + "::" + p[-1]
def pytest_runtest_logreport(report):
    k = _key(report.nodeid); prev = _st.get(k)
    if report.failed: s = "failed" if report.when == "call" else "error"
    elif report.skipped: s = "skipped"
    elif report.when == "call": s = "passed"
    else: return
    if prev in ("failed", "error"): return
    _st[k] = s
def pytest_sessionfinish(session, exitstatus):
    if _P:
        with open(_P, "w") as fh: json.dump(_st, fh)
'''
(PLUG / 'prop_trace2.py').write_text(TRACER); (PLUG / 'prop_sel.py').write_text(SEL_PLUGIN); (PLUG / 'prop_out.py').write_text(OUT_PLUGIN)

# ============================== tools ==============================
def ensure_uv():
    uv = shutil.which('uv')
    if uv: return uv
    sp([PY, '-m', 'pip', 'install', '-q', 'uv'], '/tmp', 600, cap_mb=20000)
    uv = shutil.which('uv') or str(Path(sys.executable).parent / 'uv')
    if not Path(uv).exists():
        import uv as _uv; uv = _uv.find_uv_bin()
    return uv

def load_swe_rebench(split='test'):
    cache = SW / f'swe_rebench_{split}.pkl'
    if cache.exists(): return pd.read_pickle(cache)
    try:
        from datasets import load_dataset
    except ImportError:
        sp([PY, '-m', 'pip', 'install', '-q', 'datasets'], '/tmp', 900, cap_mb=20000); from datasets import load_dataset
    df = load_dataset('nebius/SWE-rebench', split=split).to_pandas()
    for c in ('install_config', 'meta', 'FAIL_TO_PASS', 'PASS_TO_PASS'):
        if c in df: df[c] = df[c].apply(lambda v: v if isinstance(v, str) else json.dumps(_plain(v), default=str))
    df.to_pickle(cache); return df

def _plain(v):                                          # numpy arrays / structs from to_pandas() -> plain JSON values
    if isinstance(v, dict): return {k: _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)) or hasattr(v, 'tolist') and not isinstance(v, (str, bytes)):
        v = v.tolist() if hasattr(v, 'tolist') else v
        return [_plain(x) for x in v] if isinstance(v, list) else v
    return v

def _j(v, default):
    if isinstance(v, (dict, list)): return v
    try: return json.loads(v) if v else default
    except Exception: return default

EXCLUDE = {'tiangolo/fastapi', 'fastapi/fastapi', 'textualize/rich', 'psf/requests', 'encode/httpx'}   # the competition repositories
def select_tasks(df, n=150, per_repo=2, seed=0, pythons=('3.9', '3.10', '3.11', '3.12', '3.13')):
    rows = []
    for r in df.itertuples():
        if r.repo.lower() in EXCLUDE: continue
        cfg = _j(r.install_config, {})
        if str(cfg.get('python')) not in pythons or cfg.get('pre_install'): continue
        if 'pytest' not in str(cfg.get('log_parser', 'pytest')).lower(): continue
        f2p = _j(r.FAIL_TO_PASS, [])
        if not (1 <= len(f2p) <= 10): continue
        files = [f for f in changed_lines(r.patch) if f.endswith('.py') and not TESTFILE.search(f)]
        if not (1 <= len(files) <= 2): continue
        rows.append(dict(instance_id=r.instance_id, repo=r.repo, created_at=str(r.created_at)))
    C = pd.DataFrame(rows)
    if C.empty: return []
    C = C.sample(frac=1, random_state=seed).groupby('repo').head(per_repo)
    return C.instance_id.head(n).tolist()

# ============================== per-task environment ==============================
def checkout(t):
    d = REPOS / t['instance_id']
    if (d / '.prop_ok').exists(): return d
    shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True)
    url = f"https://github.com/{t['repo']}.git"
    for cmd in (['git', 'init', '-q'], ['git', 'remote', 'add', 'origin', url],
                ['git', 'fetch', '-q', '--depth', '1', 'origin', t['base_commit']], ['git', 'checkout', '-q', 'FETCH_HEAD']):
        rc, out, *_ = sp(cmd, d, 900)
        if rc != 0: raise RuntimeError(f'git {cmd[1]}: {out[-300:]}')
    (d / '.prop_ok').write_text('1'); return d

def build_env(t, d, uv):
    cfg = _j(t['install_config'], {}); e = ENVS / t['instance_id']; py = e / 'bin' / 'python'
    if (e / '.prop_ok').exists(): return py, None
    shutil.rmtree(e, ignore_errors=True)
    env = dict(os.environ, UV_PYTHON_DOWNLOADS='automatic')
    rc, out, *_ = sp([uv, 'venv', '-q', '--python', str(cfg.get('python')), e], '/tmp', 600, env=env, cap_mb=20000)
    if rc != 0: return None, 'venv: ' + out[-300:]
    pipi = [uv, 'pip', 'install', '-q', '--python', py]
    reqs = t.get('requirements') or ''
    if isinstance(reqs, str) and reqs.strip():
        lines = [l.strip() for l in reqs.splitlines() if l.strip() and not l.strip().startswith(('-e', '#', '--')) and ' @ file:' not in l]
        rf = e / 'reqs.txt'; rf.write_text('\n'.join(lines))
        rc, out, *_ = sp(pipi + ['-r', rf], d, 1800, env=env, cap_mb=20000)
        if rc != 0:                                    # one bad pin should not sink the task: install line by line
            for l in lines: sp(pipi + [l], d, 600, env=env, cap_mb=20000)
    pkgs = cfg.get('pip_packages') or []
    if isinstance(pkgs, str): pkgs = [pkgs]
    pkgs = [x for p in pkgs for x in shlex.split(str(p))]
    if pkgs:
        rc, out, *_ = sp(pipi + list(pkgs), d, 1800, env=env, cap_mb=20000)
        if rc != 0: return None, 'pip_packages: ' + out[-300:]
    inst = str(cfg.get('install') or 'pip install -e .')
    for part in [p.strip() for p in inst.split('&&') if p.strip()]:
        if part.startswith(('pip install', 'python -m pip install')):
            args = [a for a in shlex.split(part.split('install', 1)[1]) if a not in ('-q', '--quiet', '-v', '--no-build-isolation')]
            rc, out, *_ = sp(pipi + args, d, 1800, env=env, cap_mb=20000)
            if rc != 0: return None, 'install: ' + out[-300:]
    sp(pipi + ['pytest'], d, 600, env=env, cap_mb=20000) if sp([py, '-c', 'import pytest'], d, 60)[0] != 0 else None
    (e / '.prop_ok').write_text('1'); return py, None

def env_for(t, d, extra=None):
    cfg = _j(t['install_config'], {}); d = Path(d)
    env = {k: v for k, v in os.environ.items() if k not in ('COLUMNS', 'PYTHONPATH', 'MPLBACKEND', 'PYTHONSTARTUP', 'VIRTUAL_ENV')}
    pp = [str(d)] + ([str(d / 'src')] if (d / 'src').is_dir() else []) + [str(PLUG)]
    env.update(PYTHONPATH=os.pathsep.join(pp), PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    for k, v in (cfg.get('env_vars') or {}).items(): env[str(k)] = str(v)
    env.update(extra or {}); return env

def cmd_for(py):
    return [py, '-m', 'pytest', '-p', 'no:cacheprovider', '-p', 'no:randomly', '-p', 'no:xdist', '-q', '--continue-on-collection-errors']

def run(t, py, d, tag, targets=(), traced=False, timeout=None, select=None, mut=None):
    d = Path(d); xml = d / f'.junit_{tag}.xml'; xml.unlink(missing_ok=True); oc = d / f'.out_{tag}.json'; oc.unlink(missing_ok=True)
    extra = {'PROP_OUTCOMES': str(oc)}
    cmd = cmd_for(py) + list(targets) + ['-p', 'prop_out', f'--junitxml={xml}']
    if traced:
        cmd += ['-p', 'prop_trace2']; extra.update(PROP_TRACE_DIR=str(d), PROP_TRACE_ROOT=str(d), PROP_TRACE_PKG=','.join(pkgs_of(d)))
    if select is not None:
        ini = ini_for(d); (d / '.sel.txt').write_text('\n'.join(select)); dep = d / '.depth.jsonl'; dep.unlink(missing_ok=True)
        cmd = cmd_for(py) + ['--rootdir', str(d)] + (['-c', str(ini)] if ini else []) + list(targets) + ['-p', 'prop_sel', '-p', 'prop_out', f'--junitxml={xml}']
        extra.update(PROP_SELECT=str(d / '.sel.txt'), PROP_DEPTH_OUT=str(dep), PROP_ROOT=str(d),
                     PROP_MUT_FILE=str(mut[0].resolve()) if mut else '/nonexistent', PROP_MUT_QUAL=mut[1] if mut else '')
    rc, out, secs, peak, killed = sp(cmd, d, timeout or SUITE_TIMEOUT, env_for(t, d, extra))
    res = junit_cases(xml)                              # junit only as a fallback; node-id outcomes win
    try: res.update({k: (v, '') for k, v in json.load(open(oc)).items()})
    except Exception: pass
    return res, dict(rc=rc, secs=round(secs, 1), killed=killed, tail=out[-300:])

def git_apply(patch, d):
    p = patch if patch.endswith('\n') else patch + '\n'
    rc, out, *_ = sp(['git', 'apply', '-p1', '-'], d, 120, inp=p)
    return rc == 0, out[-200:]

def git_reset(d):
    sp(['git', 'checkout', '-q', '--', '.'], d, 120); sp(['git', 'clean', '-qfd', '-e', '.prop_ok'], d, 120)

# ============================== one task ==============================
IDENT = re.compile(r'[A-Za-z_][A-Za-z0-9_]{2,}')
def mentioned(func, text):
    toks = set(IDENT.findall(text or '')); p = func.split('.')
    return p[-1] in toks or (len(p) > 1 and p[-2] in toks and p[-1] != '<module>')

def ochiai(C, fail):
    ef = np.asarray(C[fail].sum(axis=0)).ravel(); tot = np.asarray(C.sum(axis=0)).ravel()
    den = np.sqrt(len(fail) * tot); den[den == 0] = 1; return ef / den
def best_rank(s, targets):
    order = (-s).argsort(kind='stable'); pos = np.empty(len(s), int); pos[order] = np.arange(1, len(s) + 1)
    return int(min(pos[x] for x in targets))

def task_pipeline(t, uv, n_mut_per_f=4, max_f=3, cap=300):
    iid = t['instance_id']; o = OUTS / iid; o.mkdir(exist_ok=True)
    if (o / 'summary.json').exists():
        r0 = json.load(open(o / 'summary.json'))
        if r0.get('v') == VERSION: return r0
    row = dict(task=iid, repo=t['repo'], python=_j(t['install_config'], {}).get('python'), v=VERSION)
    def done(**kw):
        row.update(kw); json.dump(row, open(o / 'summary.json', 'w'), default=str); return row
    try:
        d = checkout(t); py, err = build_env(t, d, uv)
        if py is None: return done(status='env-failed', msg=err)
        f2p_keys = {nid2key(x) for x in _j(t['FAIL_TO_PASS'], [])}
        p2p_keys = {nid2key(x) for x in _j(t.get('PASS_TO_PASS'), [])}; harness = f2p_keys | p2p_keys   # what the benchmark harness runs
        f2p_files = sorted({x.split('::')[0] for x in _j(t['FAIL_TO_PASS'], []) if x.split('::')[0].endswith('.py')})
        # 1) reproduction on the F2P files: fail on base + test patch, pass with the fix
        git_reset(d)
        if not git_apply(t['test_patch'], d)[0]: return done(status='test_patch-failed')
        rb, _ = run(t, py, d, 'rb', targets=f2p_files, timeout=600)
        if not git_apply(t['patch'], d)[0]: return done(status='patch-failed')
        rg, _ = run(t, py, d, 'rg', targets=f2p_files, timeout=600)
        st = lambda res, k: res.get(k, ('missing',))[0]
        fail_base = [k for k in f2p_keys if st(rb, k) != 'passed']; pass_gold = [k for k in f2p_keys if st(rg, k) == 'passed']
        if not (fail_base and len(pass_gold) == len(f2p_keys)):
            return done(status='not-reproduced', f2p=len(f2p_keys), fail_base=len(fail_base), pass_gold=len(pass_gold))
        # 2) real-bug response of the pre-existing suite: base vs fix, WITHOUT the test patch
        git_reset(d); base, mb = run(t, py, d, 'base')
        git_apply(t['patch'], d); gold, mg = run(t, py, d, 'gold')
        if mb['killed'] or mg['killed']: return done(status='suite-timeout')
        json.dump({'base': {k: v[0] for k, v in base.items()}, 'gold': {k: v[0] for k, v in gold.items()}}, open(o / 'suite.json', 'w'))
        bad = lambda s: s in ('failed', 'error')            # a test must exist in BOTH runs (tmp-path / random ids drop out as 'missing')
        d_raw = [k for k, (s, _) in gold.items() if s == 'passed' and bad(base.get(k, ('missing',))[0])]
        r_raw = [k for k, (s, _) in base.items() if s == 'passed' and bad(gold.get(k, ('missing',))[0])]
        n_missing_ids = len(set(base) ^ set(gold))
        d_old, rev_old = d_raw, r_raw
        if d_raw or r_raw:                                   # confirm: 2 more runs on each side, keep only consistent flips
            cand = sorted(set(d_raw) | set(r_raw)); files = sorted({f for f in (key_file(d, k) for k in cand) if f})
            git_reset(d); B = [run(t, py, d, f'cb{i}', targets=files, timeout=600, select=cand)[0] for i in range(2)]
            git_apply(t['patch'], d); G = [run(t, py, d, f'cg{i}', targets=files, timeout=600, select=cand)[0] for i in range(2)]
            d_old = [k for k in d_raw if all(bad(st(b, k)) for b in B) and all(st(g, k) == 'passed' for g in G)]
            rev_old = [k for k in r_raw if all(st(b, k) == 'passed' for b in B) and all(bad(st(g, k)) for g in G)]
        row.update(d_raw=len(d_raw), rev_raw=len(r_raw), n_missing_ids=n_missing_ids)
        # 3) traced fixed code + test patch: coverage, runtime graph
        git_apply(t['test_patch'], d); tr, mt = run(t, py, d, 'tr', traced=True, timeout=2 * SUITE_TIMEOUT)
        if mt['killed'] or not (d / '.trace_meta.json').exists(): return done(status='trace-failed', msg=mt['tail'])
        keys, funcs, C, meta = load_trace(d); fi = {f: i for i, f in enumerate(funcs)}; n = len(funcs)
        Fs = sorted(patch_functions_at(d, t['patch'])); Fp = [f for f in Fs if not f.endswith('.<module>')]
        F_idx = [fi[f] for f in Fp if f in fi]
        old_keys = {k for k in gold}; fail_rows = [i for i, k in enumerate(keys) if k in f2p_keys]
        Cc = C.tocsc(); hitsF = (np.asarray(C[:, F_idx].sum(1)).ravel() > 0) if F_idx else np.zeros(len(keys), bool)
        old_hit = int(sum(1 for i, k in enumerate(keys) if hitsF[i] and k in old_keys and k not in f2p_keys))
        ssp.save_npz(o / 'C.npz', C); json.dump({'tests': keys, 'funcs': funcs, 'edges': meta['edges']}, open(o / 'index.json', 'w'))
        row.update(status='reproduced', n_tests_old=len(gold), d_old=len(d_old), d_old_list=d_old[:5], rev_old=len(rev_old), nF=len(Fp), F_exec=len(F_idx),
                   old_tests_hit_F=old_hit, n_f2p=len(fail_rows), f2p_hit_F=int(hitsF[fail_rows].sum()) if fail_rows else 0, suite_secs=mb['secs'])
        # 4) localization: Ochiai from the F2P tests, then one hop over the runtime graph
        if F_idx and fail_rows:
            E = np.array(meta['edges'], dtype=np.int64).reshape(-1, 2)
            R = ssp.csr_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(n, n)); R.data[:] = 1
            Au = ((R + R.T) > 0).astype(float); deg = np.asarray(Au.sum(axis=1)).ravel(); deg[deg == 0] = 1
            PT = (ssp.diags(1 / deg) @ Au).T.tocsr(); s0 = ochiai(C, fail_rows)
            roots = set()
            for r in roots_per_test(C[fail_rows], R): roots |= set(r.tolist())
            dist = shortest_path(Au, indices=F_idx, unweighted=True, directed=False)
            row.update(rank_cov=best_rank(s0, F_idx), rank_k1=best_rank(s0 + PT @ s0, F_idx),
                       d_sf=float(dist[:, sorted(roots)].min()) if roots else float('inf'),
                       mentioned=any(mentioned(funcs[v], t.get('problem_statement', '')) for v in F_idx))
        # 5) grader observation at the fix site: planted bugs at F*, caught by any covering test vs by the F2P tests
        passed = {k for k, (s, _) in tr.items() if s == 'passed'}; rng = random.Random(zlib.crc32(iid.encode())); muts = []
        for v in F_idx[:max_f]:
            path, qual = locate(d, funcs[v])
            if path is None: continue
            src0 = path.read_text()
            try: fn = find_def(ast.parse(src0), qual)
            except SyntaxError: fn = None
            if fn is None or isinstance(fn, ast.ClassDef): continue
            pts = mutation_points(fn); rng.shuffle(pts); made = 0
            cover = [r for r in Cc[:, v].nonzero()[0].tolist() if keys[r] in passed]
            graded = [r for r in cover if keys[r] in harness]; other = [r for r in cover if keys[r] not in harness]
            if len(other) > cap: other = sorted(rng.sample(other, cap))           # harness tests that cover F* are never sampled away
            sel = sorted(set(keys[r] for r in graded + other) | (f2p_keys & passed))
            if not sel: continue
            files = sorted({f for f in (key_file(d, k) for k in sel) if f})
            for kind, node in pts:
                if made >= n_mut_per_f: break
                src1 = apply_mutation(src0, kind, node)
                if src1 is None: continue
                made += 1; path.write_text(src1)
                try:
                    res, mm = run(t, py, d, 'mut', targets=files, timeout=600, select=sel, mut=(path, qual))
                finally:
                    path.write_text(src0)
                killed = {k for k in sel if st(res, k) in ('failed', 'error')}
                muts.append(dict(site=funcs[v], op=kind, n_sel=len(sel), n_cover=len(cover), n_cover_harness=len(graded),
                                 any=bool(killed), grader=bool(killed & f2p_keys), harness=bool(killed & harness),
                                 n_missing=sum(st(res, k) == 'missing' for k in sel), timeout=bool(mm['killed'])))
        json.dump(muts, open(o / 'mutants.json', 'w'))
        ok = [m for m in muts if not m['timeout']]
        row.update(n_mut=len(ok), caught_any=float(np.mean([m['any'] for m in ok])) if ok else None,
                   caught_grader=float(np.mean([m['grader'] for m in ok])) if ok else None,
                   caught_harness=float(np.mean([m['harness'] for m in ok])) if ok else None, n_p2p=len(p2p_keys),
                   sel_missing=int(sum(m['n_missing'] for m in ok)), sel_total=int(sum(m['n_sel'] for m in ok)))
        return done()
    except Exception as e:
        return done(status='crash', msg=repr(e)[:300])
    finally:
        shutil.rmtree(ENVS / iid, ignore_errors=True); shutil.rmtree(REPOS / iid, ignore_errors=True); gc.collect()

def patch_functions_at(root, patch):
    """F* on the CURRENT tree (fixed code): functions whose new-side lines the patch touches, plus old-side mapping"""
    root = Path(root); pk = set(pkgs_of(root)); F = set()
    for rel, lines in changed_lines(patch).items():
        if not rel.endswith('.py') or TESTFILE.search(rel) or modname(rel).split('.')[0] not in pk or not (root / rel).exists(): continue
        try: spans = _spans((root / rel).read_text(errors='replace'))
        except SyntaxError: continue
        new_lines = set()
        cur = None; nl = None
        for line in patch.replace('\r\n', '\n').split('\n'):
            if line.startswith('+++ '): cur = line[6:] if line.startswith('+++ b/') else line[4:]; continue
            if line.startswith('@@'):
                m = re.match(r'@@ -\d+(?:,\d+)? \+(\d+)', line); nl = int(m.group(1)) if m else None; continue
            if cur != rel or nl is None or line.startswith('\\'): continue
            if line.startswith('+'): new_lines.add(nl); nl += 1
            elif line.startswith('-'): new_lines.add(nl)
            else: nl += 1
        mod = modname(rel)
        for ln in new_lines:
            inside = [s for s in spans if s[0] <= ln <= s[1]]
            F.add(f'{mod}.{max(inside)[2]}' if inside else f'{mod}.<module>')
    return F

def run_all(task_ids, df, workers=2, budget_s=None, log=print):
    uv = ensure_uv(); T = {r['instance_id']: r for r in df[df.instance_id.isin(task_ids)].to_dict('records')}
    rows, t0 = [], time.time()
    def guarded(t):                                     # tasks not started before the time budget are skipped (rerun to resume)
        if budget_s and time.time() - t0 > budget_s and not (OUTS / t['instance_id'] / 'summary.json').exists():
            return dict(task=t['instance_id'], repo=t['repo'], status='skipped-budget')
        return task_pipeline(t, uv)
    with cf.ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(guarded, T[i]): i for i in task_ids if i in T}
        for k, fu in enumerate(cf.as_completed(futs)):
            r = fu.result(); rows.append(r)
            log(f"   [{k + 1}/{len(futs)}] {r['task'][:50]:50s} {r.get('status'):15s} {time.time() - t0:5.0f}s  {str(r.get('msg', ''))[:80]}", flush=True)
    return pd.DataFrame(rows)

def boot_ci(df, fn, B=2000, seed=0):
    """95% CI resampling REPOSITORIES (tasks nested in repos)"""
    rng = np.random.default_rng(seed); g = {r: q for r, q in df.groupby('repo')}; reps = list(g); vals = []
    for _ in range(B):
        q = pd.concat([g[r] for r in rng.choice(reps, len(reps))])
        try: vals.append(fn(q))
        except Exception: pass
    vals = np.array([v for v in vals if v is not None and np.isfinite(v)])
    return (np.nan, np.nan) if len(vals) == 0 else tuple(np.percentile(vals, [2.5, 97.5]))
def _ci(df, fn, fmt='{:.0%}'):
    lo, hi = boot_ci(df, fn); return f"{fmt.format(fn(df))} [{fmt.format(lo)}, {fmt.format(hi)}]"

def report(S):
    S.to_csv(OUTS / 'swe_tasks.csv', index=False)
    S = S[S.status != 'skipped-budget']
    print('\nSTATUS', S.status.value_counts().to_dict())
    R = S[S.status == 'reproduced'].copy()
    if R.empty: return
    print(f"reproduced: {len(R)} tasks from {R.repo.nunique()} repositories (Python {R.python.astype(str).value_counts().to_dict()}); "
          f"brackets = 95% CI, bootstrap over repositories")
    cov = R[R.old_tests_hit_F > 0]
    print(f"\n1) REAL-BUG SILENCE: the pre-existing suite executes F* in {len(cov)}/{len(R)} tasks (median {cov.old_tests_hit_F.median():.0f} old tests, "
          f"IQR {cov.old_tests_hit_F.quantile(.25):.0f}-{cov.old_tests_hit_F.quantile(.75):.0f})")
    print(f"   silent (no old test confirmed failing on the bug and passing on the fix): {(cov.d_old == 0).sum()}/{len(cov)} = "
          + _ci(cov, lambda q: (q.d_old == 0).mean()) + f"   [unconfirmed raw flips: {(cov.d_raw > 0).sum()} tasks]")
    for _, r in R[R.d_old > 0].iterrows(): print(f'   d_old>0 {r.task}: {r.d_old} {r.d_old_list[:2]}')
    print(f"   confirmed reverse flips (old test passes on the bug, fails on the fix): {(R.rev_old > 0).sum()}/{len(R)} tasks [raw {(R.rev_raw > 0).sum()}]")
    print(f"   tasks where a FAIL_TO_PASS test executes F*: {(R.f2p_hit_F > 0).mean():.0%}; F* executed by any test: {(R.F_exec > 0).mean():.0%}")
    if 'rank_k1' in R:
        L = R.dropna(subset=['rank_k1'])
        print(f"\n2) LOCALIZATION ({len(L)} tasks): top-5 coverage only " + _ci(L, lambda q: (q.rank_cov <= 5).mean()) +
              " -> one hop on the runtime graph " + _ci(L, lambda q: (q.rank_k1 <= 5).mean()))
        print(f"   top-1 {(L.rank_cov <= 1).mean():.0%} -> {(L.rank_k1 <= 1).mean():.0%}; one hop better / same / worse: "
              f"{(L.rank_k1 < L.rank_cov).sum()} / {(L.rank_k1 == L.rank_cov).sum()} / {(L.rank_k1 > L.rank_cov).sum()}")
        fin = L[np.isfinite(L.d_sf.astype(float))]
        if len(fin) > 8:
            from scipy.stats import spearmanr
            print(f"   symptom-to-fix distance: median {fin.d_sf.median():.0f} hops; Spearman(d, rank) = "
                  + _ci(fin, lambda q: spearmanr(q.d_sf, q.rank_k1).correlation, '{:+.2f}') + f"; F* named in the issue: {L.mentioned.mean():.0%}")
    if 'caught_any' in R:
        M = R.dropna(subset=['caught_any'])
        print(f"\n3) GRADER OBSERVATION at the fix site ({int(M.n_mut.sum())} planted bugs, {len(M)} tasks; task-weighted means)")
        print("   caught by any covering test      " + _ci(M, lambda q: q.caught_any.mean()))
        print("   caught by FAIL_TO_PASS only      " + _ci(M, lambda q: q.caught_grader.mean()))
        print("   caught by the full harness       " + _ci(M, lambda q: q.caught_harness.mean()))
        print("   ACCEPTED by the harness (false-positive reward) " + _ci(M, lambda q: 1 - q.caught_harness.mean()))
        print(f"   tasks where F2P catch none: {(M.caught_grader == 0).mean():.0%}; harness catches none: {(M.caught_harness == 0).mean():.0%}; "
              f"PASS_TO_PASS listed: {(M.n_p2p > 0).mean():.0%}")
        print(f"   test-id match rate in mutant runs: {1 - M.sel_missing.sum() / max(1, M.sel_total.sum()):.4f} "
              f"({int(M.sel_missing.sum())} of {int(M.sel_total.sum())} selected tests missing)")
