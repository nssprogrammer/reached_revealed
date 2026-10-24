# proplib.py - Measured Propagator experiments: self-contained library (fresh notebook, CPU only).
# Needs only the competition data attached as notebook input. Everything else is built here.
import os, re, sys, json, time, shutil, subprocess, tarfile, ast, gc, tempfile, zipfile, tomllib
import xml.etree.ElementTree as ET
import concurrent.futures as cf
from collections import Counter, defaultdict
from email.parser import Parser
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as ssp
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import parse_wheel_filename, canonicalize_name
from packaging.tags import sys_tags
from packaging.markers import default_environment

# ============================== paths ==============================
WORKDIR = Path(os.environ.get('PROP_WORKDIR', '/kaggle/working'))
TMP = Path(os.environ.get('PROP_TMP', '/tmp/prop'))
OUT, ENVS = WORKDIR / 'propagator', WORKDIR / 'propenvs'
PLUG, SHIM, SHIM2 = WORKDIR / 'propplug', WORKDIR / 'propshim', WORKDIR / 'propshim_fallback'
SNAPS, WORK = TMP / 'snaps', TMP / 'work'
RD, TD = OUT / 'repro', OUT / 'tasks'
for _p in (OUT, ENVS, PLUG, SHIM, SHIM2, SNAPS, WORK, RD, TD): _p.mkdir(parents=True, exist_ok=True)
MEM_CAP_MB = int(os.environ.get('PROP_MEM_CAP_MB', 6000))
PY = sys.executable
TESTFILE = re.compile(r'(^|/)(tests?|testing)/.*\.py$|(^|/)test_[^/]*\.py$|_test\.py$')
NONPKG = {'tests', 'test', 'testing', 'docs', 'docs_src', 'scripts', 'examples', 'benchmarks', 'site', 'build', 'dist'}
PINS = {'fastapi': {'starlette': '==0.52.1', 'pydantic': '==2.13.4', 'pydantic-core': '==2.46.4'}}
TEST_EXTRA = ['pytest', 'pluggy', 'iniconfig', 'anyio']
REF_GATE = {'fastapi': dict(passed=2672, failed=1, error=102), 'rich': dict(passed=943, failed=9, error=0),
            'requests': dict(passed=414, failed=0, error=204), 'httpx': dict(passed=128, failed=0, error=0)}

def find_data():
    if os.environ.get('PROP_DATA'): return Path(os.environ['PROP_DATA'])
    base = Path('/kaggle/input')
    for pat in ('*/snapshots', '*/*/snapshots', '*/*/*/snapshots', '*/*/*/*/snapshots'):
        for d in sorted(base.glob(pat)):
            if d.is_dir() and any(d.glob('*.tgz')): return d.parent
    raise FileNotFoundError('Attach the competition data: no folder with snapshots/*.tgz under /kaggle/input')
DATA = find_data()

def load_tasks():
    for f in sorted(DATA.rglob('*.jsonl')):
        with open(f) as fh: first = fh.readline()
        if '"instance_id"' in first and '"patch"' in first:
            return [json.loads(l) for l in open(f) if l.strip()]
    raise FileNotFoundError('no tasks .jsonl with instance_id/patch under ' + str(DATA))
TASKS = load_tasks(); T = {t['instance_id']: t for t in TASKS}
repo_of = lambda t: t['repo'].split('/')[-1]
REPOS = list(dict.fromkeys(repo_of(t) for t in TASKS))
PILOT = {r: next(t for t in TASKS if repo_of(t) == r) for r in REPOS}

def find_wheels():
    c = Counter(p.parent for p in DATA.rglob('*.whl'))
    return c.most_common(1)[0][0] if c else None
WHEELS = find_wheels()

# ============================== snapshots ==============================
def extract_lite(task):
    dst = SNAPS / task['instance_id']
    if (dst / '.prop_done').exists(): return dst
    shutil.rmtree(dst, ignore_errors=True)
    with tarfile.open(DATA / 'snapshots' / f"{task['instance_id']}.tgz") as tf:
        keep = [m for m in tf.getmembers() if (m.isfile() or m.isdir())
                and (m.name[2:] if m.name.startswith('./') else m.name).split('/', 1)[0] != '.git']
        tf.extractall(dst, members=keep, filter='data')
    kids = [p for p in dst.iterdir()]
    if len(kids) == 1 and kids[0].is_dir() and not (dst / 'pyproject.toml').exists() and not (dst / 'setup.py').exists():
        inner = kids[0]                                   # tarball with a single top folder: flatten it
        for p in inner.iterdir(): shutil.move(str(p), dst / p.name)
        inner.rmdir()
    (dst / '.prop_done').write_text('1')
    return dst

def fresh_copy(task, name=None):
    w = WORK / (name or task['instance_id']); shutil.rmtree(w, ignore_errors=True)
    shutil.copytree(extract_lite(task), w, symlinks=True, ignore=shutil.ignore_patterns('.prop_done'))
    return w

def pkgs_of(root):
    root = Path(root)
    return sorted({d.name for b in (root, root / 'src') if b.is_dir() for d in b.iterdir()
                   if d.is_dir() and (d / '__init__.py').exists() and d.name not in NONPKG})

# ============================== processes (memory watchdog) ==============================
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

# ============================== plugins and shims ==============================
STUB = '''import os
def pytest_addoption(parser):
    for name in os.environ.get("PROP_STUB_INI", "").split(","):
        if name: parser.addini(name, "stub: plugin unavailable offline", default="")
def pytest_configure(config):
    for m in os.environ.get("PROP_STUB_MARKERS", "").split(","):
        if m: config.addinivalue_line("markers", m + ": stub marker (plugin unavailable offline)")
'''
IDENT = ('def _ident(x=None, *a, **k):\n    return x\n'
         'snapshot = external = outsource = Is = _ident\n'
         'def __getattr__(name):\n    return _ident\n')
DIRTY = r'''
import re, uuid as _uuid, json as _json, copy as _copy
class DirtyEquals:
    def __init__(self, *a, **k): self._a, self._k = a, k
    def equals(self, o): return True
    def __eq__(self, o):
        try: return bool(self.equals(o))
        except Exception: return False
    def __ne__(self, o): return not self.__eq__(o)
    def __or__(self, o): return _Or(self, o)
    def __ror__(self, o): return _Or(o, self)
    def __and__(self, o): return _And(self, o)
    def __invert__(self): return _Not(self)
    __hash__ = object.__hash__
    def __repr__(self): return f'{type(self).__name__}{self._a!r}'
class _Or(DirtyEquals):
    def equals(self, o): return self._a[0] == o or self._a[1] == o
class _And(DirtyEquals):
    def equals(self, o): return self._a[0] == o and self._a[1] == o
class _Not(DirtyEquals):
    def equals(self, o): return not (self._a[0] == o)
class AnyThing(DirtyEquals): pass
class IsInstance(DirtyEquals):
    def equals(self, o): return isinstance(o, self._a[0])
class IsDict(DirtyEquals):
    _partial = False
    def __init__(self, *a, **k): super().__init__(*a, **k); self._d = dict(*a, **k)
    def settings(self, *, partial=None, strict=None, ignore=None):
        c = _copy.copy(self)
        if partial is not None: c._partial = partial
        return c
    def equals(self, o):
        if not isinstance(o, dict): return False
        if not self._partial and set(o) != set(self._d): return False
        return all(k in o and self._d[k] == o[k] for k in self._d)
class IsPartialDict(IsDict): _partial = True
class IsOneOf(DirtyEquals):
    def equals(self, o): return any(x == o for x in self._a)
class IsStr(DirtyEquals):
    def equals(self, o):
        r = self._k.get('regex'); return isinstance(o, str) and (r is None or re.fullmatch(r, o) is not None)
IsAnyStr = IsStr
class IsInt(DirtyEquals):
    def equals(self, o): return isinstance(o, int) and not isinstance(o, bool)
class IsFloat(DirtyEquals):
    def equals(self, o): return isinstance(o, float)
class IsList(DirtyEquals):
    def equals(self, o):
        if not isinstance(o, list) or ('length' in self._k and len(o) != self._k['length']): return False
        if not self._a: return True
        if self._k.get('check_order', True): return len(o) == len(self._a) and all(x == y for x, y in zip(self._a, o))
        rest = list(o)
        for x in self._a:
            for i, y in enumerate(rest):
                if x == y: rest.pop(i); break
            else: return False
        return not rest
class IsUUID(DirtyEquals):
    def equals(self, o):
        try: _uuid.UUID(str(o)); return True
        except Exception: return False
class HasRepr(DirtyEquals):
    def equals(self, o): return repr(o) == self._a[0]
class IsJson(DirtyEquals):
    def equals(self, o): v = _json.loads(o); return (not self._a) or self._a[0] == v
def __getattr__(name): raise ImportError(f'dirty_equals shim lacks {name}')
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
(PLUG / 'prop_stub.py').write_text(STUB); (PLUG / 'prop_trace2.py').write_text(TRACER)
(SHIM / 'inline_snapshot').mkdir(exist_ok=True)
(SHIM / 'inline_snapshot' / '__init__.py').write_text(IDENT); (SHIM / 'inline_snapshot' / 'extra.py').write_text(IDENT)

# ============================== per-repo environments (pip --target from wheels/) ==============================
def _wheel_index():
    supported = {str(t) for t in sys_tags()}; idx = defaultdict(list)
    for f in (WHEELS.glob('*.whl') if WHEELS else []):
        try:
            name, ver, _, tags = parse_wheel_filename(f.name)
        except Exception:
            continue
        if any(str(t) in supported for t in tags): idx[canonicalize_name(name)].append((ver, f))
    for k in idx: idx[k].sort(key=lambda x: x[0], reverse=True)
    return idx

def _wheel_requires(f):
    with zipfile.ZipFile(f) as z:
        meta = next((n for n in z.namelist() if n.endswith('.dist-info/METADATA')), None)
        if not meta: return []
        return Parser().parsestr(z.read(meta).decode('utf-8', 'replace')).get_all('Requires-Dist') or []

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

def repo_requirements(snap):
    snap = Path(snap); raw = []
    pp = snap / 'pyproject.toml'
    if pp.exists():
        d = tomllib.loads(pp.read_text(errors='replace')); proj = d.get('project', {})
        raw += proj.get('dependencies', [])
        for v in proj.get('optional-dependencies', {}).values(): raw += v
        groups = d.get('dependency-groups', {})
        def expand(g, seen=()):
            res = []
            for x in groups.get(g, []):
                if isinstance(x, str): res.append(x)
                elif isinstance(x, dict) and x.get('include-group') not in seen + (g,): res += expand(x['include-group'], seen + (g,))
            return res
        for g in groups: raw += expand(g)
        poetry = d.get('tool', {}).get('poetry', {})
        names = list(poetry.get('dependencies', {})) + list(poetry.get('dev-dependencies', {}))
        for g in poetry.get('group', {}).values(): names += list(g.get('dependencies', {}))
        raw += [n for n in names if n.lower() != 'python']
    for f in list(snap.glob('requirements*.txt')) + list((snap / 'requirements').glob('*.txt')):
        for line in f.read_text(errors='replace').splitlines():
            line = line.split('#')[0].strip()
            if line and not line.startswith(('-', 'git+', 'http', '.')): raw.append(line)
    if (snap / 'setup.py').exists(): raw += _setup_py_reqs(snap / 'setup.py')
    if (snap / 'setup.cfg').exists():
        txt = (snap / 'setup.cfg').read_text(errors='replace')
        for m in re.finditer(r'^(?:install_requires|tests_require)\s*=\s*\n((?:[ \t]+\S.*\n?)+)', txt, flags=re.M):
            raw += [l.strip() for l in m.group(1).splitlines() if l.strip()]
    reqs = []
    for s in raw + TEST_EXTRA:
        try: reqs.append(Requirement(s))
        except Exception: pass
    return reqs

def resolve(root_reqs, pins, skip):
    idx, env = _wheel_index(), default_environment(); chosen, missing, notes = {}, [], []
    ok = lambda r, ex: r.marker is None or any(r.marker.evaluate({**env, 'extra': e}) for e in (ex or {''}))
    queue = [(r, None) for r in root_reqs]
    while queue:
        r, pex = queue.pop(0)
        if not ok(r, pex): continue
        n = canonicalize_name(r.name)
        if n in skip: continue
        if n in chosen:
            v, f, ex = chosen[n]; new = set(r.extras) - ex
            if new:
                chosen[n] = (v, f, ex | new)
                queue += [(Requirement(d), new) for d in _wheel_requires(f)]
            continue
        spec = SpecifierSet(pins[n]) if n in pins else r.specifier
        cand = [(v, f) for v, f in idx.get(n, []) if spec.contains(v, prereleases=True)]
        if not cand and n in pins:
            notes.append(f'pin {n}{pins[n]} unavailable, using {r.specifier or "any"}')
            cand = [(v, f) for v, f in idx.get(n, []) if r.specifier.contains(v, prereleases=True)]
        if not cand: missing.append(str(r)); continue
        v, f = cand[0]; chosen[n] = (v, f, set(r.extras))
        queue += [(Requirement(d), set(r.extras) or {''}) for d in _wheel_requires(f)]
    return chosen, sorted(set(missing)), notes

def env_dir(repo): return ENVS / repo

def build_env(repo, force=False):
    d = env_dir(repo); done = d / '.prop_env.json'
    if done.exists() and not force: return json.load(open(done))
    shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True)
    snap = extract_lite(PILOT[repo]); skip = {canonicalize_name(x) for x in pkgs_of(snap) + [repo]}
    chosen, missing, notes = resolve(repo_requirements(snap), {canonicalize_name(k): v for k, v in PINS.get(repo, {}).items()}, skip)
    files = [str(f) for _, f, _ in chosen.values()]; rc, out = 0, ''
    if files:
        rc, out, *_ = sp([PY, '-m', 'pip', 'install', '-q', '--no-deps', '--no-index', '--disable-pip-version-check',
                          '--no-compile', '--target', d] + files, '/tmp', 1800, cap_mb=20000)
    info = dict(repo=repo, installed={k: str(v) for k, (v, _, _) in chosen.items()}, missing=missing, notes=notes,
                pip_rc=rc, pip_tail=out[-600:])
    json.dump(info, open(done, 'w'), indent=1)
    return info

# ============================== running pytest ==============================
CFG_FILE = OUT / 'cfg.json'
CFG = json.load(open(CFG_FILE)) if CFG_FILE.exists() else {}

def env_for(repo, root, extra=None):
    root = Path(root); c = CFG.get(repo, {})
    env = {k: v for k, v in os.environ.items() if k not in ('COLUMNS', 'PYTHONPATH', 'MPLBACKEND', 'PYTHONSTARTUP')}
    pp = [str(root)] + ([str(root / 'src')] if (root / 'src').is_dir() else []) + [str(PLUG), str(SHIM), str(env_dir(repo)), str(SHIM2)]
    env.update(PYTHONPATH=os.pathsep.join(pp), PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', PYTHONDONTWRITEBYTECODE='1',
               PROP_STUB_INI=','.join(c.get('ini_stubs') or []), PROP_STUB_MARKERS=','.join(c.get('markers') or []),
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    env.update(extra or {})
    return env

def cmd_for(repo):
    c = CFG.get(repo, {})
    return ([PY, '-m', 'pytest', '-p', 'no:cacheprovider', '-q', '--continue-on-collection-errors', '-p', 'prop_stub']
            + (['-p', 'anyio.pytest_plugin'] if c.get('anyio') else []) + list(c.get('wflags') or []))

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

def suite_run(repo, w, tag, traced=False, timeout=1800, targets=()):
    w = Path(w); xml = w / f'.junit_{tag}.xml'; env = env_for(repo, w)
    cmd = cmd_for(repo) + list(targets) + [f'--junitxml={xml}']
    if traced:
        cmd += ['-p', 'prop_trace2']
        env.update(PROP_TRACE_DIR=str(w), PROP_TRACE_ROOT=str(w), PROP_TRACE_PKG=','.join(pkgs_of(w)))
    rc, out, secs, peak, killed = sp(cmd, w, timeout, env)
    return junit_cases(xml), dict(rc=rc, secs=round(secs, 1), peak_mb=peak, killed=killed, tail=out[-400:])

def detect_cfg(repo, force=False):
    """find ini options / markers of plugins we cannot load, and whether tests need the anyio plugin"""
    global CFG
    if repo in CFG and not force: return CFG[repo]
    w = fresh_copy(PILOT[repo], f'cfg_{repo}')
    tests_txt = ' '.join(p.read_text(errors='replace') for p in w.rglob('test*.py') if 'tests' in p.parts or p.name.startswith('test_'))
    c = dict(anyio=bool(re.search(r'pytest\.mark\.anyio|anyio_backend', tests_txt)), ini_stubs=[], markers=[], wflags=[])
    CFG[repo] = c
    for _ in range(6):
        rc, out, *_ = sp(cmd_for(repo) + ['--co'], w, 900, env_for(repo, w))
        ini = set(re.findall(r'Unknown config option: ([\w\-\.]+)', out)) - set(c['ini_stubs'])
        mk = set(re.findall(r"'([\w\.\-]+)' not found in `markers`", out)) - set(c['markers'])
        if not ini and not mk: break
        c['ini_stubs'] += sorted(ini); c['markers'] += sorted(mk)
    json.dump(CFG, open(CFG_FILE, 'w'), indent=1); shutil.rmtree(w, ignore_errors=True)
    return c

def ensure_dirty_equals():
    """fall back to a minimal dirty_equals shim only if no real one is importable"""
    for repo in REPOS:
        if sp([PY, '-c', 'import dirty_equals'], '/tmp', 120, env_for(repo, '/tmp'))[0] != 0:
            (SHIM2 / 'dirty_equals').mkdir(exist_ok=True); (SHIM2 / 'dirty_equals' / '__init__.py').write_text(DIRTY)
            return 'shim'
    return 'real'

def gate():
    rows = []
    for repo in REPOS:
        w = fresh_copy(PILOT[repo], f'gate_{repo}'); res, m = suite_run(repo, w, 'gate'); shutil.rmtree(w, ignore_errors=True)
        n = Counter(s for s, _ in res.values()); ref = REF_GATE.get(repo, {})
        rows.append(dict(repo=repo, pilot=PILOT[repo]['instance_id'], passed=n['passed'], failed=n['failed'], error=n['error'],
                         skipped=n['skipped'], ref=f"{ref.get('passed')}/{ref.get('failed')}/{ref.get('error')}",
                         secs=m['secs'], peak_mb=m['peak_mb'], killed=m['killed'],
                         ok=(n['passed'] >= 0.9 * ref['passed']) if ref else n['passed'] > 0))
    return pd.DataFrame(rows)

# ============================== patches ==============================
def apply_patch(patch, root):
    errs = []
    for label, p in (('raw', patch), ('nl', patch if patch.endswith('\n') else patch + '\n')):
        rc, out, *_ = sp(['git', 'apply', '--unsafe-paths', '-p1', '-'], root, 60, inp=p)
        if rc == 0: return True, f'{label}:git'
        errs.append(out.strip()[-160:])
        pc = ['patch', '-p1', '--batch', '--forward', '-s', '--fuzz=2', '--no-backup-if-mismatch']
        if sp(pc + ['--dry-run'], root, 60, inp=p)[0] == 0 and sp(pc, root, 60, inp=p)[0] == 0: return True, f'{label}:patch'
    return False, ' || '.join(errs[-2:])

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

def patch_functions(snap, patch):
    snap = Path(snap); pk = set(pkgs_of(snap)); F, notes = set(), []
    for rel, lines in changed_lines(patch).items():
        if not rel.endswith('.py') or TESTFILE.search(rel) or modname(rel).split('.')[0] not in pk:
            notes.append(f'skip {rel}'); continue
        if not (snap / rel).exists():
            notes.append(f'new file {rel}'); continue
        try:
            spans = _spans((snap / rel).read_text(errors='replace'))
        except SyntaxError:
            notes.append(f'syntax {rel}'); continue
        mod = modname(rel)
        for ln in lines:
            inside = [s for s in spans if s[0] <= ln <= s[1]]
            F.add(f'{mod}.{max(inside)[2]}' if inside else f'{mod}.<module>')
    return F, notes

def required_tests(t):
    for k in ('FAIL_TO_PASS', 'fail_to_pass'):
        v = t.get(k)
        if v:
            v = json.loads(v) if isinstance(v, str) else v
            return {x.split('::')[-1].split('[')[0] for x in v}, 'field'
    tp = t['test_patch']
    names = set(re.findall(r'^\+\s*(?:async\s+)?def\s+(test\w*)', tp, flags=re.M))
    names |= set(re.findall(r'^@@[^@]*@@.*?def\s+(test\w*)', tp, flags=re.M))
    return names, 'extracted'

def func_status(cases, req):
    st = defaultdict(list)
    for k, (s, _) in cases.items(): st[k.split('::', 1)[-1].split('[')[0]].append(s)
    return {n: ('missing' if not st.get(n) else 'pass' if all(s == 'passed' for s in st[n])
                else 'skip' if all(s == 'skipped' for s in st[n]) else 'fail') for n in req}

def test_targets(t, w):
    return sorted({f for f in re.findall(r'^\+\+\+ b/(\S+)', t['test_patch'], flags=re.M)
                   if TESTFILE.search(f) and not f.endswith('conftest.py') and (Path(w) / f).exists()})

# ============================== stage: gold-patch reproduction ==============================
def repro_task(t):
    tid, repo = t['instance_id'], repo_of(t); f = RD / f'{tid}.json'
    if f.exists(): return json.load(open(f))
    Fs, _ = patch_functions(extract_lite(t), t['patch']); Fp = [x for x in Fs if not x.endswith('.<module>')]
    res = dict(task=tid, repo=repo, cat='package' if Fp else 'module-level-only' if Fs else 'non-package', nF=len(Fp))
    w = fresh_copy(t)
    try:
        ok, how_t = apply_patch(t['test_patch'], w)
        if not ok: res.update(status='test_patch-failed', msg=how_t); return res
        targets = test_targets(t, w)
        if not targets: res.update(status='no-test-files'); return res
        req, kind = required_tests(t); runs = {}; how_p = ''
        for phase in ('base', 'gold'):
            if phase == 'gold':
                ok, how_p = apply_patch(t['patch'], w)
                if not ok: res.update(status='patch-failed', msg=how_p); return res
            runs[phase] = suite_run(repo, w, phase, timeout=600, targets=targets)
        cb, cg = runs['base'][0], runs['gold'][0]
        emp = sorted(k for k, (s, _) in cg.items() if s == 'passed' and cb.get(k, ('missing',))[0] in ('failed', 'error', 'missing'))
        sb, sg = func_status(cb, req), func_status(cg, req)
        f2p = sorted(n for n in req if sb[n] in ('fail', 'missing') and sg[n] == 'pass')
        bad = sorted(n for n in req if sg[n] != 'pass')
        status = (('reproduced-empirical' if emp else 'no-f2p') if not req else
                  'gold-fails' if bad else 'reproduced' if f2p else 'no-f2p')
        res.update(status=status, req_src=kind, n_req=len(req), f2p=f2p if req else emp, n_f2p=len(f2p) if req else len(emp),
                   apply=f'{how_t}/{how_p}', targets=len(targets), bad=[f'{n}:{sg[n]}' for n in bad][:4],
                   bad_msg=[m for k, (s, m) in cg.items() if s in ('failed', 'error')][:2] if bad else [],
                   secs=runs['base'][1]['secs'] + runs['gold'][1]['secs'])
        return res
    except Exception as e:
        res.update(status='crash', msg=repr(e)[:300]); return res
    finally:
        if res.get('status') != 'crash': json.dump(res, open(f, 'w'))
        shutil.rmtree(w, ignore_errors=True)

def run_repro(workers=4, log_every=15):
    rows, t0 = [], time.time()
    with cf.ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(repro_task, t): t for t in TASKS}
        for i, fu in enumerate(cf.as_completed(futs)):
            try:
                rows.append(fu.result())
            except Exception as e:
                t = futs[fu]; rows.append(dict(task=t['instance_id'], repo=repo_of(t), cat='?', status='crash', msg=repr(e)[:300]))
            if (i + 1) % log_every == 0: print(f'   repro {i + 1}/{len(TASKS)}  {time.time() - t0:.0f}s', flush=True)
    rep = pd.DataFrame(rows).sort_values('task')
    rep['ok'] = rep.status.isin(['reproduced', 'reproduced-empirical'])
    rep.to_csv(OUT / 'repro.csv', index=False)
    return rep

# ============================== stage: traced coverage + real-bug response ==============================
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

def nid2key(nid):
    p = nid.split('::'); f = p[0][:-3].replace('/', '.') if p[0].endswith('.py') else p[0]
    return '.'.join([f] + p[1:-1]) + '::' + p[-1]

isbad = lambda s: s in ('failed', 'error', 'missing')

def task_run(t, f2p_list):
    repo, tid = repo_of(t), t['instance_id']; d = TD / tid; d.mkdir(exist_ok=True)
    if (d / 'summary.json').exists(): return json.load(open(d / 'summary.json'))
    w = fresh_copy(t); row = dict(task=tid, repo=repo)
    try:
        base, m1 = suite_run(repo, w, 'base')
        if not apply_patch(t['patch'], w)[0]: return {**row, 'error': 'patch'}
        gold, m2 = suite_run(repo, w, 'gold')
        if not apply_patch(t['test_patch'], w)[0]: return {**row, 'error': 'test_patch'}
        tr, m3 = suite_run(repo, w, 'goldtp', traced=True)
        row.update(secs=[m1['secs'], m2['secs'], m3['secs']], peak_mb=[m1['peak_mb'], m2['peak_mb'], m3['peak_mb']])
        if any(m['killed'] for m in (m1, m2, m3)):
            return {**row, 'error': 'killed ' + str([m['killed'] for m in (m1, m2, m3)])}
        if not (w / '.trace_meta.json').exists(): return {**row, 'error': 'no trace: ' + m3['tail']}
        keys, funcs, C, meta = load_trace(w); fi = {f: i for i, f in enumerate(funcs)}
        d_old = sorted(k for k, (s, _) in gold.items() if s == 'passed' and isbad(base.get(k, ('missing',))[0]))
        d_rev = sorted(k for k, (s, _) in base.items() if s == 'passed' and isbad(gold.get(k, ('missing',))[0]))
        tpf = [f for f in re.findall(r'^\+\+\+ b/(\S+)', t['test_patch'], flags=re.M) if TESTFILE.search(f)]
        tmod = tuple(f[:-3].replace('/', '.') for f in tpf)
        in_t = lambda k: any(k.startswith(m + '.') or k.startswith(m + '::') or k == '::' + m for m in tmod)
        mod_names = required_tests(t)[0]                       # test functions the test patch adds or edits
        is_mod = lambda k: k.split('::')[-1].split('[')[0] in mod_names
        old_keys = {k for k in gold if not is_mod(k)}          # pre-existing tests (run without the test patch)
        mism = sorted(k for k in old_keys if tr.get(k, ('missing',))[0] != gold[k][0])
        ssp.save_npz(d / 'C.npz', C)
        json.dump({'tests': keys, 'funcs': funcs, 'collect': meta['collect'], 'edges': meta['edges']}, open(d / 'index.json', 'w'))
        Fs = sorted(patch_functions(extract_lite(t), t['patch'])[0]); Fp = [f for f in Fs if not f.endswith('.<module>')]
        Fcols = [fi[f] for f in Fp if f in fi]; lst = f2p_list or []
        f2p_keys = ({k for k in lst if '::' in k} if any('::' in x for x in lst) else
                    {k for k in keys if in_t(k) and k.split('::')[-1].split('[')[0] in set(lst)})
        rows_f2p = [i for i, k in enumerate(keys) if k in f2p_keys]
        old_rows = np.array([k in old_keys for k in keys], bool)
        hitsF = (np.asarray(C[:, Fcols].sum(1)).ravel() > 0) if Fcols else np.zeros(len(keys), bool)
        foot = int((np.asarray(C[rows_f2p].sum(0)).ravel() > 0).sum()) if rows_f2p else 0
        row.update(n_tests=len(keys), n_funcs=len(funcs), n_edges=len(meta['edges']), pkgs=','.join(pkgs_of(w)),
                   d_old=len(d_old), d_old_list=d_old[:8], d_rev=len(d_rev), d_rev_list=d_rev[:5],
                   fid_mismatch=len(mism), fid_list=mism[:5], nF=len(Fp), F_exec=len(Fcols),
                   F_missing=[f for f in Fp if f not in fi][:4], old_tests_hit_F=int((hitsF & old_rows).sum()),
                   n_f2p=len(rows_f2p), f2p_hit_F=bool(rows_f2p) and bool(hitsF[rows_f2p].any()), f2p_footprint=foot)
        json.dump(row, open(d / 'summary.json', 'w'))
        return row
    except Exception as e:
        return {**row, 'error': repr(e)[:300]}
    finally:
        shutil.rmtree(w, ignore_errors=True); gc.collect()

def run_step8(rep, log_every=10):
    use = rep[rep.ok & (rep['cat'] == 'package')]
    f2p = {x: json.load(open(RD / f'{x}.json')).get('f2p', []) for x in use.task}
    done = {p.parent.name for p in TD.glob('*/summary.json')}
    todo = [T[x] for x in use.task]
    probe = next((t for t in todo if t['instance_id'] not in done), None); rows = []
    workers = 2
    if probe is not None:
        r = task_run(probe, f2p[probe['instance_id']]); rows.append(r)
        peak = max(r.get('peak_mb') or [9999]); workers = 3 if peak < 2500 else 2
        print(f"   probe {probe['instance_id']}: secs {r.get('secs')} peak MB {r.get('peak_mb')} error {r.get('error')} -> workers {workers}", flush=True)
    order = {'requests': 0, 'httpx': 1, 'fastapi': 2, 'rich': 3}
    rest = sorted([t for t in todo if probe is None or t['instance_id'] != probe['instance_id']], key=lambda t: order.get(repo_of(t), 9))
    t0 = time.time()
    with cf.ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(task_run, t, f2p[t['instance_id']]): t for t in rest}
        for i, fu in enumerate(cf.as_completed(futs)):
            try:
                rows.append(fu.result())
            except Exception as e:
                t = futs[fu]; rows.append(dict(task=t['instance_id'], repo=repo_of(t), error=repr(e)[:300]))
            if (i + 1) % log_every == 0: print(f'   step8 {i + 1}/{len(rest)}  {time.time() - t0:.0f}s', flush=True)
    S = pd.DataFrame(rows).sort_values('task'); S.to_csv(OUT / 'step8_tasks.csv', index=False)
    return S

def report8(S):
    err = S[S['error'].notna()] if 'error' in S else S.iloc[:0]
    print(f'step 8: {len(S)} tasks, errors: {len(err)}')
    for _, r in err.iterrows(): print('   ERR', r.task, str(r.error)[:200])
    S = S[S['error'].isna()] if 'error' in S else S
    if S.empty: return
    print('\nper repo (median):')
    print(S.groupby('repo')[['n_tests', 'n_funcs', 'n_edges', 'nF', 'F_exec', 'n_f2p', 'f2p_footprint']].median().to_string())
    print('mean secs base/gold/traced:', {r: np.round(np.mean(g.secs.tolist(), axis=0), 1).tolist() for r, g in S.groupby('repo')})
    print('max peak MB base/gold/traced:', {r: np.max(g.peak_mb.tolist(), axis=0).tolist() for r, g in S.groupby('repo')})
    print(f"\nFIDELITY  tasks with 0 mismatches: {(S.fid_mismatch == 0).sum()}/{len(S)} | total {S.fid_mismatch.sum()}")
    for _, r in S[S.fid_mismatch > 0].nlargest(5, 'fid_mismatch').iterrows(): print(f'   {r.task}: {r.fid_mismatch} {r.fid_list[:3]}')
    print(f"\nNAMES  F* found among executed functions: {S.F_exec.sum()}/{S.nF.sum()} | "
          f"tasks whose F2P tests execute F*: {S.f2p_hit_F.sum()}/{(S.n_f2p > 0).sum()}")
    print('   F* names not seen (sample):', [x for l in S.F_missing for x in l][:8])
    print('\nREAL-BUG RESPONSE on the pre-existing suite (d_old = old tests failing on base, passing with fix):')
    print(f"   tasks with d_old = 0: {(S.d_old == 0).sum()}/{len(S)} | size: {S.d_old.describe()[['mean', '50%', 'max']].to_dict()}")
    print(f"   reverse flips (fix breaks an old test): {(S.d_rev > 0).sum()} tasks")
    cov = S[S.old_tests_hit_F > 0]
    print(f"   old suite EXECUTES F*: {len(cov)}/{len(S)} tasks (median {cov.old_tests_hit_F.median():.0f} old tests touch F*)")
    print(f"   ... and is SILENT (d_old = 0): {(cov.d_old == 0).sum()}/{len(cov)}")
    for _, r in S[S.d_old > 0].iterrows(): print(f'   d_old>0 {r.task}: {r.d_old} {r.d_old_list[:3]}')
