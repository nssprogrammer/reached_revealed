# ---- Dev-set A/B run (GPU L4 x4, internet OFF). Inputs: competition data (snapshots + wheels), the model
# google/gemma-4 -> gemma-4-31b-it-qat-w4a16-ct, the wheelhouse dataset (vLLM), and optionally the end-to-end output
# (propagator/repro/*.json) to skip the 18 tasks whose reference fix does not reproduce. Resumable: finished episodes are kept.
import os, sys, glob, json, time, subprocess, importlib
import pandas as pd
from pathlib import Path
sys.path.insert(0, '/kaggle/working')
def offline_vllm():
    """L4 sessions have no internet: install vLLM and its pinned runtime from the competition's wheel folder (no resolver, --no-deps)"""
    import re as _re
    whl = sorted(glob.glob('/kaggle/input/**/vllm-*.whl', recursive=True))
    if not whl:
        tops = sorted({'/'.join(p.split('/')[:5]) for p in glob.glob('/kaggle/input/*/*/*')})
        print('inputs attached:', *tops, sep='\n   ')
        raise RuntimeError('No vllm-*.whl under /kaggle/input. Add Input -> Datasets -> "gemma-4-developer-agent-wheelhouse" '
                           '(owner: metric). It holds vllm-0.19.1; the competition data alone does not.')
    folder = str(Path(whl[-1]).parent); files = sorted(glob.glob(folder + '/*.whl'))
    try:
        from packaging.version import Version as V
    except Exception:
        V = lambda x: x
    best = {}
    for f in files:                                   # one wheel per project (highest version)
        name, ver = Path(f).name.split('-')[:2]; key = _re.sub(r'[-_.]+', '-', name).lower()
        try: v = V(ver)
        except Exception: v = ver
        if key not in best or v > best[key][0]: best[key] = (v, f)
    picks = [f for _, f in best.values()]
    print(f'installing {len(picks)} wheels from {folder} with --no-deps (vLLM {Path(whl[-1]).name.split("-")[1]})')
    r = subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', '-q'] + picks, capture_output=True, text=True)
    print(r.stdout[-1500:], r.stderr[-3000:])
    t = subprocess.run([sys.executable, '-c', 'import vllm, torch, google.protobuf as p; print("vllm", vllm.__version__, "torch", torch.__version__, "protobuf", p.__version__)'],
                       capture_output=True, text=True)
    print(t.stdout, t.stderr[-2500:])
try:
    import vllm
except ImportError:
    offline_vllm(); importlib.invalidate_caches(); import vllm
print('vLLM', vllm.__version__)
def find_model():
    if os.environ.get('GEMMA_PATH'): return os.environ['GEMMA_PATH']
    cands = []
    for cfg in glob.glob('/kaggle/input/**/config.json', recursive=True):
        try: c = json.load(open(cfg))
        except Exception: continue
        txt = json.dumps(c).lower(); d = str(Path(cfg).parent).lower()
        if 'gemma' in txt or 'gemma' in d: cands.append((('31b' in d) * 4 + ('w4a16' in d or 'compressed' in txt) * 2 + ('-it' in d), str(Path(cfg).parent)))
    if not cands: raise FileNotFoundError('add the model as an input: Kaggle Models -> google/gemma-4 -> gemma-4-31b-it-qat-w4a16-ct')
    return sorted(cands)[-1][1]
import ab_agent as A
for repo in A.P.REPOS:
    info = A.P.build_env(repo); print(f"env {repo:9s} {len(info['installed'])} wheels | cfg {A.P.detect_cfg(repo)}")
print('dirty_equals:', A.P.ensure_dirty_equals())
for tgz in glob.glob('/kaggle/input/**/ab_out.tgz', recursive=True)[:1]:          # resume from an attached earlier A/B run
    subprocess.run(f'cd /kaggle/working && tar xzf {tgz}', shell=True); print('restored', tgz)
ok = {json.load(open(f))['task'] for f in glob.glob('/kaggle/input/**/propagator/repro/*.json', recursive=True)
      if json.load(open(f)).get('status') == 'reproduced'}
TASKS = [A.P.T[i] for i in sorted(A.P.T) if not ok or i in ok]
print(f'{len(TASKS)} tasks x 2 arms' + ('' if ok else ' (no repro info attached: running all tasks)'))

MODEL = find_model(); print('model:', MODEL)
try:
    SRV = A.start_server(MODEL, tp=4, kv_fp8=True)
except RuntimeError as e:
    print('fp8 KV cache failed, retrying with the default KV cache:', str(e)[-400:])
    SRV = A.start_server(MODEL, tp=4, kv_fp8=False)
A.calibrate_and_probe_mode()
WORKERS  = int(os.environ.get('AB_WORKERS', 16))     # concurrent episodes sharing the server
BUDGET_S = 9 * 3600                                  # stop starting new episodes after 9 h; rerun with ab_out.tgz attached to continue
THINK    = False                                     # thinking off in both arms (cost); set True to mirror thinking_budget
SEEDS    = [1]                                       # seed 0 = the first run (ab_out/v2, ab_out/v32); [1, 2] adds two more runs (~4.6 h)
print('new seeds', SEEDS, '| earlier runs restored:', sorted(p.name for p in A.AB_OUT.iterdir()) if A.AB_OUT.exists() else [])
t0 = time.time()
R = pd.concat([A.run_ab(TASKS, arms=('v2', 'v32'), workers=WORKERS, budget_s=BUDGET_S, think=THINK, seed=s) for s in SEEDS])
print(f'done in {(time.time() - t0) / 3600:.1f} h'); print(R.groupby('arm').stop.value_counts().unstack(fill_value=0).to_string())
print(R.groupby('arm')[['calls', 'turns', 'vtime', 'completion_tokens']].mean().round(1).to_string())
subprocess.run('cd /kaggle/working && tar czf ab_out.tgz ab_out vllm_server.log && ls -la ab_out.tgz', shell=True)
