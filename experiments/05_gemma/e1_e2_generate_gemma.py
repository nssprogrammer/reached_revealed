# ---- Step G-b: run Gemma 4 on the prompt pack (GPU: L4 x4; Internet ON only if vLLM must be installed).
# Inputs: the Step G-a output (gemma_pack/) + the competition's checkpoint gemma-4-31b-it-qat-w4a16-ct (Kaggle Models, "Add Input").
# Output: /kaggle/working/gemma_out/{loc_out.jsonl, repair_out.jsonl}. Resumable: finished ids are skipped on rerun.
import os, sys, glob, json, re, time, subprocess
from pathlib import Path

def find_pack():
    for p in sorted(glob.glob('/kaggle/input/**/gemma_pack/loc.jsonl', recursive=True)) + sorted(glob.glob('/kaggle/working/**/gemma_pack/loc.jsonl', recursive=True)):
        return Path(p).parent
    raise FileNotFoundError('attach the Step G-a output (gemma_pack/loc.jsonl)')
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
PACK, MODEL = find_pack(), find_model()
OUTD = Path('/kaggle/working/gemma_out'); OUTD.mkdir(parents=True, exist_ok=True)
print('pack:', PACK, '\nmodel:', MODEL)

def offline_vllm():
    """L4 sessions have no internet: install vLLM and its pinned runtime from the competition's wheel folder (no resolver, --no-deps)"""
    import re as _re
    whl = sorted(glob.glob('/kaggle/input/**/vllm-*.whl', recursive=True))
    if not whl:
        raise SystemExit('No vllm-*.whl under /kaggle/input: add the competition data (it has wheels/) as an input.')
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
    offline_vllm(); import importlib; importlib.invalidate_caches(); import vllm
import torch
from vllm import LLM, SamplingParams
ngpu = torch.cuda.device_count(); cap = torch.cuda.get_device_capability(0) if ngpu else (0, 0)
print(f'vLLM {vllm.__version__} | GPUs: {ngpu} x {torch.cuda.get_device_name(0) if ngpu else "-"} (compute {cap[0]}.{cap[1]})')
if cap < (8, 0): print('WARNING: 4-bit w4a16 kernels need compute capability >= 8.0 (L4/A100); T4/P100 will likely fail. Choose "GPU L4 x4".')
TP = max(k for k in (1, 2, 4, 8) if k <= max(ngpu, 1))
MAX_LEN = int(os.environ.get('GEMMA_MAX_LEN', 32768))
GPU_UTIL = float(os.environ.get('GEMMA_GPU_UTIL', 0.80)); MAX_SEQS = int(os.environ.get('GEMMA_MAX_SEQS', 64))   # 0.90/256 ran out of memory in sampler warm-up (262k vocab)
llm = LLM(model=MODEL, tensor_parallel_size=TP, max_model_len=MAX_LEN, gpu_memory_utilization=GPU_UTIL, max_num_seqs=MAX_SEQS, seed=0)
tok = llm.get_tokenizer()

def load(p): return [json.loads(l) for l in open(p) if l.strip()]
def done_ids(p): return {json.loads(l)['id'] for l in open(p)} if p.exists() else set()
def fits(prompt, reserve):
    return len(tok.encode(prompt)) + reserve < MAX_LEN

def run(rows, outp, params_list, chunk=48):
    todo = [r for r in rows if r['id'] not in done_ids(outp)]; t0 = time.time(); ntok = 0
    skipped = [r['id'] for r in todo if not fits(r['prompt'], max(p.max_tokens for p in params_list) + 64)]
    todo = [r for r in todo if r['id'] not in set(skipped)]
    print(f'{outp.name}: {len(todo)} to run, {len(skipped)} too long for {MAX_LEN} tokens')
    for i in range(0, len(todo), chunk):
        batch = todo[i:i + chunk]; msgs = [[{'role': 'user', 'content': r['prompt']}] for r in batch]
        outs = [llm.chat(msgs, p, use_tqdm=False) for p in params_list]
        with open(outp, 'a') as fh:
            for j, r in enumerate(batch):
                texts = [c.text for o in outs for c in o[j].outputs]
                ntok += sum(len(c.token_ids) for o in outs for c in o[j].outputs)
                fh.write(json.dumps(dict(id=r['id'], task=r['task'], cond=r.get('cond'), texts=texts)) + '\n')
        el = time.time() - t0
        print(f'   {min(i + chunk, len(todo))}/{len(todo)}  {el / 60:.1f} min  {ntok / max(el, 1):.0f} output tok/s', flush=True)

# E1 localization: one greedy answer per prompt
run(load(PACK / 'loc.jsonl'), OUTD / 'loc_out.jsonl', [SamplingParams(temperature=0.0, max_tokens=1024)])
# E2 repair: one greedy patch + 4 samples (temperature 0.8) per task
run(load(PACK / 'repair.jsonl'), OUTD / 'repair_out.jsonl',
    [SamplingParams(temperature=0.0, max_tokens=8192), SamplingParams(n=4, temperature=0.8, top_p=0.95, max_tokens=8192, seed=0)], chunk=24)

# ------------------------------------------------------------ immediate E1 readout (no execution needed)
import pandas as pd
def parse_names(text, cands):
    m = None
    for m in re.finditer(r'\{[^{}]*"functions"[^{}]*\}', text, flags=re.S): pass
    names = []
    if m:
        try: names = json.loads(m.group(0)).get('functions', [])
        except Exception: names = re.findall(r'"([^"]+)"', m.group(0))[1:]
    out = []
    for nm in names:
        nm = str(nm).split('(')[0].strip().strip('`').strip()
        hit = nm if nm in cands else next((c for c in cands if c.endswith('.' + nm) or nm.endswith('.' + c) or nm.endswith(c)), None)
        if hit and hit not in out: out.append(hit)
    return out, m is not None
P = {r['id']: r for r in load(PACK / 'loc.jsonl')}; rows = []
for o in load(OUTD / 'loc_out.jsonl'):
    r = P[o['id']]; names, ok = parse_names(o['texts'][0], r['candidates']); F = set(r['fstar'])
    rows.append(dict(task=r['task'], cond=r['cond'], n_cand=len(r['candidates']), in_cands=bool(F & set(r['candidates'])), parsed=ok,
                     top1=bool(names[:1] and names[0] in F), top3=bool(F & set(names[:3])), chance1=len(F & set(r['candidates'])) / len(r['candidates'])))
L = pd.DataFrame(rows); L.to_csv(OUTD / 'loc_scores.csv', index=False)
print('\nE1 GEMMA LOCALIZATION (fix function named by Gemma; candidates chosen by each method)')
print(L.groupby('cond').agg(tasks=('task', 'count'), cands=('n_cand', 'median'), recall_K=('in_cands', 'mean'), parsed=('parsed', 'mean'),
                            gemma_top1=('top1', 'mean'), gemma_top3=('top3', 'mean'), chance_top1=('chance1', 'mean')).round(3).to_string())
both = L.pivot_table(index='task', columns='cond', values='top1', aggfunc='first').dropna()
from scipy.stats import binomtest
if {'rt1', 'pv1'} <= set(both.columns):
    a = int((both.rt1 & ~both.pv1).sum()); b = int((~both.rt1 & both.pv1).sum())
    print(f'paired rt1 vs pv1 top-1 on {len(both)} tasks: rt1 only {a}, pv1 only {b}, sign-test p = {binomtest(a, a + b).pvalue if a + b else 1:.3g}')
if {'rt1', 'bm25'} <= set(both.columns):
    a = int((both.rt1 & ~both.bm25).sum()); b = int((~both.rt1 & both.bm25).sum())
    print(f'paired rt1 vs bm25 top-1: rt1 only {a}, bm25 only {b}, sign-test p = {binomtest(a, a + b).pvalue if a + b else 1:.3g}')
subprocess.run('cd /kaggle/working && tar czf gemma_out.tgz gemma_out && ls -la gemma_out.tgz', shell=True)
