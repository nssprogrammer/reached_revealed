# ---- Step 9: empirical Green's function G from planted bugs (resumable; ~3.5 h) ----
import os, sys, importlib, json
import pandas as pd
W = os.environ.get('PROP_WORKDIR', '/kaggle/working')
missing = [p for p in ('proplib.py', 'proplib9.py', 'propagator/step8_tasks.csv') if not os.path.exists(os.path.join(W, p))]
if missing: raise RuntimeError(f'Missing in {W}: {missing}. Run Cells A, B and C of this notebook first.')
sys.path.insert(0, W)
import proplib as P; importlib.reload(P)
import proplib9 as G9; importlib.reload(G9)

S8 = pd.read_csv(P.OUT / 'step8_tasks.csv')
# 1) retry Step 8 once, with a 60-minute limit, for any task that errored there (e.g. a traced-suite timeout)
if 'error' in S8:
    for tid in S8[S8['error'].notna()].task:
        r = G9.retry_step8(tid, timeout=3600); print('retry step 8', tid, '->', r.get('error') or 'ok', r.get('secs'))

# 2) mutant campaign on every task with Step 8 data (identity control per task, fixed pytest invocation)
ids = sorted(p.parent.name for p in P.TD.glob('*/summary.json'))
print(f'{len(ids)} tasks')
R = G9.run_g(ids, fid_ids=[], workers=3)
R.to_csv(P.OUT / 'step9_tasks.csv', index=False)
M = G9.collect_g(); M.to_csv(P.OUT / 'step9_mutants.csv', index=False)
G9.report_g(M, R)
