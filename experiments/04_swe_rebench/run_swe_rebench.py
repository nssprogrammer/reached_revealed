# Runner for SWE-rebench v3 (Kaggle CPU, internet ON). Needs propagator/swelib.py importable.
import os, sys, glob, time, subprocess
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'uv', 'datasets'], check=True)
for tgz in glob.glob('/kaggle/input/**/swe_out.tgz', recursive=True)[:1]:      # resume from an attached earlier run
    subprocess.run(f'cd /kaggle/working && tar xzf {tgz}', shell=True); print('restored', tgz)
sys.path.insert(0, '/kaggle/working')
import swelib as S
N_TASKS   = 150         # candidates (at most 2 per repository), Python 3.9-3.13
WORKERS   = 3
BUDGET_S  = 10 * 3600   # stop starting new tasks after 10 h (Kaggle sessions end at 12 h); rerun to continue
df = S.load_swe_rebench('test')
print('Python versions in SWE-rebench:', df.install_config.map(lambda c: S._j(c, {}).get('python')).astype(str).value_counts().to_dict())
ids = S.select_tasks(df, n=N_TASKS, per_repo=2, seed=0)
print(f'{len(df)} SWE-rebench tasks -> {len(ids)} selected from {df[df.instance_id.isin(ids)].repo.nunique()} repositories')
t0 = time.time(); R = S.run_all(ids, df, workers=WORKERS, budget_s=BUDGET_S)
print(f'done in {(time.time() - t0) / 60:.0f} min')
