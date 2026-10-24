import sys, importlib, json, os
import pandas as pd
sys.path.insert(0, os.environ.get('PROP_WORKDIR', '/kaggle/working'))
import proplib as P; importlib.reload(P)
print('data:', P.DATA, '| tasks:', len(P.TASKS), '| repos:', P.REPOS, '| wheels dir:', P.WHEELS)
print('pilot task per repo:', {r: t['instance_id'] for r, t in P.PILOT.items()})

# 1) per-repo environments from the competition wheels (cached after the first run)
for repo in P.REPOS:
    info = P.build_env(repo)
    print(f"env {repo:9s}: {len(info['installed'])} wheels installed, pip rc {info['pip_rc']} {info['notes']}"
          f"\n               not in wheels (system site used): {info['missing'][:10]}")
print('dirty_equals:', P.ensure_dirty_equals())

# 2) stubs for pytest plugins that cannot load offline
for repo in P.REPOS: print('cfg', repo, P.detect_cfg(repo))

# 3) gate: pilot suites must match the numbers recorded earlier
G = P.gate(); print('\nGATE\n' + G.to_string(index=False))
if not G.ok.all():
    raise RuntimeError('Gate failed: paste this output before going further')

# 4) gold-patch reproduction for every task (resumable)
rep = P.run_repro(workers=4)
print('\n' + pd.crosstab(rep.repo, rep.status, margins=True).to_string())
use = rep[rep.ok & (rep['cat'] == 'package')]
print(f'USABLE (reproduced AND package): {len(use)}  {use.groupby("repo").size().to_dict()}')
for _, r in rep[~rep.ok].iterrows():
    print(f"   {r.task:22s} {r.status:18s} {str(r.get('bad'))[:80]} | {str(r.get('bad_msg') or r.get('msg'))[:160]!r}")

# 5) step 8: real-bug response on the old suite + traced coverage (resumable)
S = P.run_step8(rep)
P.report8(S)
