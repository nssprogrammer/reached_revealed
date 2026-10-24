# Report + archive for SWE-rebench v3 (run after run_swe_rebench.py in the same session).
import subprocess
import swelib as S
import glob, json, pandas as pd
R = pd.DataFrame([json.load(open(f)) for f in glob.glob('/kaggle/working/swe/out/*/summary.json')])
R = R[R.get('v', pd.Series(0, index=R.index)) == S.VERSION]
S.report(R)
NR = R[R.status == 'not-reproduced']
if len(NR): print('\nnot reproduced: F2P failing on base / passing with fix / total:', NR[['fail_base', 'pass_gold', 'f2p']].astype(int).values.tolist())
if 'msg' in R:
    print('\nfailure reasons:'); print(R[R.msg.notna()].groupby('status').msg.apply(lambda m: m.astype(str).str[-90:].str.replace('\\n', ' ').value_counts().head(4)).to_string()[:3000])
subprocess.run('cd /kaggle/working && tar czf swe_out.tgz swe/out && ls -la swe_out.tgz', shell=True)
