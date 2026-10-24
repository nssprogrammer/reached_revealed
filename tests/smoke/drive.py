import os, sys, threading, json
W = sys.argv[1]; os.environ.update(PROP_WORKDIR=W, PROP_DATA=sys.argv[2], PROP_TMP=W + '/tmp', AB_WORK=W + '/abwork')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, sys.argv[3])
import fake_server
from http.server import ThreadingHTTPServer
threading.Thread(target=ThreadingHTTPServer(('127.0.0.1', 8000), fake_server.H).serve_forever, daemon=True).start()
import ab_agent as A
for repo in A.P.REPOS: A.P.build_env(repo); print(A.P.detect_cfg(repo))
A.P.ensure_dirty_equals()
tasks = []
for tid in sorted(A.P.T):
    t = dict(A.P.T[tid]); t['problem_statement'] = f'TASK={tid} clamp/total misbehave for large values'; A.P.T[tid] = t; tasks.append(t)
A.calibrate_and_probe_mode()
R = A.run_ab(tasks, workers=4, log_every=2, seed=int(os.environ.get('SEED', 0)))
print(R[['task', 'arm', 'stop', 'calls', 'turns', 'submitted', 'probe', 'errors']].to_string())
for f in sorted((A.AB_OUT).glob('*/*.json')):
    r = json.load(open(f)); print('\n==', f.parent.name, r['task'], r['stop'], 'vtime %.1f' % r['vtime'], r['tools']); print(r['diff'][:400])
    for x in r['transcript']:
        if x['role'] == 'tool' and ('probe' in x['content'] or '===' in x['content']): print('   TOOL:', x['content'][:300])
