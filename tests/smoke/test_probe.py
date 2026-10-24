"""The agent's tracer (propagator/probe.py) on the mini fixture: snippet localization and --focus regression check.
Needs pytest. Run: python3 tests/smoke/test_probe.py"""
import subprocess, sys, tarfile, tempfile, pathlib, re, os
ROOT = pathlib.Path(__file__).resolve().parents[2]; PROBE = ROOT / 'propagator' / 'probe.py'
assert not re.search(r'[{}]', PROBE.read_text()), 'the probe is embedded in an ADK instruction: no curly braces allowed'
with tempfile.TemporaryDirectory() as d:
    with tarfile.open(ROOT / 'tests/fixtures/mini_competition/snapshots/mini_1.tgz') as tf: tf.extractall(d, filter='data')
    env = dict(os.environ, PYTHONPATH=d)                     # the package under test must be importable (editable install in the sandbox)
    snip = subprocess.run([sys.executable, str(PROBE)], cwd=d, env=env, input='from mini import clamp\nassert clamp(50, 0, 10) == 10\n',
                          capture_output=True, text=True).stdout
    assert 'mini/core.py' in snip and 'clamp' in snip, snip
    foc = subprocess.run([sys.executable, str(PROBE), '--focus', 'clamp', 'tests/test_core.py'], cwd=d, env=env, capture_output=True, text=True,
                         stdin=subprocess.DEVNULL).stdout
    assert re.search(r'=== \d+ of \d+ tests execute clamp', foc), foc
print('probe smoke test passed'); print(snip.strip().splitlines()[-3:]); print(foc.strip().splitlines()[-2:])
