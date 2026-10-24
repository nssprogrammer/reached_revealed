# ---- Generalization: the measured propagator on 8 open-source Python libraries (internet ON; CPU; ~1 h) ----
import os, sys, importlib, time
sys.path.insert(0, os.environ.get('PROP_WORKDIR', '/kaggle/working'))
import genlib as GL; importlib.reload(GL)
names = os.environ.get('GEN_LIBS', '').split(',') if os.environ.get('GEN_LIBS') else [l['name'] for l in GL.LIBS]
t0 = time.time()
for lib in [l for l in GL.LIBS if l['name'] in names]:
    try:
        GL.campaign(lib)
    except Exception as e:
        print('   FAILED', lib['name'], repr(e)[:300])
    print(f'   ... {time.time() - t0:.0f}s', flush=True)
GL.report(names)
