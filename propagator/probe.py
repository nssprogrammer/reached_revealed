import sys, os, re, itertools, traceback, collections
args = sys.argv[1:]; FOCUS = args[args.index('--focus') + 1] if '--focus' in args else ''
TESTS = [x for i, x in enumerate(args) if not x.startswith('--') and (i == 0 or args[i - 1] != '--focus')]
ROOT = os.getcwd() + os.sep
SKIP = re.compile(r'(^|/)(tests?|testing)/|(^|/)test_[^/]*\.py$|_test\.py$|conftest\.py$|-packages/')
keys, per, last, cur, tick = dict(), dict(), dict(), ['run'], itertools.count()
def prof(fr, ev, arg):
    if ev != 'call': return
    c = fr.f_code; k = keys.get(c, 0)
    if k == 0:
        f = c.co_filename; rel = f[len(ROOT):] if f.startswith(ROOT) else None
        k = keys[c] = None if rel is None or SKIP.search(rel) or c.co_name.startswith('<') else (rel, c.co_firstlineno, getattr(c, 'co_qualname', c.co_name))
    if k and cur[0] in per: per[cur[0]][1].add(k); last[k] = next(tick)
if TESTS:
    import pytest
    class Probe:
        @pytest.hookimpl(hookwrapper=True)
        def pytest_runtest_call(self, item):
            cur[0] = item.nodeid; per[item.nodeid] = ['?', set()]; sys.setprofile(prof)
            try: yield
            finally: sys.setprofile(None)
        def pytest_runtest_logreport(self, report):
            if report.nodeid in per and (report.when == 'call' or report.failed): per[report.nodeid][0] = report.outcome if report.when == 'call' else 'error'
    pytest.main(['-q', '-p', 'no:cacheprovider', '--tb=line'] + TESTS, plugins=[Probe()])
else:
    code = sys.stdin.read(); per['run'] = ['passed', set()]; sys.setprofile(prof)
    try: exec(compile(code, '<issue>', 'exec'), dict(__name__='__main__'))
    except BaseException: sys.setprofile(None); per['run'][0] = 'failed'; traceback.print_exc(limit=-4)
    finally: sys.setprofile(None)
nm = lambda k: '%s:%d %s' % k
if FOCUS:
    hit = dict((n, o) for n, (o, s) in per.items() if any(k[2] == FOCUS or k[2].endswith('.' + FOCUS) for k in s))
    bad = [n for n, o in hit.items() if o != 'passed']
    print('\n=== %d of %d tests execute %s; %d of those do not pass' % (len(hit), len(per), FOCUS, len(bad)))
    for n in bad + [n for n in hit if n not in bad][:10]: print('  %-7s %s' % (hit[n], n))
else:
    fail = [s for o, s in per.values() if o != 'passed']; tot = collections.Counter(k for _, s in per.values() for k in s)
    ef = collections.Counter(k for s in fail for k in s); sc = dict((k, ef[k] / (len(fail) * tot[k]) ** 0.5) for k in ef)
    print('\n=== %d of %d runs fail; repo functions they execute, most suspicious first:' % (len(fail), len(per)) if fail else '\n=== nothing failed; %d repo functions executed' % len(tot))
    for k in (sorted(sc, key=lambda k: (-round(sc[k], 3), -last[k])) if fail else list(tot))[:25]: print('  %.2f %s' % (sc.get(k, 0), nm(k)))
