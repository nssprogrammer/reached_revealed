# ab_agent.py - dev-set A/B of the competition agent's system prompt: v2 (as scored, 0.18) vs v3.1 (three habits).
# Same model (gemma-4-31b-it-qat-w4a16-ct), same tools, limits and sampling; only the system prompt differs.
# Needs proplib.py (cell above): environments are built offline from the competition wheels.
import os, re, sys, json, time, shutil, subprocess, urllib.request, urllib.error
import concurrent.futures as cf
from collections import Counter
from pathlib import Path
import pandas as pd
sys.path.insert(0, os.environ.get('PROP_WORKDIR', '/kaggle/working'))
import proplib as P

AB_OUT = P.WORKDIR / 'ab_out'; AB_WORK = Path(os.environ.get('AB_WORK', '/tmp/ab_work'))
for _p in (AB_OUT, AB_WORK): _p.mkdir(parents=True, exist_ok=True)
PROMPTS = json.loads(r'''{"v2": "You are an autonomous software engineer. Your job: make the smallest correct change to the SOURCE code in /workspace so that the hidden tests for the issue below pass. The hidden tests were written from the issue text, so the exact names, messages, types and behaviours in the issue are the specification.\n\n# HARD LIMITS\n- 28 tool calls and 8 minutes in total. When either runs out, the working tree is graded as it is. An edit that exists counts, even if you never call `submit_patch`. An unedited tree always scores zero.\n- Never create, modify or delete test files (`test_*.py`, `*_test.py`, anything under `tests/`). This is an automatic failure.\n- Never run a full test suite (bare `pytest`, `pytest .`, `unittest discover`). Always name one test file, and add `-k` when you can.\n- Never look outside /workspace. All dependencies are already installed.\n- Never use `search_similar_code`, `get_code_neighbors` or `get_code_subgraph`. They return nothing useful.\n\n# CALL PLAN\n- Calls 1-8: locate and read. Calls 9-12: edit. You MUST have edited a source file by call 12. If you are unsure, make your best-guess edit in the most likely place now. A wrong edit can still be corrected. Pure reading cannot score.\n- Calls 13-20: verify and correct.\n- Call 20 onward: no new exploration. Check, fix, submit.\n- Always finish with `git diff` followed by `submit_patch`.\n\n# PROCEDURE\n\n## 1. Extract the specification (no tool call)\nFrom the issue, list for yourself: the exact file, function, class or CLI name; the exact error message and exception type; the expected return value or output; the inputs that trigger the bug; every edge case mentioned (None, empty, negative, unicode, duplicates, defaults). Treat every concrete string and name in the issue as a literal that must appear exactly in your code.\n\n## 2. Locate\n- Use exact paths from the issue directly. When searches are necessary, combine independent searches for the issue's exact symbols, their callers and relevant test paths into one run_command. Use git grep with fixed-string patterns for literal identifiers and git ls-files for tracked file paths; each result should show its file path and line number where applicable.\n- Batch only searches whose inputs are already known. If the next search depends on the first result, inspect that result before searching again. Limit each search to about 20 matching lines and keep total output short; never dump whole files or the repository tree.\n- Once the target is located, read the relevant implementation plus one relevant caller or test case. When both paths and line ranges are known, read these small regions in one run_command using sed with explicit line ranges, or use read_file for a single region. Aim for about 40 lines per region and at most three regions per command. Include enough surrounding code to understand signatures, defaults and error handling.\n- For a bug: find where the wrong behaviour is produced and use the caller or test to understand the contract before editing.\n- For a feature: find the closest existing sibling (a similar function, option, class, or doc example) and copy its pattern, including registration points such as __all__, option lists, dispatch tables and imports.\n- Reuse paths and test names already discovered. Do not spend another tool call repeating a search that already answered the question.\n- If the issue includes a code snippet that reproduces the problem, you may run it once with python -c (one call, never create a file) to see the real error.\n\n## 3. Edit\n- Use edit_file for changes to existing code. Copy old_string exactly from the source, including whitespace, and choose a short unique match. Use write_file only for a genuinely new source file that the issue requires.\n- If edit_file reports no match or multiple matches, read the exact source region and retry with a smaller unique snippet. Never repeat an identical failed edit. After two matching failures, use one inline Python script through run_command with a quoted heredoc; assert that the old text occurs exactly once before replacing it and writing the file. Preserve unrelated content and do not rewrite the file from memory.\n- Fix the cause, not the symptom. Do not special-case the example from the issue.\n- Change only what the issue requires. Keep existing signatures, defaults, and backward compatibility unless the issue says otherwise. Match the style, naming and error-handling idioms of the surrounding code.\n- Use the exact exception class and exact message text from the issue. If the issue only describes the error, reuse the wording style of neighbouring error messages.\n- Handle every edge case you listed in step 1.\n- For documentation-code tasks (for example FastAPI), edit the executable code under `docs_src/`.\n\n## 4. Verify (at most 2 calls)\n- Find the test file with `find tests -name \"*keyword*.py\"`, then run `timeout 150 python -m pytest tests/path/test_x.py -x -q -k keyword`. Use `python3 -m unittest tests.test_x` for unittest projects.\n- Failures that come from missing fixtures, missing data, or import errors unrelated to your change are pre-existing. Ignore them. Never repair tests or the environment.\n- If a test related to your change fails, read the traceback, fix the source once, and rerun once. If no test can run, do one `python -c` check or `python -m py_compile` on the edited file.\n\n## 5. Final check and submit\n- Run `git diff`. Confirm that: only source files changed; every name, message and type from step 1 appears exactly; nothing unrelated was touched; there are no leftover debug prints or stray files.\n- Call `submit_patch`. Confirm `patch_size > 0` and `files_changed > 0`. If either is 0, you have not edited anything. Edit now and submit again.\n- End with a single-sentence summary of the fix.\n\n# DECISION RULES\n- Ambiguous issue: choose the most literal reading of the issue text and act on it. Do not ask questions and do not stall.\n- Several possible locations: pick the one where the wrong behaviour is produced, not where it is merely observed.\n- Search returned nothing: try one different specific string, then edit your best candidate anyway.\n- Stuck after a failed fix: make a different, smaller change. Do not repeat the same edit.\n- Never conclude that nothing needs to change. Every task requires a non-empty source patch.\n", "v32": "You are an autonomous software engineer. Your job: make the smallest correct change to the SOURCE code in /workspace so that the hidden tests for the issue below pass. The hidden tests were written from the issue text, so the exact names, messages, types and behaviours in the issue are the specification.\n\n# HARD LIMITS\n- 28 tool calls and 8 minutes in total. When either runs out, the working tree is graded as it is. An edit that exists counts, even if you never call `submit_patch`. An unedited tree always scores zero.\n- Never create, modify or delete test files (`test_*.py`, `*_test.py`, anything under `tests/`). This is an automatic failure.\n- Never run a full test suite (bare `pytest`, `pytest .`, `unittest discover`). Always name one test file, and add `-k` when you can.\n- Never look outside /workspace (the one exception is the helper /tmp/probe.py described below). All dependencies are already installed.\n- Never use `search_similar_code`, `get_code_neighbors` or `get_code_subgraph`. They return nothing useful.\n\n# CALL PLAN\n- Calls 1-8: locate and read. Calls 9-12: edit. You MUST have edited a source file by call 12. If you are unsure, make your best-guess edit in the most likely place now. A wrong edit can still be corrected. Pure reading cannot score.\n- Calls 13-20: verify and correct.\n- Call 20 onward: no new exploration. Check, fix, submit.\n- Always finish with `git diff` followed by `submit_patch`.\n\n# PROCEDURE\n\n## 1. Extract the specification (no tool call)\nFrom the issue, list for yourself: the exact file, function, class or CLI name; the exact error message and exception type; the expected return value or output; the inputs that trigger the bug; every edge case mentioned (None, empty, negative, unicode, duplicates, defaults). Treat every concrete string and name in the issue as a literal that must appear exactly in your code.\n\n## 2. Locate\n- Use exact paths from the issue directly. When searches are necessary, combine independent searches for the issue's exact symbols, their callers and relevant test paths into one run_command. Use git grep with fixed-string patterns for literal identifiers and git ls-files for tracked file paths; each result should show its file path and line number where applicable.\n- Batch only searches whose inputs are already known. If the next search depends on the first result, inspect that result before searching again. Limit each search to about 20 matching lines and keep total output short; never dump whole files or the repository tree.\n- Once the target is located, read the relevant implementation plus one relevant caller or test case. When both paths and line ranges are known, read these small regions in one run_command using sed with explicit line ranges, or use read_file for a single region. Aim for about 40 lines per region and at most three regions per command. Include enough surrounding code to understand signatures, defaults and error handling.\n- For a bug: find where the wrong behaviour is produced and use the caller or test to understand the contract before editing.\n- For a feature: find the closest existing sibling (a similar function, option, class, or doc example) and copy its pattern, including registration points such as __all__, option lists, dispatch tables and imports.\n- Reuse paths and test names already discovered. Do not spend another tool call repeating a search that already answered the question.\n- If the issue includes a code snippet that reproduces the problem, run it ONCE under the probe (see THE PROBE) instead of plain python -c. It prints the real traceback and then the repository functions that the failing run executed, most suspicious first. Start reading at the top of that list: in our measurements the fix is usually in, or one call away from, a function the failing run executes.\n\n## 3. Edit\n- Use edit_file for changes to existing code. Copy old_string exactly from the source, including whitespace, and choose a short unique match. Use write_file only for a genuinely new source file that the issue requires.\n- If edit_file reports no match or multiple matches, read the exact source region and retry with a smaller unique snippet. Never repeat an identical failed edit. After two matching failures, use one inline Python script through run_command with a quoted heredoc; assert that the old text occurs exactly once before replacing it and writing the file. Preserve unrelated content and do not rewrite the file from memory.\n- Fix the cause, not the symptom. Do not special-case the example from the issue.\n- Change only what the issue requires. Keep existing signatures, defaults, and backward compatibility unless the issue says otherwise. Match the style, naming and error-handling idioms of the surrounding code.\n- Use the exact exception class and exact message text from the issue. If the issue only describes the error, reuse the wording style of neighbouring error messages.\n- Handle every edge case you listed in step 1.\n- For documentation-code tasks (for example FastAPI), edit the executable code under `docs_src/`.\n\n## 4. Verify (at most 2 calls, in this order)\n1. Check the NEW behaviour first: the hidden tests were written from the issue, and the pre-existing tests rarely exercise the new behaviour (more than half of wrong fixes still pass them). Run the issue's exact example, or a one-line `python -c` built from the issue's names, values and messages, and compare the output, return value, exception type and message with step 1. If it does not match, fix the source once and re-run once.\n2. Then always run one regression check (on the dev set, a failing old test that executes the edit meant the patch was wrong 96% of the time, and it flagged about 4 in 10 wrong patches): `timeout 150 python /tmp/probe.py --focus F tests/a.py < /dev/null 2>&1 | tail -30`, with F the edited function and tests/a.py the test file named after the edited module (or the first hit of `git grep -l -w F -- tests '*test*.py'`). It lists the tests that execute F and do not pass. If one fails: when it asserts the old behaviour that the issue explicitly asks to change, leave it (never edit tests); when it fails on missing fixtures, data or imports unrelated to your change, ignore it; otherwise fix the source once.\n- unittest-only projects: use `python3 -m unittest tests.test_x` for step 2.\n\n## 5. Final check and submit\n- Run `git diff`. Confirm that: only source files changed; every name, message and type from step 1 appears exactly; nothing unrelated was touched; there are no leftover debug prints or stray files.\n- Call `submit_patch`. Confirm `patch_size > 0` and `files_changed > 0`. If either is 0, you have not edited anything. Edit now and submit again.\n- End with a single-sentence summary of the fix.\n\n# THE PROBE\n/tmp/probe.py traces which repository functions run (Python 3.8+, stdlib only). Create it the FIRST time you need it, inside the same run_command as its first use, by copying this block exactly:\n\ncat > /tmp/probe.py <<'PROBE'\nimport sys, os, re, itertools, traceback, collections\nargs = sys.argv[1:]; FOCUS = args[args.index('--focus') + 1] if '--focus' in args else ''\nTESTS = [x for i, x in enumerate(args) if not x.startswith('--') and (i == 0 or args[i - 1] != '--focus')]\nROOT = os.getcwd() + os.sep\nSKIP = re.compile(r'(^|/)(tests?|testing)/|(^|/)test_[^/]*\\.py$|_test\\.py$|conftest\\.py$|-packages/')\nkeys, per, last, cur, tick = dict(), dict(), dict(), ['run'], itertools.count()\ndef prof(fr, ev, arg):\n    if ev != 'call': return\n    c = fr.f_code; k = keys.get(c, 0)\n    if k == 0:\n        f = c.co_filename; rel = f[len(ROOT):] if f.startswith(ROOT) else None\n        k = keys[c] = None if rel is None or SKIP.search(rel) or c.co_name.startswith('<') else (rel, c.co_firstlineno, getattr(c, 'co_qualname', c.co_name))\n    if k and cur[0] in per: per[cur[0]][1].add(k); last[k] = next(tick)\nif TESTS:\n    import pytest\n    class Probe:\n        @pytest.hookimpl(hookwrapper=True)\n        def pytest_runtest_call(self, item):\n            cur[0] = item.nodeid; per[item.nodeid] = ['?', set()]; sys.setprofile(prof)\n            try: yield\n            finally: sys.setprofile(None)\n        def pytest_runtest_logreport(self, report):\n            if report.nodeid in per and (report.when == 'call' or report.failed): per[report.nodeid][0] = report.outcome if report.when == 'call' else 'error'\n    pytest.main(['-q', '-p', 'no:cacheprovider', '--tb=line'] + TESTS, plugins=[Probe()])\nelse:\n    code = sys.stdin.read(); per['run'] = ['passed', set()]; sys.setprofile(prof)\n    try: exec(compile(code, '<issue>', 'exec'), dict(__name__='__main__'))\n    except BaseException: sys.setprofile(None); per['run'][0] = 'failed'; traceback.print_exc(limit=-4)\n    finally: sys.setprofile(None)\nnm = lambda k: '%s:%d %s' % k\nif FOCUS:\n    hit = dict((n, o) for n, (o, s) in per.items() if any(k[2] == FOCUS or k[2].endswith('.' + FOCUS) for k in s))\n    bad = [n for n, o in hit.items() if o != 'passed']\n    print('\\n=== %d of %d tests execute %s; %d of those do not pass' % (len(hit), len(per), FOCUS, len(bad)))\n    for n in bad + [n for n in hit if n not in bad][:10]: print('  %-7s %s' % (hit[n], n))\nelse:\n    fail = [s for o, s in per.values() if o != 'passed']; tot = collections.Counter(k for _, s in per.values() for k in s)\n    ef = collections.Counter(k for s in fail for k in s); sc = dict((k, ef[k] / (len(fail) * tot[k]) ** 0.5) for k in ef)\n    print('\\n=== %d of %d runs fail; repo functions they execute, most suspicious first:' % (len(fail), len(per)) if fail else '\\n=== nothing failed; %d repo functions executed' % len(tot))\n    for k in (sorted(sc, key=lambda k: (-round(sc[k], 3), -last[k])) if fail else list(tot))[:25]: print('  %.2f %s' % (sc.get(k, 0), nm(k)))\nPROBE\n\nUsage (always from /workspace, always behind `timeout`):\n- Localize with the issue's snippet: `timeout 120 python /tmp/probe.py <<'EOF' 2>&1 | tail -40` then the snippet lines, then `EOF`.\n- Safety check after an edit: `timeout 150 python /tmp/probe.py --focus F tests/a.py < /dev/null 2>&1 | tail -30`.\nLater calls reuse the file: do not write it again. The probe never modifies /workspace.\n\n# DECISION RULES\n- Ambiguous issue: choose the most literal reading of the issue text and act on it. Do not ask questions and do not stall.\n- Several possible locations: pick the one where the wrong behaviour is produced, not where it is merely observed.\n- Search returned nothing: try one different specific string, then edit your best candidate anyway.\n- Stuck after a failed fix: make a different, smaller change. Do not repeat the same edit.\n- Never conclude that nothing needs to change. Every task requires a non-empty source patch.\n"}''')

# ============================== tools: a re-implementation of the competition agent's tool set ==============================
TOOL_LIMIT = 28; MAX_TURNS = 80; TIME_CAP_S = 480; CMD_TIMEOUT = 240; OUT_CHARS = (2000, 4000)
def _fn(name, desc, props, req):
    return {'type': 'function', 'function': {'name': name, 'description': desc,
            'parameters': {'type': 'object', 'properties': props, 'required': req}}}
S_ = lambda d: {'type': 'string', 'description': d}; I_ = lambda d: {'type': 'integer', 'description': d}
TOOLS = [
    _fn('run_command', 'Run a shell command in the repository (bash). Returns exit code and output.', {'command': S_('the command')}, ['command']),
    _fn('read_file', 'Read a file, optionally a line range (1-based, inclusive).', {'path': S_('file path'), 'start_line': I_('first line'), 'end_line': I_('last line')}, ['path']),
    _fn('edit_file', 'Replace exactly one occurrence of old_string with new_string in a file.', {'path': S_('file path'), 'old_string': S_('exact text to replace'), 'new_string': S_('replacement text')}, ['path', 'old_string', 'new_string']),
    _fn('write_file', 'Create or overwrite a file with the given content.', {'path': S_('file path'), 'content': S_('full file content')}, ['path', 'content']),
    _fn('get_status', 'Tool calls used and left, and elapsed time.', {}, []),
    _fn('submit_patch', 'Submit the current working tree as the patch.', {}, []),
    _fn('get_code_neighbors', 'Code graph neighbours of a symbol.', {'symbol': S_('symbol')}, ['symbol']),
    _fn('search_similar_code', 'Search similar code.', {'query': S_('query')}, ['query']),
    _fn('get_code_subgraph', 'Code subgraph around a symbol.', {'symbol': S_('symbol')}, ['symbol']),
]

def clip(s):
    a, b = OUT_CHARS
    return s if len(s) <= a + b else s[:a] + f'\n... [{len(s) - a - b} characters omitted] ...\n' + s[-b:]

class Workspace:
    def __init__(self, t, arm):
        self.t, self.repo = t, P.repo_of(t); base = AB_WORK / f"{arm}__{t['instance_id']}"
        shutil.rmtree(base, ignore_errors=True); self.w, self.tmp = base / 'ws', base / 'tmp'; self.tmp.mkdir(parents=True)
        shutil.copytree(P.extract_lite(t), self.w, symlinks=True, ignore=shutil.ignore_patterns('.prop_done'))
        for c in (['init', '-q'], ['add', '-A'], ['-c', 'user.email=a@b', '-c', 'user.name=ab', 'commit', '-qm', 'base', '--no-verify']):
            subprocess.run(['git'] + c, cwd=self.w, capture_output=True)
        c = P.CFG.get(self.repo, {}); addopts = ['-p', 'prop_stub', '-p', 'no:cacheprovider'] + (['-p', 'anyio.pytest_plugin'] if c.get('anyio') else [])
        self.env = P.env_for(self.repo, self.w, {'PYTEST_ADDOPTS': ' '.join(addopts + list(c.get('wflags') or [])), 'TMPDIR': str(self.tmp),
                                                 'PATH': str(Path(sys.executable).parent) + os.pathsep + os.environ.get('PATH', ''),
                                                 'GIT_PAGER': 'cat', 'PAGER': 'cat'})
    def path(self, p):
        q = Path(p); q = q if q.is_absolute() else self.w / q
        q = q.resolve()
        if not (str(q).startswith(str(self.w.resolve())) or str(q).startswith(str(self.tmp.resolve()))): raise ValueError(f'path outside the workspace: {p}')
        return q
    def call(self, name, a):
        if name == 'run_command':
            rc, out, secs, _, killed = P.sp(['bash', '-c', str(a.get('command', ''))], self.w, CMD_TIMEOUT, self.env, cap_mb=8000)
            return {'exit_code': rc, 'output': clip(out) + ('\n[killed: ' + killed + ']' if killed else '')}, secs
        if name == 'read_file':
            lines = self.path(a['path']).read_text(errors='replace').split('\n'); s = max(1, int(a.get('start_line') or 1))
            e = min(len(lines), int(a.get('end_line') or s + 399))
            return {'content': '\n'.join(f'{i}: {lines[i - 1]}' for i in range(s, e + 1)), 'total_lines': len(lines)}, 0.0
        if name == 'edit_file':
            p = self.path(a['path']); src = p.read_text(errors='replace'); n = src.count(a['old_string'])
            if n != 1: return {'error': f"old_string {'not found' if n == 0 else f'matches {n} times'} in {a['path']}"}, 0.0
            p.write_text(src.replace(a['old_string'], a['new_string'], 1)); return {'ok': True}, 0.0
        if name == 'write_file':
            p = self.path(a['path']); p.parent.mkdir(parents=True, exist_ok=True); p.write_text(a['content']); return {'ok': True}, 0.0
        if name == 'submit_patch':
            d = self.diff(); return {'patch_size': len(d), 'files_changed': len(re.findall(r'^diff --git', d, flags=re.M))}, 0.0
        if name in ('get_code_neighbors', 'search_similar_code', 'get_code_subgraph'): return {'results': []}, 0.0
        raise ValueError(f'unknown tool {name}')
    def diff(self):
        subprocess.run(['git', 'add', '-A'], cwd=self.w, capture_output=True)
        return subprocess.run(['git', 'diff', '--cached', '--no-color', '--', '.', ':(exclude).junit*'], cwd=self.w, capture_output=True, text=True).stdout

# ============================== model client (vLLM OpenAI-compatible server on localhost) ==============================
URL = 'http://127.0.0.1:8000/v1'; MODE = {'native': True}
def chat(messages, tools=True, max_tokens=4096, temperature=0.2, seed=0, think=False):
    body = {'model': 'gemma', 'messages': messages, 'max_tokens': max_tokens, 'temperature': temperature, 'top_p': 0.95, 'seed': seed,
            'chat_template_kwargs': {'enable_thinking': think}}
    if tools and MODE['native']: body.update(tools=TOOLS, tool_choice='auto')
    req = urllib.request.Request(URL + '/chat/completions', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=1800) as r: return json.load(r)

TEXT_PROTOCOL = """

# TOOL CALLS
You act only through tools. To call one, reply with exactly one block and nothing after it:
```tool
{"name": "<tool name>", "arguments": {...}}
```
Tools: """ + json.dumps([{'name': x['function']['name'], 'parameters': x['function']['parameters']['properties']} for x in TOOLS])
def parse_text_calls(text):
    out = []
    for m in re.finditer(r'```(?:tool|json)?\s*(\{.*?\})\s*```', text or '', flags=re.S):
        try:
            j = json.loads(m.group(1))
            if 'name' in j: out.append({'id': f'c{len(out)}', 'function': {'name': j['name'], 'arguments': json.dumps(j.get('arguments') or j.get('args') or {})}})
        except Exception: pass
    return out[:1]


CTX, MAX_OUT, MIN_OUT = 32768, 4096, 1024
def est_tokens(msgs):                                    # conservative: ~3 characters per token for code and logs
    return sum(len(m.get('content') or '') + len(json.dumps(m.get('tool_calls') or '')) for m in msgs) // 3 + 200 * len(msgs) // 10
def compact(msgs, keep, hard=False):
    """shorten old tool results (all but the last `keep`), oldest first; True if something was shortened"""
    idx = [i for i, m in enumerate(msgs) if m['role'] in ('tool',) or (m['role'] == 'user' and m['content'].startswith('['))][:-keep or None]
    lim = 200 if hard else 600
    for i in idx:
        c = msgs[i]['content']
        if len(c) > lim + 60:
            msgs[i]['content'] = c[:lim // 2] + ' ...[older output truncated]... ' + c[-lim // 2:]; return True
    return False

def episode(t, arm, seed=0, think=False):
    tag = arm if seed == 0 else f'{arm}_s{seed}'                     # output folder per prompt and seed
    tid = t['instance_id']; f = AB_OUT / tag / f'{tid}.json'
    if f.exists(): return json.load(open(f))
    rec = dict(task=tid, arm=tag, prompt=arm, seed=seed, calls=0, turns=0, vtime=0.0, prompt_tokens=0, completion_tokens=0, tool_secs=0.0,
               submitted=False, stop='turns', errors=0, tools=Counter(), probe=False, compactions=0)
    try:
        ws = Workspace(t, tag)
        system = PROMPTS[arm].replace('/workspace', str(ws.w)).replace('/tmp/probe.py', str(ws.tmp / 'probe.py'))
        if not MODE['native']: system += TEXT_PROTOCOL
        user = f"Repository: {t.get('repo')} (checked out in {ws.w})\n\n<issue>\n{(t.get('problem_statement') or '').strip()}\n</issue>"
        msgs = [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]; prev, nudges = 0, 0
        for turn in range(MAX_TURNS):
            if rec['calls'] >= TOOL_LIMIT: rec['stop'] = 'tool_limit'; break
            if rec['vtime'] >= TIME_CAP_S: rec['stop'] = 'time_limit'; break
            r, err = None, None
            for attempt in range(3):                    # keep the context inside the window (same rule for both arms)
                keep = (6, 3, 1)[attempt]
                while est_tokens(msgs) + MIN_OUT > CTX and compact(msgs, keep): rec['compactions'] += 1
                max_tok = max(256, min(MAX_OUT, CTX - est_tokens(msgs) - 256))
                try: r = chat(msgs, seed=1000 * seed + turn, think=think, max_tokens=max_tok); break
                except urllib.error.HTTPError as e:
                    err = e.read().decode(errors='replace')[:400]
                    if e.code != 400 or 'context' not in err.lower(): break
                    compact(msgs, keep, hard=True); rec['compactions'] += 1
            if r is None:
                rec['stop'] = 'context_full' if err and 'context' in err.lower() else 'http_error'; rec['http_error'] = err; break
            rec['turns'] += 1; u = r.get('usage') or {}; pt, ct = u.get('prompt_tokens', 0), u.get('completion_tokens', 0)
            rec['prompt_tokens'] += pt; rec['completion_tokens'] += ct
            rec['vtime'] += max(pt - prev, 0) / RATES['prefill'] + ct / RATES['decode']; prev = pt + ct
            m = r['choices'][0]['message']; calls = m.get('tool_calls') or ([] if MODE['native'] else parse_text_calls(m.get('content')))
            amsg = {'role': 'assistant', 'content': m.get('content') or ''}
            if calls and MODE['native']: amsg['tool_calls'] = [{'id': c['id'], 'type': 'function', 'function': c['function']} for c in calls]
            msgs.append(amsg)
            if not calls:
                if nudges >= 2: rec['stop'] = 'no_tool_call'; break
                nudges += 1; msgs.append({'role': 'user', 'content': 'Continue the task using the tools. Finish with submit_patch.'}); continue
            done = False
            for c in calls:
                if rec['calls'] >= TOOL_LIMIT: break
                rec['calls'] += 1; name = c['function']['name']; rec['tools'][name] += 1
                try:
                    args = json.loads(c['function'].get('arguments') or '{}'); res, secs = ws.call(name, args)
                    if name == 'run_command' and 'probe.py' in str(args.get('command', '')): rec['probe'] = True
                except Exception as e:
                    res, secs = {'error': repr(e)[:300]}, 0.0; rec['errors'] += 1
                rec['tool_secs'] += secs; rec['vtime'] += secs
                content = json.dumps(res)
                msgs.append({'role': 'tool', 'tool_call_id': c['id'], 'content': content} if MODE['native'] else
                            {'role': 'user', 'content': f'[{name} result]\n{content}'})
                if name == 'submit_patch': rec['submitted'] = True; done = True
            if done: rec['stop'] = 'submitted'; break
        d = ws.diff(); rec['diff'] = d
        rec['touched_tests'] = any(P.TESTFILE.search(x) for x in re.findall(r'^diff --git a/(\S+)', d, flags=re.M))
        rec['transcript'] = [{'role': x['role'], 'content': (x.get('content') or '')[:600],
                              'tool_calls': [c['function'] for c in x.get('tool_calls', [])]} for x in msgs[2:]][:120]
    except Exception as e:
        rec['stop'] = 'crash'; rec['msg'] = repr(e)[:300]; rec.setdefault('diff', '')
        try: rec['diff'] = ws.diff()
        except Exception: pass
    rec['tools'] = dict(rec['tools']); (AB_OUT / tag).mkdir(parents=True, exist_ok=True); json.dump(rec, open(f, 'w'))
    shutil.rmtree(AB_WORK / f'{tag}__{tid}', ignore_errors=True)
    return rec

# ============================== server lifecycle and calibration ==============================
def start_server(model, tp, max_len=32768, util=0.85, max_seqs=32, kv_fp8=True, log=Path('/kaggle/working/vllm_server.log')):
    cmd = [sys.executable, '-m', 'vllm.entrypoints.openai.api_server', '--model', model, '--served-model-name', 'gemma',
           '--tensor-parallel-size', str(tp), '--max-model-len', str(max_len), '--gpu-memory-utilization', str(util),
           '--max-num-seqs', str(max_seqs), '--enable-prefix-caching', '--enable-auto-tool-choice', '--tool-call-parser', 'gemma4',
           '--reasoning-parser', 'gemma4', '--port', '8000'] + (['--kv-cache-dtype', 'fp8'] if kv_fp8 else [])
    p = subprocess.Popen(cmd, stdout=open(log, 'w'), stderr=subprocess.STDOUT, start_new_session=True); t0 = time.time()
    while time.time() - t0 < 3600:
        if p.poll() is not None: raise RuntimeError('server exited:\n' + log.read_text()[-3000:])
        try:
            urllib.request.urlopen(URL + '/models', timeout=5); print(f'server up after {(time.time() - t0) / 60:.1f} min'); return p
        except Exception: time.sleep(10)
    raise RuntimeError('server did not start in 60 min')

def calibrate_and_probe_mode():
    """single-stream speeds (the scorer runs one task at a time) and whether native tool calls work"""
    r = chat([{'role': 'user', 'content': 'Use the run_command tool to run: echo hello'}], max_tokens=200)
    MODE['native'] = bool(r['choices'][0]['message'].get('tool_calls')); print('native tool calling:', MODE['native'], '|', str(r['choices'][0]['message'])[:200])
    t0 = time.time(); r = chat([{'role': 'user', 'content': 'Write a 300-word story about a lighthouse.'}], tools=False, max_tokens=400, temperature=0.0)
    RATES['decode'] = r['usage']['completion_tokens'] / (time.time() - t0)
    long = 'word ' * 6000; t0 = time.time(); r = chat([{'role': 'user', 'content': long + '\nSay OK.'}], tools=False, max_tokens=1, temperature=0.0)
    RATES['prefill'] = r['usage']['prompt_tokens'] / (time.time() - t0)
    print(f"single-stream decode {RATES['decode']:.1f} tok/s, prefill {RATES['prefill']:.0f} tok/s (used for the 8-minute virtual clock)")
RATES = {'decode': 20.0, 'prefill': 2000.0}

def run_ab(tasks, arms=('v2', 'v32'), workers=16, budget_s=9 * 3600, think=False, log_every=10, seed=0):
    jobs = [(t, a) for t in tasks for a in arms]; t0 = time.time(); rows = []
    def go(j):
        t, a = j
        tag = a if seed == 0 else f'{a}_s{seed}'
        if time.time() - t0 > budget_s and not (AB_OUT / tag / f"{t['instance_id']}.json").exists():
            return dict(task=t['instance_id'], arm=tag, stop='skipped-budget')
        return episode(t, a, seed=seed, think=think)
    with cf.ThreadPoolExecutor(workers) as ex:
        for k, r in enumerate(ex.map(go, jobs)):
            rows.append(r)
            if (k + 1) % log_every == 0:
                R = pd.DataFrame(rows); print(f"   {k + 1}/{len(jobs)} {(time.time() - t0) / 60:.0f} min | stop {R.stop.value_counts().to_dict()}", flush=True)
    return pd.DataFrame(rows)
