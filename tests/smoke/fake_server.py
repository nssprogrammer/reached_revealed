# scripted OpenAI-compatible server: plays a tiny agent policy so the A/B loop, tools, probe path substitution and grading can be tested
import json, re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

FIX = {'mini_1': ('mini/core.py', '    if x > hi:\n        return lo', '    if x > hi:\n        return hi'),
       'mini_2': ('mini/core.py', 'clamp(v, 0, 10)', 'clamp(v, 0, 100)')}
def call(name, **args): return {'id': f'call_{name}', 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args)}}

def policy(body):
    msgs = body['messages']; sys_ = msgs[0]['content'] if msgs[0]['role'] == 'system' else ''
    if 'tools' in body and 'echo hello' in msgs[-1].get('content', ''): return None, [call('run_command', command='echo hello')]
    if 'tools' not in body: return 'ok ' * 50, None
    user = next(m['content'] for m in msgs if m['role'] == 'user')
    task = re.search(r'TASK=(\w+)', user).group(1)
    n_tool = sum(1 for m in msgs if m['role'] == 'tool'); arm31 = 'THE PROBE' in sys_
    plan = [call('run_command', command="git grep -n 'def clamp' -- '*.py' | head")]
    if arm31:
        block = sys_[sys_.index('cat > '):]; block = block[:block.index('\nPROBE\n') + len('\nPROBE\n')]
        probe = re.search(r'cat > (\S+) <<', block).group(1)
        plan.append(call('run_command', command=block + f"timeout 60 python {probe} <<'EOF'\nfrom mini import clamp\nassert clamp(50, 0, 10) == 10\nEOF"))
    if task == 'mini_2' and not arm31: plan.append(None)                       # v2 on mini_2 gives up (no edit) -> nudges then stop
    else:
        f, a, b = FIX[task]; plan.append(call('edit_file', path=f, old_string=a, new_string=b))
        if arm31: plan.append(call('run_command', command=f"timeout 60 python {probe} --focus clamp tests/test_core.py < /dev/null 2>&1 | tail -5"))
        plan.append(call('run_command', command='git diff')); plan.append(call('submit_patch'))
    step = plan[min(n_tool, len(plan) - 1)] if n_tool < len(plan) else call('submit_patch')
    return ('I cannot find it.', None) if step is None else ('', [step])

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b'{"data":[{"id":"gemma"}]}')
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        content, calls = policy(body)
        msg = {'role': 'assistant', 'content': content}
        if calls: msg['tool_calls'] = calls
        n = sum(len(json.dumps(m)) for m in body['messages']) // 4
        out = {'choices': [{'message': msg, 'finish_reason': 'tool_calls' if calls else 'stop'}], 'usage': {'prompt_tokens': n, 'completion_tokens': 60}}
        b = json.dumps(out).encode(); self.send_response(200); self.send_header('Content-Type', 'application/json'); self.end_headers(); self.wfile.write(b)

if __name__ == '__main__':
    ThreadingHTTPServer(('127.0.0.1', 8000), H).serve_forever()
