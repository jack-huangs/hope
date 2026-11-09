"""Serial streaming benchmark for the school endpoint; never persist API keys."""
import getpass
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

URL = 'https://apiai.sztu.edu.cn/v1/chat/completions'
MODEL = 'deepseek-v4-pro'
PROMPT = ('请写一篇不少于两千字的中文科普文章，详细解释大语言模型推理过程中的预填充、'
          '逐token生成、显存带宽和KV缓存。使用连贯段落，给出具体例子，不要列表或表格。')

def measure(session, key, name, thinking, limit, prompt):
    payload = dict(model=MODEL, messages=[dict(role='user', content=prompt)],
                   stream=True, stream_options={'include_usage': True},
                   max_tokens=limit, temperature=0.7,
                   chat_template_kwargs={'thinking': thinking, 'reasoning_effort': 'high'})
    started = time.perf_counter()
    first = first_content = last = None
    content = reasoning = ''
    events = []
    usage = None
    finish = None
    done = False
    with session.post(URL, json=payload, headers={'Authorization': 'Bearer ' + key},
                      stream=True, timeout=(30, 600), allow_redirects=False) as response:
        if response.status_code != 200:
            raise RuntimeError(f'HTTP {response.status_code}: ' + response.text[:600].replace(key, '[REDACTED]'))
        headers_time = time.perf_counter()
        print(name, 'HTTP 200; streaming', flush=True)
        response.encoding = 'utf-8'
        for line in response.iter_lines(chunk_size=1, decode_unicode=True):
            if not line or not line.startswith('data:'):
                continue
            data = line[5:].strip()
            if data == '[DONE]':
                done = True
                break
            obj = json.loads(data)
            if obj.get('error'):
                raise RuntimeError(str(obj['error']).replace(key, '[REDACTED]'))
            if obj.get('usage'):
                usage = obj['usage']
            for choice in obj.get('choices', []):
                delta = choice.get('delta') or {}
                c, r = delta.get('content') or '', delta.get('reasoning_content') or ''
                if c or r:
                    now = time.perf_counter()
                    if first is None:
                        first = now
                        print(name, f'first generated text at {now-started:.3f}s', flush=True)
                    if c and first_content is None:
                        first_content = now
                    last = now
                    content += c
                    reasoning += r
                    events.append({'elapsed_s': now-started, 'content_chars': len(c), 'reasoning_chars': len(r)})
                if choice.get('finish_reason'):
                    finish = choice['finish_reason']
        ended = time.perf_counter()
    if not done or not usage or first is None:
        raise RuntimeError('Incomplete stream or missing authoritative token usage')
    n = usage['completion_tokens']
    result = dict(name=name, thinking=thinking, max_tokens=limit, model=MODEL,
                  header_s=headers_time-started, ttft_s=first-started,
                  first_answer_s=None if first_content is None else first_content-started,
                  generation_span_s=last-first, total_s=ended-started,
                  approximate_decode_tps=(n-1)/(last-first) if last>first and n>1 else None,
                  end_to_end_tps=n/(ended-started), usage=usage, finish_reason=finish,
                  stream_done=done, events=events, answer=content,
                  reasoning_chars=len(reasoning))
    print(json.dumps({k:v for k,v in result.items() if k not in ('events','answer')}, ensure_ascii=True), flush=True)
    return result

def main():
    key = getpass.getpass('School API key: ')
    out = Path('runs/school_api_benchmark_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    out.mkdir(parents=True)
    session = requests.Session()
    results = []
    tests = [('warmup', False, 32, '请用一句话介绍预填充和逐token生成的区别。')]
    tests += [(f'non_thinking_{i}', False, 512, PROMPT) for i in range(1,4)]
    tests += [('thinking_high_1', True, 512, PROMPT)]
    for name, thinking, limit, prompt in tests:
        try:
            result = measure(session, key, name, thinking, limit, prompt)
        except Exception as exc:
            print('FAILED', str(exc).replace(key, '[REDACTED]'), flush=True)
            raise SystemExit(1)
        results.append(result)
        (out/'results.json').write_text(json.dumps({'endpoint': URL, 'tested_at_utc': datetime.now(timezone.utc).isoformat(), 'prompt': PROMPT, 'results': results}, ensure_ascii=False, indent=2), encoding='utf-8')
        time.sleep(1)
    normal = [r for r in results if r['name'].startswith('non_thinking')]
    summary = {k: {'mean': statistics.mean(r[k] for r in normal), 'median': statistics.median(r[k] for r in normal), 'min': min(r[k] for r in normal), 'max': max(r[k] for r in normal)} for k in ['approximate_decode_tps', 'end_to_end_tps', 'ttft_s']}
    (out/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('SUMMARY', json.dumps(summary), 'RESULTS', out, flush=True)

if __name__ == '__main__':
    main()
