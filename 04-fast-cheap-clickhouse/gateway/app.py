"""Transparent streaming recorder; all inference is pinned to OpenRouter."""
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
import httpx
from fast_mode import FAST_MODEL, ready_resources

app = FastAPI()
MODEL = 'deepseek/deepseek-v4.1-flash'
LOG = Path('/evidence/model-calls.jsonl')
RENDER_LOG = Path('/evidence/local-render-calls.jsonl')
LOCK = asyncio.Lock()
CLIENT = httpx.AsyncClient(timeout=httpx.Timeout(60, connect=10), trust_env=False)
CAP = 2.0
MAX_CALLS = int(os.environ.get('DEMO_MAX_MODEL_CALLS', '100'))


@app.get('/assets/plotly-3.1.0.min.js')
async def chart_library():
    return FileResponse('/assets/plotly-3.1.0.min.js', media_type='text/javascript',
                        headers={'Cache-Control': 'public, max-age=31536000, immutable',
                                 'Access-Control-Allow-Origin': '*',
                                 'Cross-Origin-Resource-Policy': 'cross-origin'})


@app.get('/assets/plotly-cartesian-3.1.0.min.js')
async def chart_library_cartesian():
    return FileResponse('/assets/plotly-cartesian-3.1.0.min.js', media_type='text/javascript',
                        headers={'Cache-Control': 'public, max-age=31536000, immutable',
                                 'Access-Control-Allow-Origin': '*',
                                 'Cross-Origin-Resource-Policy': 'cross-origin'})


@app.get('/health')
async def health():
    return {'ok': True}


@app.get('/v1/models')
async def models():
    return {'object': 'list', 'data': [{'id': MODEL, 'object': 'model', 'owned_by': 'deepseek'},
                                     {'id': FAST_MODEL, 'object': 'model', 'owned_by': 'local-app'}]}


def render_response(body, resources):
    """Pass through verified UI markers; no inference and no generated narration."""
    response_id = 'local-render-' + uuid.uuid4().hex
    content = '\n\n'.join(resource['marker'] for resource in resources)
    created = int(time.time())
    usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
    with RENDER_LOG.open('a') as f:
        f.write(json.dumps({'id': response_id, 'at': datetime.now(timezone.utc).isoformat(),
                            'source': 'deterministic-ui-renderer', 'inference_calls': 0,
                            'accounted_usd': 0, 'resources': resources}) + '\n')
    if not body.get('stream'):
        return JSONResponse({'id': response_id, 'object': 'chat.completion', 'created': created,
                             'model': FAST_MODEL, 'usage': usage,
                             'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': content},
                                          'finish_reason': 'stop'}]})

    async def stream():
        common = {'id': response_id, 'object': 'chat.completion.chunk', 'created': created, 'model': FAST_MODEL}
        yield 'data: ' + json.dumps({**common, 'choices': [{'index': 0,
            'delta': {'role': 'assistant', 'content': content}, 'finish_reason': None}]}) + '\n\n'
        yield 'data: ' + json.dumps({**common, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}], 'usage': usage}) + '\n\n'
        yield 'data: [DONE]\n\n'
    return StreamingResponse(stream(), media_type='text/event-stream',
                             headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@app.post('/v1/chat/completions')
async def completion(request: Request):
    body = await request.json()
    requested_model = body.get('model')
    if requested_model not in (MODEL, FAST_MODEL):
        raise HTTPException(400, 'This demo permits only its pinned model')
    if requested_model == FAST_MODEL:
        resources = ready_resources(body.get('messages', []))
        if resources:
            return render_response(body, resources)
    body['model'] = MODEL
    body.pop('max_completion_tokens', None)
    body.pop('reasoning_effort', None)
    body.pop('stop', None)
    body.update(temperature=0, max_tokens=2048, reasoning={'enabled': False}, service_tier='default')
    body['provider'] = {'only': ['inference-net/fp8'], 'quantizations': ['fp8'], 'allow_fallbacks': False, 'require_parameters': True}
    body['stream'] = True
    body['stream_options'] = {'include_usage': True}
    await LOCK.acquire()
    previous = [json.loads(line) for line in LOG.read_text().splitlines()] if LOG.exists() else []
    reserve = len(json.dumps(body).encode()) * 0.145e-6 + 2048 * 0.30e-6
    if len(previous) >= MAX_CALLS or sum(r['accounted_usd'] for r in previous) + reserve > CAP:
        LOCK.release()
        raise HTTPException(429, 'Demo request/spending cap reached')
    record = {'call_id': str(uuid.uuid4()), 'requested_model': requested_model, 'app_mode': 'fast' if requested_model == FAST_MODEL else 'standard', 'started_at': datetime.now(timezone.utc).isoformat(), 'request': body, 'events': [], 'first_content_ms': None, 'first_tool_delta_ms': None, 'error': None, 'accounted_usd': reserve}
    start = time.perf_counter()
    headers = {'Authorization': 'Bearer ' + os.environ['OPENROUTER_API_KEY'], 'X-OpenRouter-Title': 'ClickHouse LibreChat exploration demo'}
    try:
        response = await CLIENT.send(CLIENT.build_request('POST', 'https://openrouter.ai/api/v1/chat/completions', json=body, headers=headers), stream=True)
    except Exception as exc:
        record['error'] = type(exc).__name__
        record['total_ms'] = (time.perf_counter() - start) * 1000
        with LOG.open('a') as f:
            f.write(json.dumps(record) + '\n')
        LOCK.release()
        raise HTTPException(502, 'OpenRouter connection failed') from exc
    record['http_status'] = response.status_code
    if response.status_code != 200:
        data = await response.aread()
        await response.aclose()
        record['error'] = data.decode(errors='replace').replace(os.environ['OPENROUTER_API_KEY'], '<REDACTED>')[:4000]
        record['total_ms'] = (time.perf_counter() - start) * 1000
        with LOG.open('a') as f:
            f.write(json.dumps(record) + '\n')
        LOCK.release()
        return JSONResponse({'error': {'message': record['error']}}, status_code=response.status_code)

    async def stream():
        buffer = b''
        try:
            async for chunk in response.aiter_bytes():
                buffer += chunk
                while b'\n' in buffer:
                    line, buffer = buffer.split(b'\n', 1)
                    if not line.startswith(b'data:') or line[5:].strip() == b'[DONE]':
                        continue
                    try:
                        event = json.loads(line[5:].strip())
                    except json.JSONDecodeError:
                        continue
                    elapsed = (time.perf_counter() - start) * 1000
                    record['events'].append({'at_ms': elapsed, 'data': event})
                    if event.get('usage'):
                        record['usage'] = event['usage']
                        if event['usage'].get('cost') is not None:
                            record['accounted_usd'] = event['usage']['cost']
                    for choice in event.get('choices', []):
                        delta = choice.get('delta', {})
                        if delta.get('content') and record['first_content_ms'] is None:
                            record['first_content_ms'] = elapsed
                        if delta.get('tool_calls') and record['first_tool_delta_ms'] is None:
                            record['first_tool_delta_ms'] = elapsed
                yield chunk
        finally:
            await response.aclose()
            record['total_ms'] = (time.perf_counter() - start) * 1000
            record['ended_at'] = datetime.now(timezone.utc).isoformat()
            with LOG.open('a') as f:
                f.write(json.dumps(record) + '\n')
            LOCK.release()
    return StreamingResponse(stream(), media_type='text/event-stream', headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})
