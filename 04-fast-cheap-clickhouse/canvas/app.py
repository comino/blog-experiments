import asyncio
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import sqlite3
import time
import uuid
import re

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
import httpx
from pydantic import BaseModel, Field

from domain import blank, now, validate_state, validate_plan, apply_operations, resolve_intent
from selection import build_scope, apply_scope
from datasets import (dataset_id, DATASETS, catalogue, validate_sql, playback_scope, range_sql,
                      playback_range, vessel_sql, storm_sql, MAX_PLAYBACK_POINTS, MAX_STORM_POINTS,
                      MAX_PLAYBACK_BYTES)

ROOT = Path(__file__).parent
STORE = Path(os.environ.get('CANVAS_STORE', '/data/canvas.sqlite'))
STORE.parent.mkdir(parents=True, exist_ok=True)
DB = sqlite3.connect(STORE, check_same_thread=False)
DB.execute('PRAGMA journal_mode=WAL')
DB.execute('CREATE TABLE IF NOT EXISTS workspaces (owner TEXT PRIMARY KEY, version INTEGER NOT NULL, body TEXT NOT NULL)')
DB.execute('''CREATE TABLE IF NOT EXISTS canvas_sessions (
    id TEXT PRIMARY KEY, owner TEXT NOT NULL, title TEXT NOT NULL, body TEXT NOT NULL,
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL)''')
DB.execute('CREATE INDEX IF NOT EXISTS canvas_sessions_owner ON canvas_sessions(owner,updated_at)')
DB.commit()
HTTP = httpx.AsyncClient(timeout=httpx.Timeout(75, connect=10), trust_env=False)
USER_LOCKS = {}
PLAYBACK_LOCKS = {}
PLAYBACK_WAITERS = {}
PROMPT = (ROOT / 'prompt.txt').read_text()
PROMPTS = {'wildlife': PROMPT, 'maritime': (ROOT / 'prompt-maritime.txt').read_text()}
MCP = Client(StreamableHttpTransport('http://mcp:8000/mcp', headers={'Authorization': 'Bearer ' + os.environ['MCP_AUTH_TOKEN']}), timeout=20, cache=False)
OP = {'type': 'object', 'additionalProperties': False, 'properties': {
    'action': {'type': 'string', 'enum': ['add', 'update']},
    'target_id': {'type': ['string', 'null']}, 'source_id': {'type': ['string', 'null']},
    'title': {'type': 'string'}, 'sql': {'type': ['string', 'null']},
    'chart': {'type': 'string', 'enum': ['line', 'bar', 'heatmap', 'surface', 'scatter', 'table', 'map', 'gallery', 'tracks', 'density', 'seasons']},
    'x': {'type': 'string'}, 'y': {'type': 'string'}, 'z': {'type': 'string'},
    'color': {'type': 'string', 'enum': ['teal', 'blue', 'coral', 'violet', 'gold']},
}, 'required': ['action', 'target_id', 'source_id', 'title', 'sql', 'chart', 'x', 'y', 'z', 'color']}
TOOL = {'type': 'function', 'function': {'name': 'paint_canvas', 'description': 'Add or update data cards, or ask a clarification. Existing cards persist.', 'parameters': {
    'type': 'object', 'additionalProperties': False, 'properties': {
        'operations': {'type': 'array', 'items': OP, 'maxItems': 3}, 'question': {'type': ['string', 'null']}
    }, 'required': ['operations', 'question']}}}


@asynccontextmanager
async def lifespan(app):
    async with MCP:
        yield
    await HTTP.aclose()


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)


async def owner(request):
    auth = request.headers.get('Authorization', '')
    if not auth.startswith('Bearer ') or len(auth) > 10000:
        raise HTTPException(401, 'Please sign in')
    response = await HTTP.get('http://librechat:3080/api/user', headers={'Authorization': auth})
    if response.status_code != 200:
        raise HTTPException(401, 'Please sign in again')
    user = response.json()
    user_id = user.get('id') or user.get('_id')
    if not isinstance(user_id, str) or not user_id:
        raise HTTPException(401, 'Invalid user identity')
    return user_id


def remember_session(user, state):
    """Called inside the same transaction as the active workspace write."""
    title = next((m['text'] for m in state.get('messages', [])
                  if isinstance(m, dict) and m.get('role') == 'user' and isinstance(m.get('text'), str)), None)
    title = (title or (state.get('name') if isinstance(state.get('name'), str) else '') or 'New chat')[:100]
    timestamp = now()
    changed = DB.execute('''INSERT INTO canvas_sessions VALUES (?,?,?,?,?,?)
        ON CONFLICT(id) DO UPDATE SET title=excluded.title,body=excluded.body,updated_at=excluded.updated_at
        WHERE canvas_sessions.owner=excluded.owner''',
        (state['session_id'], user, title, json.dumps(state), timestamp, timestamp)).rowcount
    if changed != 1:
        raise ValueError('This session is not available')


def load(user):
    row = DB.execute('SELECT version,body FROM workspaces WHERE owner=?', (user,)).fetchone()
    if row is None:
        state = blank() | {'session_id': uuid.uuid4().hex}
        with DB:
            DB.execute('INSERT INTO workspaces VALUES (?,?,?)', (user, 0, json.dumps(state)))
            remember_session(user, state)
        return state
    state = json.loads(row[1]) | {'version': row[0]}
    migrated = 'dataset_id' not in state
    state['dataset_id'] = dataset_id(state.get('dataset_id'))
    # Adopt an existing canvas without discarding its chat, results, or undo history.
    known = DB.execute('SELECT 1 FROM canvas_sessions WHERE id=? AND owner=?',
                       (state.get('session_id') if isinstance(state.get('session_id'), str) else None, user)).fetchone()
    if not known or migrated:
        if not known:
            state['session_id'] = uuid.uuid4().hex
        with DB:
            DB.execute('UPDATE workspaces SET body=? WHERE owner=?', (json.dumps(state), user))
            remember_session(user, state)
    return state


def save(user, expected, state):
    current = load(user)
    state = state | {'dataset_id': dataset_id(state.get('dataset_id', current['dataset_id']))}
    if state['dataset_id'] != current['dataset_id']:
        raise ValueError('A chat belongs to one dataset. Start a new chat to change datasets.')
    validate_state(state)
    if state.get('session_id', current['session_id']) != current['session_id']:
        raise ValueError('A different chat is open. Reload before applying this change.')
    state = state | {'version': expected + 1, 'session_id': current['session_id']}
    with DB:
        changed = DB.execute('UPDATE workspaces SET version=?,body=? WHERE owner=? AND version=?',
                            (state['version'], json.dumps(state), user, expected)).rowcount
        if changed != 1:
            raise ValueError('The canvas changed in another tab. Reload before applying this change.')
        remember_session(user, state)
    return state


def activate_session(user, expected, state):
    """A monotonic workspace version fences off saves/results from the previous chat."""
    state = state | {'version': expected + 1, 'dataset_id': dataset_id(state.get('dataset_id'))}
    validate_state(state)
    with DB:
        changed = DB.execute('UPDATE workspaces SET version=?,body=? WHERE owner=? AND version=?',
                             (state['version'], json.dumps(state), user, expected)).rowcount
        if changed != 1:
            raise ValueError('The canvas changed in another tab. Reload before opening another chat.')
        remember_session(user, state)
    return state


class SessionRequest(BaseModel):
    version: int
    dataset_id: str | None = Field(default=None, pattern='^(wildlife|maritime)$')


@app.get('/canvas/api/sessions')
async def get_sessions(request: Request):
    user = await owner(request)
    state = load(user)
    rows = DB.execute("SELECT id,title,created_at,updated_at,json_extract(body,'$.dataset_id') FROM canvas_sessions WHERE owner=? ORDER BY updated_at DESC,id", (user,)).fetchall()
    return {'active_id': state['session_id'], 'dataset_id': state['dataset_id'], 'sessions': [
        dict(zip(['id','title','created_at','updated_at'], row[:4])) |
        {'dataset_id': dataset_id(row[4])} for row in rows]}


@app.post('/canvas/api/sessions')
async def new_session(task: SessionRequest, request: Request):
    user = await owner(request)
    current = load(user)
    try:
        return activate_session(user, task.version, blank(task.dataset_id or current['dataset_id']) |
                                {'session_id': uuid.uuid4().hex, 'name': 'New chat'})
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@app.post('/canvas/api/sessions/{session_id}/activate')
async def open_session(session_id: str, task: SessionRequest, request: Request):
    user = await owner(request)
    load(user)
    row = DB.execute('SELECT body FROM canvas_sessions WHERE id=? AND owner=?', (session_id, user)).fetchone()
    if not row:
        raise HTTPException(404, 'This chat is not available')
    try:
        saved = json.loads(row[0])
        if task.dataset_id and task.dataset_id != dataset_id(saved.get('dataset_id')):
            raise ValueError('That chat belongs to a different dataset.')
        return activate_session(user, task.version, saved)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@app.get('/canvas/api/workspace')
async def get_workspace(request: Request):
    return load(await owner(request))


@app.get('/canvas/api/datasets')
async def get_datasets(request: Request):
    state = load(await owner(request))
    return {'active_id': state['dataset_id'], 'datasets': catalogue()}


@app.put('/canvas/api/workspace')
async def put_workspace(request: Request):
    user = await owner(request)
    body = await request.body()
    if len(body) > 8_000_000:
        raise HTTPException(413, 'Canvas is too large')
    data = json.loads(body)
    try:
        return save(user, int(data['version']), data)
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(409, str(exc)) from exc


class PaintRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=2000)
    selected_id: str | None = None
    intent: str = Field(default='auto', pattern='^(auto|add|update)$')
    version: int
    request_id: str = Field(min_length=10, max_length=80)
    selection: dict | None = None
    dataset_id: str | None = Field(default=None, pattern='^(wildlife|maritime)$')


def event(kind, body):
    return 'event: ' + kind + '\ndata: ' + json.dumps(body) + '\n\n'


async def plan_canvas(state, task):
    active_dataset = dataset_id(state.get('dataset_id'))
    context = {'dataset_id': active_dataset, 'selected_id': task.selected_id, 'intent': task.intent, 'cards': [
        {k: c.get(k) for k in ['id', 'title', 'sql', 'chart', 'x', 'y', 'z', 'color']} |
        {'columns': c['data']['columns'], 'rows': len(c['data']['rows']), 'preview': c['data']['rows'][:2]}
        for c in state['cards']]}
    if active_dataset == 'maritime':
        metadata = next(item for item in catalogue() if item['id'] == active_dataset).get('metadata', {})
        if metadata:
            context['loaded_coverage'] = {key: metadata[key] for key in ('complete', 'ais', 'missing_dates') if key in metadata}
    if task.selection:
        scope = build_scope(task.selection, state, task.selected_id)
        context['marked_data'] = {k: scope[k] for k in ['description', 'selected', 'base_table', 'predicate']}
    body = {'model': 'deepseek/deepseek-v4.1-flash', 'stream': True,
            'messages': [{'role': 'system', 'content': PROMPTS[active_dataset]},
                         {'role': 'user', 'content': 'Canvas context:\n' + json.dumps(context) + '\n\nRequest: ' + task.prompt}],
            'tools': [TOOL], 'tool_choice': {'type': 'function', 'function': {'name': 'paint_canvas'}}}
    arguments = ''
    usage = {}
    started = time.perf_counter()
    async with HTTP.stream('POST', 'http://gateway:8080/v1/chat/completions', json=body,
                           headers={'X-Demo-Client': 'canvas'}) as response:
        if response.status_code != 200:
            await response.aread()
            raise ValueError('The model is temporarily unavailable or the demo spending limit has been reached.')
        async for line in response.aiter_lines():
            if not line.startswith('data:') or line[5:].strip() == '[DONE]':
                continue
            chunk = json.loads(line[5:].strip())
            if chunk.get('error'):
                raise ValueError('The model could not complete this change. Try a shorter request.')
            if chunk.get('usage'):
                usage = chunk['usage']
            for choice in chunk.get('choices', []):
                for call in choice.get('delta', {}).get('tool_calls', []):
                    if call.get('index', 0) != 0:
                        raise ValueError('Please change at most three views in one request.')
                    arguments += call.get('function', {}).get('arguments', '')
    return json.loads(arguments), (time.perf_counter() - started) * 1000, usage


async def query(sql, max_rows=1000):
    started = time.perf_counter()
    result = await MCP.call_tool('run_query', {'query': sql})
    texts = [c.text for c in result.content if getattr(c, 'type', None) == 'text']
    data = json.loads(texts[0]) if texts else result.data
    if not isinstance(data, dict) or 'rows' not in data or 'columns' not in data:
        raise ValueError('The query did not return a table. Try rephrasing the request.')
    if not isinstance(data['columns'], list) or not isinstance(data['rows'], list) or len(data['rows']) > max_rows:
        raise ValueError(f'Aggregate the query to at most {max_rows:,} rows.')
    if any(not isinstance(row, list) or len(row) != len(data['columns']) for row in data['rows']):
        raise ValueError('The query did not return a consistent table.')
    return {'data': data, 'query_ms': (time.perf_counter() - started) * 1000}


async def photo_metadata(result):
    """Attach licensed images to observation IDs; never ask the LLM for URLs."""
    cols, rows = result['data']['columns'], result['data']['rows']
    if 'observation_uuid' not in cols or not rows:
        return result
    at = cols.index('observation_uuid')
    ids = list(dict.fromkeys(str(r[at]) for r in rows))[:1000]
    if not all(re.fullmatch(r'[0-9a-fA-F-]{36}', i) for i in ids):
        raise ValueError('Invalid observation identifier')
    selected = ','.join("'" + str(uuid.UUID(i)) + "'" for i in ids)
    photos = await query(f"""SELECT toString(p.observation_uuid) AS observation_uuid,
 p.photo.1 AS photo_id,p.photo.2 AS extension,p.photo.3 AS license,
 if(empty(o.name),o.login,o.name) AS credit
FROM (SELECT observation_uuid,argMin(tuple(photo_id,extension,license,observer_id),tuple(position,photo_id)) AS photo
 FROM exploration.wildlife_photos WHERE observation_uuid IN ({selected}) GROUP BY observation_uuid) p
LEFT ANY JOIN exploration.wildlife_observers o ON p.photo.4=o.observer_id""")
    known = {str(r[0]): r[1:] for r in photos['data']['rows']}
    extra = ['photo_id', 'photo_url', 'photo_license', 'photo_credit', 'photo_page']
    if any(c in cols for c in extra):
        raise ValueError('Photo metadata is added automatically; select observation_uuid instead')
    enriched = []
    for row in rows:
        photo = known.get(str(row[at]))
        values = [None] * len(extra)
        if photo:
            pid, extension, license, credit = photo
            if re.fullmatch(r'\d+', str(pid)) and extension in {'jpg', 'jpeg', 'png', 'webp'} and license in {'CC0', 'CC-BY'}:
                values = [pid, f'https://inaturalist-open-data.s3.amazonaws.com/photos/{pid}/medium.{extension}', license, credit, f'https://www.inaturalist.org/photos/{pid}']
        enriched.append(row + values)
    result['data'] = {'columns': cols + extra, 'rows': enriched}
    result['query_ms'] += photos['query_ms']
    return result


@app.post('/canvas/api/paint')
async def paint(task: PaintRequest, request: Request):
    user = await owner(request)
    lock = USER_LOCKS.setdefault(user, asyncio.Lock())
    if lock.locked():
        raise HTTPException(409, 'Another canvas change is still running')
    state = load(user)
    if state['version'] != task.version:
        raise HTTPException(409, 'The canvas changed in another tab. Reload to continue.')
    if task.dataset_id and task.dataset_id != state['dataset_id']:
        raise HTTPException(409, 'The dataset changed. Reload this chat before asking another question.')
    if task.selected_id and task.selected_id not in {c['id'] for c in state['cards']}:
        raise HTTPException(400, 'The selected card no longer exists')
    task.intent = resolve_intent(task.prompt, task.selected_id, task.intent)
    if task.intent == 'update' and not task.selected_id:
        raise HTTPException(400, 'Select a view to change first')
    try:
        scope = build_scope(task.selection, state, task.selected_id)
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        raise HTTPException(400, str(exc)) from exc

    async def stream():
        async with lock:
            start = time.perf_counter()
            record = {'request_id': task.request_id, 'started_at': now(), 'selected_id': task.selected_id,
                      'prompt': task.prompt, 'status': 'started', 'base_version': task.version,
                      'dataset_id': state['dataset_id']}
            try:
                if load(user)['version'] != task.version:
                    raise ValueError('The canvas changed while this request was waiting. Reload to continue.')
                yield event('status', {'phase': 'thinking', 'text': 'Updating your view…' if task.intent == 'update' else 'Shaping a new view…', 'intent': task.intent})
                plan, model_ms, usage = await plan_canvas(state, task)
                if load(user)['version'] != task.version:
                    raise ValueError('The chat changed while the model was working. The previous result was not applied.')
                intent = task.intent
                raw_operations = plan.get('operations')
                if isinstance(raw_operations, list) and all(isinstance(op, dict) for op in raw_operations):
                    if intent == 'update' and task.selected_id and raw_operations:
                        if len(raw_operations) > 1:
                            raise ValueError('Refine one selected view at a time, or use Add alongside for new views.')
                        operation = raw_operations[0]
                        if operation.get('action') in {'add', 'update'}:
                            operation['action'] = 'update'
                            operation['target_id'] = task.selected_id
                    elif intent == 'add':
                        for operation in raw_operations:
                            if operation.get('action') == 'update':
                                operation['source_id'] = operation.get('target_id') or task.selected_id
                                operation['target_id'] = None
                                operation['action'] = 'add'
                operations = validate_plan(plan, state, task.selected_id)
                record.update(plan=plan, model_ms=model_ms, usage=usage)
                if scope:
                    record['selection'] = scope
                if not operations:
                    state['messages'] = (state['messages'] + [{'role': 'user', 'text': task.prompt},
                        {'role': 'assistant', 'text': plan['question'], 'changes': []}])[-100:]
                    state_saved = save(user, task.version, state)
                    yield event('result', {'workspace': state_saved, 'changes': [], 'question': plan['question'],
                                           'model_ms': model_ms, 'query_ms': 0})
                else:
                    results = []
                    for op in operations:
                        if op.get('sql'):
                            validate_sql(op['sql'], state['dataset_id'])
                            op['sql'] = apply_scope(op['sql'], scope)
                            validate_sql(op['sql'], state['dataset_id'])
                            if scope:
                                # Bin keys can be centers, not boundaries. Describe the verified mark
                                # instead of presenting a model-inferred geographic range as fact.
                                name = {'gallery': 'Photos', 'map': 'Map', 'table': 'Table', 'line': 'Trend',
                                        'bar': 'Breakdown', 'heatmap': 'Heatmap', 'surface': '3D view', 'scatter': 'Scatter',
                                        'tracks': 'Tracks', 'density': 'Density', 'seasons': 'Seasons'}[op['chart']]
                                op['title'] = name + ' · ' + scope['description'].lower()
                                if op['action'] == 'add':
                                    op['source_id'] = scope['card_id']
                            yield event('status', {'phase': 'querying', 'text': 'Querying ClickHouse…', 'title': op['title']})
                            result = await query(op['sql'])
                            if op['chart'] == 'gallery':
                                result = await photo_metadata(result)
                            results.append(result)
                        else:
                            results.append(None)
                    updated, changes = apply_operations(state, operations, results, task.prompt)
                    total_ms = (time.perf_counter() - start) * 1000
                    summary = ' · '.join(('Added ' if c['action'] == 'add' else 'Updated ') + c['title'] for c in changes)
                    if scope and scope['description'].lower() not in summary.lower():
                        summary += ' · ' + scope['description']
                    updated['messages'] = (updated['messages'] + [{'role': 'user', 'text': task.prompt, 'target_id': task.selected_id},
                        {'role': 'assistant', 'text': summary, 'changes': changes, 'elapsed_ms': total_ms}])[-100:]
                    state_saved = save(user, task.version, updated)
                    record.update(changes=changes, queries=results, total_ms=total_ms)
                    yield event('result', {'workspace': state_saved, 'changes': changes, 'elapsed_ms': total_ms,
                                          'model_ms': model_ms, 'query_ms': sum(r['query_ms'] for r in results if r),
                                          'inference_cost_usd': usage.get('cost')})
                record['status'] = 'succeeded'
            except asyncio.CancelledError:
                record['status'] = 'cancelled'
                raise
            except Exception as exc:
                record.update(status='failed', error=str(exc))
                yield event('error', {'message': 'The change could not be applied. Your canvas is unchanged.',
                                      'detail': str(exc)[:400]})
            finally:
                record['ended_at'] = now()
                with Path('/evidence/canvas-calls.jsonl').open('a') as f:
                    f.write(json.dumps(record) + '\n')
    return StreamingResponse(stream(), media_type='text/event-stream',
                             headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


class PlaybackRequest(BaseModel):
    card_id: str = Field(min_length=1, max_length=64)
    revision: str = Field(min_length=1, max_length=64)
    session_id: str = Field(min_length=1, max_length=64)
    version: int


def playback_card(user, task, expected_card=None):
    state = load(user)
    if state['session_id'] != task.session_id or (expected_card is None and state['version'] != task.version):
        raise HTTPException(409, 'The chat changed. Reload the track view before replaying it.')
    if state['dataset_id'] != 'maritime':
        raise HTTPException(400, 'Vessel and storm playback is available in Ships & storms.')
    card = next((card for card in state['cards'] if card['id'] == task.card_id), None)
    if not card:
        raise HTTPException(404, 'This track view is not available in the current chat.')
    if card.get('revision') != task.revision:
        raise HTTPException(409, 'The track view changed. Reload it before replaying.')
    if expected_card is not None and (card.get('sql') != expected_card.get('sql') or card.get('data') != expected_card.get('data')):
        raise HTTPException(409, 'The track selection changed. Reload it before replaying.')
    return card


@asynccontextmanager
async def playback_slot(user):
    """Serialize loaded cards without turning a read queue into a save conflict."""
    lock = PLAYBACK_LOCKS.setdefault(user, asyncio.Lock())
    if PLAYBACK_WAITERS.get(user, 0) >= 12:
        raise HTTPException(429, 'Several track views are still loading. Try again shortly.', headers={'Retry-After': '2'})
    PLAYBACK_WAITERS[user] = PLAYBACK_WAITERS.get(user, 0) + 1
    try:
        try:
            await asyncio.wait_for(lock.acquire(), timeout=30)
        except TimeoutError as exc:
            raise HTTPException(429, 'Track views are still loading. Try this replay again shortly.', headers={'Retry-After': '2'}) from exc
        try:
            yield
        finally:
            lock.release()
    finally:
        PLAYBACK_WAITERS[user] -= 1
        if not PLAYBACK_WAITERS[user]:
            PLAYBACK_WAITERS.pop(user, None)


@app.post('/canvas/api/playback')
async def playback(task: PlaybackRequest, request: Request):
    user = await owner(request)
    card = playback_card(user, task)
    empty = {'columns': [], 'rows': [], 'storm': {'columns': [], 'rows': []}, 'range': None,
             'truncated': False, 'sampling': {'interval_seconds': 0, 'max_vessels': 24, 'max_points': MAX_PLAYBACK_POINTS}}
    try:
        scope = playback_scope(card)
        if not scope['ids']:
            return empty
        async with playback_slot(user), asyncio.timeout(35):
            playback_card(user, task, card)
            extent = await query(range_sql(scope))
            window = playback_range(extent['data'])
            if not window:
                playback_card(user, task, card)
                return empty
            rows, columns = [], []
            elapsed = extent['query_ms']
            if scope['table'] == 'ais_positions':
                # The ClickHouse reader independently caps a result at 10,000
                # rows / 1 MB. Two batches of 12 ships keep those limits intact.
                for offset in range(0, len(scope['ids']), 12):
                    result = await query(vessel_sql(scope, window, scope['ids'][offset:offset+12]), max_rows=6000)
                    if columns and columns != result['data']['columns']:
                        raise ValueError('The playback query returned inconsistent columns.')
                    columns = result['data']['columns']
                    rows.extend(result['data']['rows'])
                    elapsed += result['query_ms']
                    if len(rows) > MAX_PLAYBACK_POINTS:
                        raise ValueError('Choose a smaller set of ships for playback.')
            storm = await query(storm_sql(scope, window), max_rows=MAX_STORM_POINTS)
            elapsed += storm['query_ms']
            # Fence late responses after a session switch, same as model results.
            playback_card(user, task, card)
            names = []
            if 'storm_name' in storm['data']['columns']:
                at = storm['data']['columns'].index('storm_name')
                names = sorted({str(row[at]) for row in storm['data']['rows']})
            response = {
                'columns': columns, 'rows': rows, 'storm': storm['data'],
                'range': {k: window[k] for k in ('start', 'end')}, 'bounds': window['bounds'],
                'truncated': False, 'dataset_id': 'maritime', 'query_ms': elapsed,
                'sampling': {'interval_seconds': window['interval_seconds'] if rows else 0,
                             'max_vessels': 24, 'max_points': MAX_PLAYBACK_POINTS,
                             'selected_tracks': len(scope['ids']), 'source_positions': window['positions'],
                             'returned_positions': len(rows), 'method': 'last observed position per time bucket',
                             'gap_policy': 'interpolate short gaps only; disconnect long gaps'},
                'storm_overlay': {'names': names, 'scope': 'selected storm tracks' if scope['table'] == 'storm_tracks'
                                  else 'nearby storm tracks within the vessel time range and an 8-degree margin'},
                'source': 'NOAA MarineCadastre AIS and NOAA IBTrACS · UTC',
            }
            if len(json.dumps(response).encode()) > MAX_PLAYBACK_BYTES:
                raise ValueError('Choose a smaller track selection for playback.')
            return response
    except TimeoutError as exc:
        raise HTTPException(408, 'This replay took too long to load. Choose fewer ships or a shorter time window.') from exc
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get('/canvas/health')
async def health():
    return {'ok': True, 'mcp_connected': MCP.is_connected()}


@app.get('/canvas/api/dataset')
async def dataset(request: Request):
    await owner(request)
    path = Path('/dataset/dataset.json')
    return json.loads(path.read_text()) if path.exists() else {'loading': True}


@app.get('/canvas/vendor/{filename:path}')
async def vendor(filename: str):
    path = (ROOT / 'static/vendor' / filename).resolve()
    if not path.is_relative_to((ROOT / 'static/vendor').resolve()) or not path.is_file():
        raise HTTPException(404)
    return FileResponse(path, headers={'Cache-Control': 'public,max-age=86400'})


@app.get('/canvas/')
async def index():
    return FileResponse(ROOT / 'static/index.html', headers={'Cache-Control': 'no-cache'})


@app.get('/canvas/{filename}')
async def asset(filename: str):
    if filename not in {'app.js', 'style.css', 'surface.js', 'media.js', 'scifi.css', 'marks.js', 'motion.js', 'gif-worker.js', 'camera.js',
                        'explore-viz.js', 'explore-viz.css', 'datasets.css'}:
        raise HTTPException(404)
    return FileResponse(ROOT / 'static' / filename, headers={'Cache-Control': 'no-cache'})
