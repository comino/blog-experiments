"""Canvas transactions. Results are committed together after every query succeeds."""
from copy import deepcopy
from datetime import datetime, timezone
import json
import re
import uuid

from datasets import dataset_id, DATASETS

CHARTS = {'line', 'bar', 'heatmap', 'surface', 'table', 'scatter', 'map', 'gallery', 'tracks', 'density', 'seasons'}
COLORS = {'teal', 'blue', 'coral', 'violet', 'gold'}
MAX_CARDS = 12


def resolve_intent(prompt, selected_id, requested='auto'):
    """Protect existing views: only an explicit change request updates a selection."""
    if requested in {'add', 'update'}:
        return requested
    if not selected_id:
        return 'add'
    text = re.sub(r'\s+', ' ', prompt.lower().replace('’', "'")).strip()
    text = re.sub(r"^(?:(?:please|(?:can|could|would) (?:you|we)|let's|i (?:want|would like) (?:you )?to|i'd like to)\s+)+", '', text)
    # Explicit copies/additions override an edit. Merely referring to an existing
    # comparison must not turn "Change this comparison chart" into an addition.
    if re.search(r'\b(?:alongside|(?:new|another|separate|additional) (?:view|card|chart|map|table|gallery|plot))\b', text):
        return 'add'
    if re.match(r'^(?:change|update|edit|modify|replace|switch|convert|rename|recolou?r|restyle|adjust|refine|filter|restrict|limit|exclude|include|remove|sort|zoom|only|just)\b', text):
        return 'update'
    if re.match(r'^(?:make|turn)\s+(?:it|this|that|these|those|the\s+(?:selected|current|same))\b|^make\s+(?:bars|3d)\b', text):
        return 'update'
    if re.match(r'^(?:show|display|render)\s+(?:(?:it|this|that|the\s+(?:selected|current|same)\s+(?:view|chart|map))\s+)?as\b', text):
        return 'update'
    if re.match(r'^(?:show|keep)\s+(?:only|just)\b', text):
        return 'update'
    if re.match(r'^now\s+(?:(?:for|in|use)\s+)?\d{4}\b|^(?:break|drill)\s+(?:it|this|that)\s+(?:down|into)\b|^use\s+(?:teal|blue|coral|violet|gold)\b|^add\s+(?:a\s+)?(?:filter|legend|labels?|title|trendline|annotation)\b', text):
        return 'update'
    return 'add'


def now():
    return datetime.now(timezone.utc).isoformat()


def blank(dataset='wildlife'):
    dataset = dataset_id(dataset)
    return {'cards': [], 'messages': [], 'undo': [], 'redo': [], 'version': 0,
            'name': 'Life, all around us' if dataset == 'wildlife' else 'Ships & storms', 'dataset_id': dataset}


def validate_state(state):
    if not isinstance(state, dict) or not isinstance(state.get('cards'), list):
        raise ValueError('Invalid canvas')
    if len(state['cards']) > MAX_CARDS or len(json.dumps(state).encode()) > 8_000_000:
        raise ValueError('This canvas is full. Remove a card or some old revisions first.')
    active = dataset_id(state.get('dataset_id'))
    ids = [c.get('id') for c in state['cards']]
    if any(not isinstance(i, str) or not re.fullmatch(r'[a-zA-Z0-9_-]{1,64}', i) for i in ids) or len(set(ids)) != len(ids):
        raise ValueError('Invalid card identifiers')
    for c in state['cards']:
        if dataset_id(c.get('dataset_id', active)) != active:
            raise ValueError('Views belong to one dataset. Start a separate chat for another dataset.')
        validate_view(c)
        if not isinstance(c.get('title'), str) or len(c['title']) > 240:
            raise ValueError('Invalid card title')
        if len(c.get('history', [])) > 12:
            raise ValueError('Too many revisions')
        for key, low, high in [('left', 0, 6000), ('top', 0, 6000), ('width', 300, 1600), ('height', 250, 1200)]:
            if not isinstance(c.get(key), (int, float)) or not low <= c[key] <= high:
                raise ValueError('Invalid card position or size')
    if len(state.get('messages', [])) > 100 or len(state.get('undo', [])) > 25 or len(state.get('redo', [])) > 25:
        raise ValueError('Canvas history limit reached')
    return state


def validate_view(card):
    if card.get('chart') not in CHARTS or card.get('color', 'teal') not in COLORS:
        raise ValueError('Unsupported chart style')
    data = card.get('data', {})
    cols, rows = data.get('columns'), data.get('rows')
    if not isinstance(cols, list) or not 1 <= len(cols) <= 50 or not all(isinstance(c, str) for c in cols):
        raise ValueError('A query must return named columns')
    if not isinstance(rows, list) or len(rows) > 1000 or any(not isinstance(r, list) or len(r) != len(cols) for r in rows):
        raise ValueError('Aggregate the query to at most 1,000 rows')
    axes = ['x', 'y', 'z'] if card['chart'] in {'surface', 'heatmap', 'density'} or (card['chart'] == 'seasons' and card.get('z')) else ['x', 'y']
    if card['chart'] == 'gallery' and 'observation_uuid' not in cols:
        raise ValueError('Photo views need observation_uuid from the wildlife data')
    if card['chart'] == 'tracks' and not {'vessel_id', 'storm_id'}.intersection(cols):
        raise ValueError('Track views need vessel_id or storm_id from the maritime data')
    if card['chart'] not in {'table', 'gallery'} and any(card.get(axis) not in cols for axis in axes):
        raise ValueError('Chart axes must name returned columns')


def snapshot(card):
    return {k: deepcopy(v) for k, v in card.items() if k != 'history'}


def validate_plan(plan, state, selected_id):
    operations = plan.get('operations')
    if not isinstance(operations, list) or len(operations) > 3:
        raise ValueError('Change at most three cards in one request')
    if not operations:
        question = plan.get('question')
        if not isinstance(question, str) or not question.strip():
            raise ValueError('The model did not return a canvas change')
        return []
    known = {c['id'] for c in state['cards']}
    active = dataset_id(state.get('dataset_id'))
    touched = set()
    for op in operations:
        if op.get('action') not in {'add', 'update'}:
            raise ValueError('Unsupported canvas operation')
        if op.get('source_id') and op['source_id'] not in known:
            raise ValueError('The source card no longer exists')
        if op['action'] == 'update':
            target = op.get('target_id') or selected_id
            if target not in known or target in touched:
                raise ValueError('Choose an existing card to update')
            touched.add(target)
            op['target_id'] = target
        if not isinstance(op.get('title'), str) or not op['title'].strip() or len(op['title']) > 240:
            raise ValueError('A chart needs a short title')
        if op.get('chart') not in CHARTS or op.get('color', 'teal') not in COLORS:
            raise ValueError('Unsupported visualization')
        if op['chart'] not in DATASETS[active]['charts']:
            raise ValueError('That visualization is not available for this dataset.')
        sql = op.get('sql')
        if sql is not None and (not isinstance(sql, str) or not 1 <= len(sql.strip()) <= 12000):
            raise ValueError('Invalid SQL')
        if not sql:
            source = op.get('source_id') or op.get('target_id')
            if source not in known:
                raise ValueError('A new card needs a query or a source card')
            op['source_id'] = source
    if len(known) + sum(o['action'] == 'add' for o in operations) > MAX_CARDS:
        raise ValueError('Keep at most 12 cards on one canvas')
    return operations


def apply_operations(state, operations, results, prompt):
    out = deepcopy(state)
    before = deepcopy(out['cards'])
    changes = []
    for op, result in zip(operations, results, strict=True):
        old = next((c for c in out['cards'] if c['id'] == op.get('target_id')), None)
        if op['action'] == 'update':
            card = deepcopy(old)
            card['history'] = (card.get('history', []) + [snapshot(old)])[-12:]
        else:
            n = len(out['cards'])
            card = {'id': uuid.uuid4().hex, 'left': 38 + (n % 2) * 580, 'top': 130 + (n // 2) * 465,
                    'width': 548, 'height': 425, 'history': [], 'created_at': now(),
                    'derived_from': op.get('source_id') or None}
        source = next((c for c in state['cards'] if c['id'] == op.get('source_id')), old)
        data = result['data'] if result else deepcopy(source['data'])
        card.update({k: op.get(k, '') for k in ['title', 'chart', 'x', 'y', 'z']})
        card.update(color=op.get('color', 'teal'), data=data, sql=op.get('sql') or source['sql'],
                    revision=uuid.uuid4().hex, updated_at=now(), change=prompt,
                    query_ms=result['query_ms'] if result else 0, reused_data=result is None,
                    dataset_id=dataset_id(state.get('dataset_id')))
        validate_view(card)
        if old and op['action'] == 'update':
            out['cards'][next(i for i, c in enumerate(out['cards']) if c['id'] == old['id'])] = card
        else:
            out['cards'].append(card)
        changes.append({'id': card['id'], 'action': op['action'], 'title': card['title'],
                        'rows': len(data['rows']), 'reused_data': result is None})
    out['undo'] = (out.get('undo', []) + [{'label': prompt, 'cards': before}])[-25:]
    out['redo'] = []
    validate_state(out)
    return out, changes
