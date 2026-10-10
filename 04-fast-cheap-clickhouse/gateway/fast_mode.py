"""Render successful visual tool results without a second inference request.

This is deterministic UI plumbing, not a model response or a cached SQL result.
Only the latest, fully completed visual-tool round may take this path. Errors,
schema lookups, and subsequent user questions still go to the model.
"""
import json
import re

FAST_MODEL = 'taxi-explorer-fast'
VISUAL_TOOL = 'explore_mcp_clickhouse'
NOTICE = re.compile(
    r'^System notice: this turn has about \d+ more tool-calling rounds left before it is cut off\. '
    r'Stop broadening the search and start converging: use what you already have, make only calls '
    r'you cannot finish without, and leave room to write the final answer\.$'
)
RESOURCE = re.compile(
    r'\n\nUI Resource ID: ([a-f0-9]{10})\n'
    r'UI Resource Marker: \\ui\{\1\}\n'
    r'Resource URI: ui://taxi-explorer/([a-f0-9]{32})\n'
    r'Resource MIME Type: text/html(?:\n|$)'
)


def is_framework_notice(message):
    content = message.get('content')
    return (
        message.get('role') == 'user'
        and isinstance(content, list) and len(content) == 1
        and isinstance(content[0], dict)
        and content[0].get('type') == 'text'
        and content[0].get('cache_control') == {'type': 'ephemeral', 'ttl': '1h'}
        and bool(NOTICE.fullmatch(content[0].get('text', '')))
    )


def ready_resources(messages):
    """Return validated UI markers from this round, or None for normal inference."""
    if not isinstance(messages, list) or any(not isinstance(m, dict) for m in messages):
        return None
    assistant_index = next((i for i in range(len(messages) - 1, -1, -1)
                            if messages[i].get('role') == 'assistant'), None)
    if assistant_index is None:
        return None
    calls = messages[assistant_index].get('tool_calls') or []
    if not isinstance(calls, list) or not calls or any(
        not isinstance(c, dict) or not isinstance(c.get('function'), dict)
        or c['function'].get('name') != VISUAL_TOOL for c in calls
    ):
        return None
    call_ids = [c.get('id') for c in calls]
    if any(not isinstance(i, str) or not i for i in call_ids) or len(set(call_ids)) != len(call_ids):
        return None
    found = {}
    for message in messages[assistant_index + 1:]:
        if is_framework_notice(message):
            continue
        if message.get('role') != 'tool' or message.get('name') != VISUAL_TOOL:
            return None
        call_id = message.get('tool_call_id')
        if call_id not in call_ids or call_id in found:
            return None
        content = message.get('content')
        if not isinstance(content, str):
            return None
        try:
            summary, end = json.JSONDecoder().raw_decode(content)
        except (json.JSONDecodeError, ValueError):
            return None
        if not isinstance(summary, dict) or summary.get('error'):
            return None
        if type(summary.get('row_count')) is not int or not 0 <= summary['row_count'] <= 1000:
            return None
        if not isinstance(summary.get('columns'), list) or not isinstance(summary.get('preview'), list):
            return None
        match = RESOURCE.match(content, end)
        if match is None:
            return None
        found[call_id] = {'marker': '\\ui{' + match[1] + '}', 'resource_id': match[2],
                          'row_count': summary['row_count']}
    if set(found) != set(call_ids):
        return None
    return [found[call_id] for call_id in call_ids]
