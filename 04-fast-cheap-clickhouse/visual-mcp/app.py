"""Add a reusable visual tool to the pinned official ClickHouse MCP server."""
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time
from typing import Literal
import uuid

from fastmcp.tools import ToolResult
from mcp.types import EmbeddedResource, TextContent, TextResourceContents
from mcp_clickhouse.mcp_server import mcp, run_query_async
from mcp_clickhouse.main import main

ASSET_BASE_URL = os.environ.get('CHART_ASSET_BASE_URL', 'http://localhost:3081').rstrip('/')
TEMPLATE = Path(__file__).with_name('chart.html').read_text().replace('http://localhost:3081', ASSET_BASE_URL)
LOG = Path('/evidence/visual-tools.jsonl')


@mcp.tool
async def explore(
    query: str,
    title: str,
    chart: Literal['line', 'bar', 'heatmap', 'surface', 'table'],
    x: str,
    y: str,
    z: str = '',
) -> ToolResult:
    """Run read-only ClickHouse SQL and display an interactive chart and table.

    Use this tool for every visual answer. Supply one aggregate SELECT, a concise
    title, and result-column names for axes. line/bar: x is category and y is value.
    heatmap/surface: x and y are categories, z is value. At most 1000 result rows.
    The server renders the chart; never write HTML/JavaScript or repeat the data.
    After the result, emit its UI Resource Marker first, then at most one sentence.
    Query exploration.trips: source_month Date (file month), pickup_datetime and
    dropoff_datetime (NYC local wall clock), pickup_borough, pickup_zone,
    dropoff_borough, dropoff_zone, trip_distance (miles), fare_amount, tip_amount,
    total_amount (USD), payment_type (1=card,2=cash). 128202548 raw yellow taxi trips,
    Jan 2023-Dec 2025. source_month is FILE PROVENANCE, not pickup month. For a
    calendar year or event date range filter pickup_datetime, NOT source_month.
    For 2025: pickup_datetime >= '2025-01-01' AND pickup_datetime < '2026-01-01'.
    Source files contain some out-of-month timestamps. Use source_month only when
    explicitly asked for source-file month. Recorded tips only cover
    card payments. For journey duration exclude <=0 or >180 minutes and state it.
    """
    call_id = uuid.uuid4().hex
    record = dict(call_id=call_id, started_at=datetime.now(timezone.utc).isoformat(),
                  query=query, title=title, chart=chart, x=x, y=y, z=z)
    start = time.perf_counter()
    try:
        raw = await run_query_async(query)
        record['query_tool_ms'] = (time.perf_counter() - start) * 1000
        data = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(data, dict) or 'columns' not in data or 'rows' not in data:
            raise ValueError('Query did not return tabular data: ' + str(data)[:800])
        for axis in ([x, y, z] if chart in ('heatmap', 'surface') else [x, y]):
            if chart != 'table' and axis not in data['columns']:
                raise ValueError(f'Axis {axis!r} is not a returned column')
        if len(data['rows']) > 1000:
            raise ValueError('Aggregate the result to at most 1000 rows')
        record['data'] = data
        payload = {**data, 'id': call_id, 'title': title, 'chart': chart,
                   'x': x, 'y': y, 'z': z, 'sql': query,
                   'query_ms': round(record['query_tool_ms'])}
        # Escape HTML delimiters even though this is a non-executable JSON script.
        encoded = json.dumps(payload, ensure_ascii=False).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
        html = TEMPLATE.replace('__PAYLOAD__', encoded)
        statistics = {}
        for i, column in enumerate(data['columns']):
            values = [row[i] for row in data['rows'] if isinstance(row[i], (int, float)) and math.isfinite(row[i])]
            if values:
                statistics[column] = {'min': min(values), 'max': max(values), 'sum': sum(values)}
        summary = {'row_count': len(data['rows']), 'columns': data['columns'],
                   'preview': data['rows'][:3], 'statistics_over_all_returned_rows': statistics, 'title': title,
                   'instruction': 'Emit the UI Resource Marker first. Do not infer a range or trend from the three-row preview. State only facts supported by the complete-result statistics, or simply identify the chart.'}
        return ToolResult(content=[
            TextContent(type='text', text=json.dumps(summary)),
            EmbeddedResource(type='resource', resource=TextResourceContents(
                uri=f'ui://taxi-explorer/{call_id}', mimeType='text/html', text=html)),
        ])
    except Exception as exc:
        record['error'] = str(exc)
        raise
    finally:
        record['total_ms'] = (time.perf_counter() - start) * 1000
        record['ended_at'] = datetime.now(timezone.utc).isoformat()
        with LOG.open('a') as f:
            f.write(json.dumps(record) + '\n')


if __name__ == '__main__':
    main()
