#!/usr/bin/env python3
"""Generate local demo credentials; no credentials are printed or recorded as evidence."""
import json
import os
from pathlib import Path
import secrets

BASE = Path(__file__).resolve().parent
RUNTIME = BASE / 'runtime'
os.umask(0o077)
RUNTIME.mkdir(exist_ok=True)
assert not (RUNTIME / 'credentials.json').exists(), 'Credentials already exist; refusing to overwrite'
loader = secrets.token_urlsafe(32)
reader = secrets.token_urlsafe(32)
mcp = secrets.token_urlsafe(32)
login = secrets.token_urlsafe(24)


def env(name, values):
    (RUNTIME / name).write_text(''.join(f'{k}={v}\n' for k, v in values.items()))


env('clickhouse.env', {'CLICKHOUSE_DB': 'exploration', 'CLICKHOUSE_USER': 'loader', 'CLICKHOUSE_PASSWORD': loader, 'CLICKHOUSE_DEFAULT_ACCESS_MANAGEMENT': '1'})
env('mcp.env', {
    'CLICKHOUSE_HOST': 'clickhouse', 'CLICKHOUSE_PORT': '8123', 'CLICKHOUSE_SECURE': 'false',
    'CLICKHOUSE_USER': 'explorer', 'CLICKHOUSE_PASSWORD': reader, 'CLICKHOUSE_DATABASE': 'exploration',
    'CLICKHOUSE_ALLOW_WRITE_ACCESS': 'false', 'CLICKHOUSE_MCP_SERVER_TRANSPORT': 'http',
    'CLICKHOUSE_MCP_BIND_HOST': '0.0.0.0', 'CLICKHOUSE_MCP_BIND_PORT': '8000',
    'CLICKHOUSE_MCP_AUTH_TOKEN': mcp, 'CLICKHOUSE_MCP_ALLOWED_HOSTS': 'mcp:8000,localhost:8000',
    'CLICKHOUSE_MCP_QUERY_TIMEOUT': '10', 'CLICKHOUSE_MCP_MAX_WORKERS': '2',
})
env('librechat.env', {
    'HOST': '0.0.0.0', 'PORT': '3080', 'NODE_ENV': 'production',
    'MONGO_URI': 'mongodb://mongodb:27017/LibreChat', 'SEARCH': 'false', 'ENDPOINTS': 'custom,agents',
    'DOMAIN_CLIENT': 'http://localhost:3080', 'DOMAIN_SERVER': 'http://localhost:3080',
    'ALLOW_EMAIL_LOGIN': 'true', 'ALLOW_REGISTRATION': 'true', 'ALLOW_UNVERIFIED_EMAIL_LOGIN': 'true',
    'ALLOW_SOCIAL_LOGIN': 'false', 'ALLOW_PASSWORD_RESET': 'false',
    'JWT_SECRET': secrets.token_hex(48), 'JWT_REFRESH_SECRET': secrets.token_hex(48),
    'CREDS_KEY': secrets.token_hex(32), 'CREDS_IV': secrets.token_hex(16),
    'MCP_AUTH_TOKEN': mcp, 'NO_PROXY': '*', 'DEBUG_LOGGING': 'false', 'DEBUG_CONSOLE': 'false',
})
(RUNTIME / 'credentials.json').write_text(json.dumps({'loader': loader, 'reader': reader, 'login': login, 'email': 'explorer@example.invalid'}, indent=2) + '\n')
(RUNTIME / 'init.sql').write_text(f"""
CREATE USER IF NOT EXISTS explorer IDENTIFIED WITH sha256_password BY '{reader}'
SETTINGS readonly=1, max_threads=4, max_memory_usage=6000000000, max_execution_time=10,
max_result_rows=10000, max_result_bytes=1000000, result_overflow_mode='throw', use_query_cache=0;
GRANT SELECT ON exploration.* TO explorer;
""")
print('Created private runtime credentials.')
