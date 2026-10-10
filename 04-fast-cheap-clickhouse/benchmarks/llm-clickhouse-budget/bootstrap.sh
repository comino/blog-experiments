#!/usr/bin/env bash
set -euo pipefail
ROOT=/opt/ch-budget
mkdir -p "$ROOT/runtime" "$ROOT/bin"
if ! id chbench >/dev/null 2>&1; then
  useradd --system --home-dir "$ROOT/runtime" --shell /usr/sbin/nologin chbench
fi
python3 - <<'PY'
import hashlib, json, shutil, tarfile, urllib.request
from pathlib import Path
root=Path('/opt/ch-budget')
manifest=json.loads((root/'code/clickhouse-binary.json').read_text())
archive=root/'runtime/clickhouse.tgz'
binary=root/'bin/clickhouse'
if not binary.exists():
    urllib.request.urlretrieve(manifest['source'],archive)
    assert hashlib.sha512(archive.read_bytes()).hexdigest()==manifest['archive_sha512']
    with tarfile.open(archive) as t:
        member=next(x for x in t.getmembers() if x.name.endswith('/usr/bin/clickhouse'))
        with t.extractfile(member) as src, binary.open('wb') as dst:
            shutil.copyfileobj(src,dst)
    binary.chmod(0o755)
print('ClickHouse binary verified')
PY
chown -R chbench:chbench "$ROOT/runtime"
cat > /etc/systemd/system/ch-budget.service <<'UNIT'
[Unit]
Description=Temporary ClickHouse SQL model benchmark
After=network.target
[Service]
User=chbench
Group=chbench
ExecStart=/opt/ch-budget/bin/clickhouse server --config-file=/opt/ch-budget/code/server-config.xml
WorkingDirectory=/opt/ch-budget/runtime
LimitNOFILE=65536
Restart=no
[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl start ch-budget
python3 - <<'PY'
import time,urllib.request
for attempt in range(30):
    try:
        response=urllib.request.urlopen('http://127.0.0.1:18123/ping',timeout=1).read()
        print('ClickHouse:',response.decode().strip())
        break
    except Exception:
        if attempt==29: raise
        time.sleep(1)
PY
