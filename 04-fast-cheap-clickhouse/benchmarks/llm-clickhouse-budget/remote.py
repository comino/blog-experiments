#!/usr/bin/env python3
"""Operate only the temporary server recorded in infrastructure.json."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess

BASE = Path(__file__).resolve().parent
ROOT = "/opt/ch-budget/code"
OPTIONS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=yes", "-o", "UserKnownHostsFile=/tmp/ch-budget-known-hosts"]


def target():
    server = json.loads((BASE / "infrastructure.json").read_text())
    assert server["name"] == "ch-llm-budget-20261007"
    return "root@" + server["public_net"]["ipv4"]["ip"]


def ssh(command, **kwargs):
    return subprocess.run(["ssh", *OPTIONS, target(), command], check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["upload", "download", "exec", "python", "launch", "status"])
    parser.add_argument("args", nargs="*")
    args = parser.parse_args()
    if args.command == "upload":
        subprocess.run(["scp", *OPTIONS, *[str(BASE / name) for name in args.args], target() + ":" + ROOT + "/"], check=True)
    elif args.command == "download":
        for name in args.args:
            subprocess.run(["scp", "-r", *OPTIONS, target() + ":" + ROOT + "/" + name, str(BASE) + "/"], check=True)
    elif args.command == "exec":
        ssh("cd " + ROOT + " && " + shlex.join(args.args))
    elif args.command == "python":
        code = Path(args.args[0]).read_text()
        ssh("cd " + ROOT + " && python3 -c " + shlex.quote(code))
    elif args.command == "launch":
        phase = args.args[0]
        assert phase in ("screen", "pilot", "full")
        code = f"""
import os,subprocess,sys
from pathlib import Path
root=Path({ROOT!r})
pidfile=root/'results/{phase}.pid'
if pidfile.exists():
    try:
        pid=int(pidfile.read_text())
        stat=Path('/proc')/str(pid)/'stat'
        if stat.exists() and stat.read_text().split()[2] != 'Z':
            raise RuntimeError('Phase process is still running; refusing duplicate launch')
    except ProcessLookupError:
        pass
env=dict(os.environ)
env['OPENROUTER_API_KEY']=sys.stdin.read().strip()
assert env['OPENROUTER_API_KEY']
with (root/'results/{phase}.log').open('ab') as log:
    p=subprocess.Popen(['python3','-u','benchmark.py',{phase!r}],cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
pidfile.write_text(str(p.pid))
print({phase!r},'started, PID',p.pid)
"""
        ssh("python3 -c " + shlex.quote(code), input=os.environ["OPENROUTER_API_KEY"], text=True)
    elif args.command == "status":
        phase = args.args[0]
        assert phase in ("screen", "pilot", "full")
        code = f"""
from pathlib import Path
import json,collections
p=Path({ROOT!r})/'results/{phase}.jsonl'
rows=[json.loads(line) for line in p.read_text().splitlines()] if p.exists() else []
print({phase!r},'attempts',len(rows),'billed USD',round(sum(r.get('cost_usd') or 0 for r in rows),6))
for model in sorted(set(r['model_id'] for r in rows)):
    rs=[r for r in rows if r['model_id']==model]
    print(model,len(rs),'correct',sum(r['correct'] for r in rs),'API errors',sum(r['error'] is not None for r in rs),'SQL errors',sum(r.get('comparison')=='sql_error' for r in rs))
log=p.with_suffix('.log')
if log.exists(): print(*log.read_text().splitlines()[-5:],sep=chr(10))
"""
        ssh("python3 -c " + shlex.quote(code))


if __name__ == "__main__":
    main()
