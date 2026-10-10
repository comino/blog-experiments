#!/usr/bin/env python3
"""Operate only the temporary exploration server recorded beside this script."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess

BASE = Path(__file__).resolve().parent
ROOT = '/opt/ch-exploration'
KNOWN_HOSTS = '/tmp/ch-exploration-20261007-known-hosts'
OPTIONS = ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', '-o', 'StrictHostKeyChecking=accept-new', '-o', 'UserKnownHostsFile=' + KNOWN_HOSTS]


def target():
    server = json.loads((BASE / 'infrastructure.json').read_text())
    assert server['name'] == 'ch-explore-demo-20261007'
    assert server['id'] == 169244472
    return 'root@' + server['public_net']['ipv4']['ip']


def ssh(command, **kwargs):
    return subprocess.run(['ssh', *OPTIONS, target(), 'cd ' + ROOT + ' && ' + command], check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['exec', 'upload', 'download', 'python'])
    parser.add_argument('args', nargs='*')
    args = parser.parse_args()
    if args.command == 'exec':
        ssh(shlex.join(args.args))
    elif args.command == 'upload':
        for name in args.args:
            path = Path(name)
            assert not path.is_absolute() and '..' not in path.parts
            ssh('mkdir -p ' + shlex.quote(str(path.parent)))
            subprocess.run(['scp', '-r', *OPTIONS, str(BASE / path), target() + ':' + ROOT + '/' + str(path.parent) + '/'], check=True)
    elif args.command == 'download':
        for name in args.args:
            path = Path(name)
            assert not path.is_absolute() and '..' not in path.parts
            (BASE / path.parent).mkdir(parents=True, exist_ok=True)
            subprocess.run(['scp', '-r', *OPTIONS, target() + ':' + ROOT + '/' + name, str(BASE / path.parent) + '/'], check=True)
    elif args.command == 'python':
        ssh('python3 -', input=Path(args.args[0]).read_text(), text=True)


if __name__ == '__main__':
    main()
