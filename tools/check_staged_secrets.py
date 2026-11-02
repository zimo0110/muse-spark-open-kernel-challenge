#!/usr/bin/env python3
"""Check indexed files without printing credential values or matching text."""
import io
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys


def git(*args, **kwargs):
    return subprocess.check_output(['git', *args], **kwargs)


def local_credentials(root):
    values = set()
    for folder, dirs, names in os.walk(root):
        dirs[:] = [d for d in dirs if d not in {'.git', '.venv', 'venv', '__pycache__', 'node_modules'}]
        for name in names:
            if name != '.env' and not name.startswith('.env.'):
                continue
            for line in (Path(folder) / name).read_text(errors='replace').splitlines():
                line = line.strip().removeprefix('export ')
                if not line or line.startswith('#') or '=' not in line:
                    continue
                key, raw = line.split('=', 1)
                raw = raw.strip()
                if raw[:1] in {'"', "'"}:
                    try:
                        raw = ' '.join(shlex.split(raw, comments=True))
                    except ValueError:
                        continue
                else:
                    raw = raw.split(' #', 1)[0].strip()
                if len(raw) >= 8:
                    values.add(raw.encode())
                if key.strip() == 'SSH_NAME':
                    try:
                        target = shlex.split(raw)[-1]
                    except (ValueError, IndexError):
                        continue
                    if '@' in target:
                        for value in (target, target.rsplit('@', 1)[-1]):
                            if len(value) >= 8:
                                values.add(value.encode())
    for key in ('OKC_TOKEN', 'SSH_PASS', 'GITHUB_TOKEN', 'GH_TOKEN'):
        value = os.environ.get(key)
        if value and len(value) >= 8:
            values.add(value.encode())
    return values


def main():
    root = Path(git('rev-parse', '--show-toplevel', text=True).strip())
    entries = []
    failures = []
    for item in git('ls-files', '--stage', '-z').split(b'\0'):
        if not item:
            continue
        metadata, raw_path = item.split(b'\t', 1)
        mode, oid, stage = metadata.split()
        path = raw_path.decode(errors='replace')
        name = Path(path).name.lower()
        if name == '.env' or name.startswith('.env.') or name.endswith(('.pem', '.key', '.p12', '.pfx')):
            failures.append((path, 'credential file is indexed'))
        if stage == b'0' and mode != b'160000':
            entries.append((path, oid))
    known = local_credentials(root)
    patterns = [
        ('private key', re.compile(rb'-----BEGIN (?:OPENSSH |RSA |EC |DSA |ENCRYPTED )?PRIVATE KEY-----')),
        ('GitHub token', re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,})')),
        ('API key', re.compile(rb'sk-(?:proj-)?[A-Za-z0-9_-]{32,}')),
        ('literal credential assignment', re.compile(rb'''(?i)(?:api[_-]?key|access[_-]?token|ssh[_-]?pass|password)\s*[:=]\s*["'][^"'\r\n]{8,}["']''')),
    ]
    query = b''.join(oid + b'\n' for _, oid in entries)
    batch = subprocess.run(['git', 'cat-file', '--batch'], input=query, capture_output=True, check=True)
    stream = io.BytesIO(batch.stdout)
    for path, _ in entries:
        header = stream.readline().split()
        if len(header) != 3 or header[1] != b'blob':
            raise RuntimeError('Cannot inspect an indexed object; commit blocked.')
        content = stream.read(int(header[2]))
        assert stream.read(1) == b'\n'
        if any(value in content for value in known):
            failures.append((path, 'contains a known local credential value'))
        for label, pattern in patterns:
            if pattern.search(content):
                failures.append((path, label))
    if failures:
        print('Commit blocked. Matching credential values are withheld.', file=sys.stderr)
        for path, reason in sorted(set(failures)):
            print(f'  {path}: {reason}', file=sys.stderr)
        return 1
    print(f'Secret check passed: {len(entries)} indexed files; no environment files or detected credentials.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
