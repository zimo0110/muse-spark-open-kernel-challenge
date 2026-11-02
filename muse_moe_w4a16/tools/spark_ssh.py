"""Run a Python development probe over SSH without exposing saved credentials.

Loads only SSH_NAME/SSH_PASS from the existing local credential file. Never
prints command-line connection arguments, pexpect diagnostic buffers, or secrets.
The remote Python source is provided as a local file; credentials are not copied.
"""
import argparse
import base64
import importlib.util
import re
import shlex
import sys
import zlib
from pathlib import Path
import pexpect


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('python_file',type=Path)
    parser.add_argument('--timeout',type=int,default=90)
    parser.add_argument('--sudo',action='store_true',help='Run the development probe with sudo; password is sent only through the SSH channel.')
    args=parser.parse_args()
    envfile=Path(__file__).resolve().parents[2]/'muse_bf16_proj'/'.env.local'
    if importlib.util.find_spec('dotenv'):
        from dotenv import dotenv_values
        values=dotenv_values(envfile)
    else:
        values={}
        for line in envfile.read_text().splitlines():
            line=line.strip()
            if not line or line.startswith('#') or '=' not in line:continue
            key,value=line.removeprefix('export ').split('=',1)
            values[key.strip()]=' '.join(shlex.split(value,comments=True))
    name,password=values.get('SSH_NAME'),values.get('SSH_PASS')
    if not name or not password:raise SystemExit('Required SSH variables are missing.')
    argv=shlex.split(name)
    if not argv or Path(argv[0]).name!='ssh':raise SystemExit('SSH_NAME must be a plain SSH command.')
    payload=base64.b64encode(zlib.compress(args.python_file.read_bytes(),9)).decode('ascii')
    if len(payload)>64000:raise SystemExit('Development probe is too large; split the file transfer.')
    launcher="import base64,zlib;exec(compile(zlib.decompress(base64.b64decode("+repr(payload)+")),'<development-probe>','exec'))"
    command='python3 -c '+shlex.quote(launcher)
    if args.sudo:
        command="sudo -S -p '[codex-sudo] ' "+command
    redact=[password,name]
    # The saved command's last token is its login target, never print it.
    if argv[-1] and not argv[-1].startswith('-'):
        redact.append(argv[-1])
        redact.extend(v for v in argv[-1].split('@') if v)
    def safe(value):
        for secret in sorted(set(redact),key=len,reverse=True):
            value=value.replace(secret,'[REDACTED]')
        return value
    child=pexpect.spawn('ssh',['-o','StrictHostKeyChecking=accept-new','-o','ConnectTimeout=10','-o','NumberOfPasswordPrompts=1',*argv[1:],command],encoding='utf-8',timeout=args.timeout,echo=False)
    child.logfile=None
    chunks=[];sent=False;sudo_sent=False
    child.delaybeforesend=0
    try:
        while True:
            event=child.expect([r'(?i)password[^\r\n]*:',r'\[codex-sudo\] ',pexpect.EOF,pexpect.TIMEOUT])
            if event==0:
                if sent:raise RuntimeError('authentication failed')
                child.sendline(password);sent=True
            elif event==1:
                if sudo_sent or not args.sudo:raise RuntimeError('sudo authentication failed')
                child.sendline(password);sudo_sent=True
            elif event==2:
                chunks.append(child.before);break
            else:raise RuntimeError('probe timed out')
        child.close()
        sys.stdout.write(safe(''.join(chunks)))
        return child.exitstatus or (1 if child.signalstatus else 0)
    except (pexpect.ExceptionPexpect,RuntimeError):
        print('SSH development probe failed or timed out; transport buffers withheld.')
        return 1
    finally:
        if child.isalive():child.close(force=True)


if __name__=='__main__':
    raise SystemExit(main())
