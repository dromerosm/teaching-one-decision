"""Run one job on a Google Colab GPU with the Colab CLI, and always release the GPU afterwards.

Prerequisites on your machine: the Colab CLI (`colab`), already authenticated, and Python 3.10+.
The job bundle is a tar file with everything the remote side needs (data and scripts).

Example (Qwen prompt v2 on the round-2 test):
  python infra/colab/run_on_colab.py \
      --session <SESSION_NAME> --gpu T4 \
      --bundle <PATH_TO_BUNDLE.tar> \
      --command "python -u evaluate.py qwen_prompt" \
      --download qwen_prompt.json --download qwen_prompt.log \
      --with-causal-conv1d

The remote working directory is /content/job. Commands run through `colab console`, so the
script waits for a completion marker and polls the assignment every 5 minutes: if the Colab
assignment disappears, it stops waiting instead of hanging.
"""
import argparse
import base64
import os
import pty
import re
import select
import shlex
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REMOTE = "/content/job"


def sh(args, timeout=600, check=True):
    p = subprocess.run(args, text=True, capture_output=True, timeout=timeout)
    print(p.stdout[-3000:], p.stderr[-2000:], flush=True)
    if check and p.returncode:
        raise RuntimeError(f"{args} exited {p.returncode}")
    return p


def alive(session):
    out = subprocess.run(["colab", "usage"], text=True, capture_output=True).stdout
    return "Active assignments: 0" not in out


def remote_python(session, code, timeout):
    """Run Python code on the Colab machine through `colab console`; return its exit code."""
    payload = base64.b64encode(code.encode()).decode()
    command = "python3 -c " + shlex.quote(f"import base64;exec(base64.b64decode({payload!r}))") + "; printf '\\nJOB_RESULT:%s\\n' \"$?\"\n"
    master, slave = pty.openpty()
    proc = subprocess.Popen(["colab", "console", "-s", session], stdin=slave, stdout=slave, stderr=slave, start_new_session=True)
    os.close(slave)
    buf, sent, result, start, last_check = b"", False, None, time.time(), time.time()
    try:
        while time.time() - start < timeout:
            if time.time() - last_check > 300:
                last_check = time.time()
                if not alive(session):
                    print("Colab assignment disappeared", flush=True)
                    return 125
            if not select.select([master], [], [], 1)[0]:
                continue
            chunk = os.read(master, 65536)
            buf += chunk
            if not sent and b"/content" in buf:
                os.write(master, command.encode()); sent = True; buf = b""
            m = re.search(rb"JOB_RESULT:(\d+)", buf)
            if sent and m:
                result = int(m.group(1)); os.write(master, b"exit\n"); break
        print(re.sub(rb"\x1b\[[0-?]*[ -/]*[@-~]", b"", buf).decode(errors="replace")[-6000:], flush=True)
        return 124 if result is None else result
    finally:
        proc.terminate(); os.close(master)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True, help="a name for the Colab session, e.g. <SESSION_NAME>")
    ap.add_argument("--gpu", default="T4")
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--command", required=True, help="shell command run in /content/job")
    ap.add_argument("--download", action="append", default=[], help="file in /content/job to copy back")
    ap.add_argument("--with-causal-conv1d", action="store_true")
    ap.add_argument("--timeout", type=int, default=4 * 3600)
    a = ap.parse_args()
    try:
        sh(["colab", "usage"])
        for _ in range(8):  # T4 capacity is sometimes temporarily unavailable
            if sh(["colab", "new", "-s", a.session, "--gpu", a.gpu], check=False).returncode == 0:
                break
            time.sleep(300)
        else:
            raise RuntimeError("no GPU assignment")
        for _ in range(3):  # uploads occasionally fail with a broken pipe
            if sh(["colab", "upload", "-s", a.session, a.bundle, "/content/job.tar"], check=False).returncode == 0:
                break
        else:
            raise RuntimeError("upload failed")
        setup = (HERE / "setup_remote.py").read_text()
        argv = ["setup_remote.py"] + (["--with-causal-conv1d"] if a.with_causal_conv1d else [])
        code = (f"import tarfile,os,sys\nos.makedirs({REMOTE!r},exist_ok=True)\n"
                f"tarfile.open('/content/job.tar').extractall({REMOTE!r},filter='data')\n"
                f"sys.argv={argv!r}\n" + setup)
        assert remote_python(a.session, code, 5400) == 0, "setup failed"
        job = f"import subprocess,sys\nsys.exit(subprocess.call({a.command!r},shell=True,cwd={REMOTE!r}))\n"
        rc = remote_python(a.session, job, a.timeout)
        for f in a.download:
            sh(["colab", "download", "-s", a.session, f"{REMOTE}/{f}", f], check=False)
        sys.exit(rc)
    finally:
        sh(["colab", "stop", "-s", a.session], check=False)
        sh(["colab", "usage"], check=False)


if __name__ == "__main__":
    main()
