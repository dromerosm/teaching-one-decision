"""Full-epoch QLoRA of Qwen3.5-9B (run B) on one T4 session; releases the assignment in all exit paths.

Run A (same day) stopped at 448 examples because its plan measured speed during Triton
warm-up. This run trains on the whole training split, then builds causal-conv1d (only
after training, never alongside the 9B model), then evaluates the adapter on the test.

Order: setup -> causal-conv1d build with nothing else running (attempt 2b showed that
building it while a 9B model trains exhausts the VM) -> qwen_prompt, qwen_fewshot,
qwen_think (100-email subset, time-capped) -> QLoRA training -> adapter download ->
qwen_qlora. Prompt-only variants are saved before the QLoRA starts. Costs Colab compute units: run only when approved.
"""
import hashlib, json, signal, subprocess, sys, tarfile, time, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SESSION = "claude-qwen-qlora-full-20261012"
REMOTE = "/content/laya-finetune"
BUDGET = 17000  # QLoRA training time cap, seconds (one epoch is about 4.4 h)
events = []
TEST_ROWS = len(json.loads((ROOT / "data" / "emails.json").read_text()))


def stop_signal(signum, frame):
    raise KeyboardInterrupt(f"Signal {signum}")


signal.signal(signal.SIGTERM, stop_signal)
signal.signal(signal.SIGALRM, stop_signal)
signal.alarm(34000)


def assignment_alive():
    try:
        out = subprocess.run(["colab", "usage"], text=True, capture_output=True, timeout=60).stdout
        return "Active assignments: 0" not in out
    except Exception:
        return True  # unknown: do not kill on a transient CLI error


def run(args, timeout=180, check=True):
    started = time.time()
    proc = subprocess.Popen(args, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    watch = args[0] == sys.executable  # long remote console calls get a liveness watchdog
    last_check, code, note = time.time(), None, ""
    while True:
        try:
            out, _ = proc.communicate(timeout=30)
            code = proc.returncode
            break
        except subprocess.TimeoutExpired:
            if time.time() - started > timeout:
                proc.kill(); out, _ = proc.communicate(); code, note = 124, "Local command timeout"
                break
            if watch and time.time() - last_check > 300:
                last_check = time.time()
                if not assignment_alive():
                    proc.kill(); out, _ = proc.communicate(); code, note = 125, "Colab assignment disappeared"
                    break
    event = {"args": args, "start": started, "seconds": time.time() - started, "returncode": code, "output": (out or "") + note}
    p = subprocess.CompletedProcess(args, code, event["output"])
    events.append(event); (ROOT / "session3-log.json").write_text(json.dumps(events, indent=2))
    print(p.stdout[-5000:], flush=True)
    if check and p.returncode:
        raise RuntimeError(f"Failed command: {args}; exit {p.returncode}")
    return p


def remote(code, name, timeout=180, check=True):
    f = ROOT / name; f.write_text(code)
    return run([sys.executable, str(ROOT / "console_exec.py"), SESSION, str(f), str(timeout - 15)], timeout=timeout, check=check)


def download(name, check=True, timeout=300):
    return run(["colab", "download", "-s", SESSION, f"{REMOTE}/{name}", str(ROOT / name)], timeout=timeout, check=check)


def phase(script, args, name, result, timeout, env=""):
    code = (f"import subprocess,sys,os\nfrom pathlib import Path\np=Path({REMOTE!r})\n"
            f"env=dict(os.environ,{env})\n"
            f"with (p/{name + '.log'!r}).open('w') as log:\n"
            f"    log.write(subprocess.run(['free','-m'],capture_output=True,text=True).stdout);log.flush()\n"
            f"    try:\n        r=subprocess.run([sys.executable,'-u',str(p/{script!r})]+{args!r},stdout=log,stderr=subprocess.STDOUT,timeout={timeout - 120},env=env).returncode\n"
            f"    except subprocess.TimeoutExpired:\n        r=124\n"
            f"print((p/{name + '.log'!r}).read_text()[-4000:])\nsys.exit(r)\n")
    out = remote(code, f"remote3_{name}.py", timeout, check=False)
    download(name + ".log", check=False)
    download(result, check=False)
    return out


def collect(dirname, prefix):
    remote(f"import subprocess,sys\nsubprocess.check_call([sys.executable,{REMOTE + '/pack_checkpoint.py'!r},{dirname!r},{prefix!r}])\n", f"remote3_pack_{prefix}.py", timeout=300)
    download(f"{prefix}_manifest.json")
    manifest = json.loads((ROOT / f"{prefix}_manifest.json").read_text())
    with (ROOT / f"{prefix}.tar").open("wb") as archive:
        for part in manifest["parts"]:
            download(part["name"], timeout=600)
            content = (ROOT / part["name"]).read_bytes()
            assert len(content) == part["bytes"] and hashlib.sha256(content).hexdigest() == part["sha256"]
            archive.write(content); (ROOT / part["name"]).unlink()
    with (ROOT / f"{prefix}.tar").open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == manifest["sha256"]
    with tarfile.open(ROOT / f"{prefix}.tar") as archive:
        archive.extractall(ROOT, filter="data")
    print(f"{prefix.upper()}_DOWNLOADED_AND_VERIFIED", flush=True)


SETUP = (
    "import tarfile,subprocess,sys,shutil,json,os,torch\nfrom pathlib import Path\n"
    "p=Path('/content/laya-finetune');p.mkdir(exist_ok=True)\n"
    "with tarfile.open('/content/laya-bundle3.tar') as t:t.extractall(p,filter='data')\n"
    "packages=['laya==0.4.2','transformers==5.19.0','bitsandbytes==0.50.2','accelerate==1.15.0','sentencepiece','jsonschema==4.26.0',"
    "'lm-format-enforcer @ git+https://github.com/noamgat/lm-format-enforcer.git@817f944fcc8851917c40300a47f62d1defc3ffc3','flash-linear-attention==0.5.2','peft==0.21.0']\n"
    "cmd=([shutil.which('uv'),'pip','install','--system','--python',sys.executable] if shutil.which('uv') else [sys.executable,'-m','pip','install'])+packages\n"
    "subprocess.check_call(cmd)\nimport fla,peft\n"

    "print(json.dumps({'python':sys.version,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),'peft':peft.__version__}))\n")

WAIT_CAUSAL = (
    "import subprocess,time,importlib.util,json,os,sys\nfrom pathlib import Path\nstart=time.time()\nbusy=True\n"
    "env=dict(os.environ,TORCH_CUDA_ARCH_LIST='7.5',MAX_JOBS='2',CAUSAL_CONV1D_FORCE_BUILD='TRUE')\n"
    "subprocess.Popen(['nohup',sys.executable,'-m','pip','install','--no-build-isolation','causal-conv1d==1.7.0'],stdout=open('/content/laya-finetune/causal_build3.log','w'),stderr=subprocess.STDOUT,env=env,start_new_session=True)\n"
    "time.sleep(30)\n"
    "while time.time()-start<5400:\n"
    "    busy=subprocess.run(['pgrep','-f','pip install --no-build-isolation causal-conv1d'],capture_output=True).returncode==0\n"
    "    if not busy:break\n    time.sleep(20)\n"
    "ok=importlib.util.find_spec('causal_conv1d') is not None\n"
    "mem=subprocess.run(['free','-m'],capture_output=True,text=True).stdout\n"
    "Path('/content/laya-finetune/causal_status3.json').write_text(json.dumps({'installed':ok,'waited_seconds':time.time()-start,'still_building':busy,'free_m':mem}))\n"
    "print('CAUSAL_STATUS',ok,busy)\n")

try:
    run(["colab", "usage"])
    for attempt in range(8):  # T4 capacity is sometimes temporarily unavailable
        if run(["colab", "new", "-s", SESSION, "--gpu", "T4"], timeout=240, check=False).returncode == 0:
            break
        time.sleep(300)
    else:
        raise RuntimeError("no T4 assignment after 8 attempts")
    run(["colab", "usage"])
    for attempt in range(3):
        if run(["colab", "upload", "-s", SESSION, str(ROOT / "bundle3.tar"), "/content/laya-bundle3.tar"], timeout=600, check=False).returncode == 0:
            break
    else:
        raise RuntimeError("bundle upload failed three times")
    remote(SETUP, "remote3_setup.py", timeout=1200)
    qlora_ok = False
    try:
        phase("train_qlora.py", [], "qlora_train_full", "qlora_report_full.json", BUDGET + 3600,
              env=f"QLORA_BUDGET_SECONDS='{BUDGET}',QLORA_TAG='full'")
        rep = json.loads((ROOT / "qlora_report_full.json").read_text())
        if rep.get("status") == "complete":
            collect("qlora_adapter_full", "qlora_adapter_full")
            qlora_ok = True
    except Exception:
        (ROOT / "qlora-full-controller-error.txt").write_text(traceback.format_exc())
    run(["colab", "usage"])
    if qlora_ok:
        remote(WAIT_CAUSAL, "remote3_wait_causal.py", timeout=5600, check=False)
        download("causal_status3.json", check=False)
        run(["colab", "usage"])
        phase("evaluate.py", ["qwen_qlora_full"], "qwen_qlora_full", "qwen_qlora_full.json", 3600)
        run(["colab", "usage"])
except BaseException:
    (ROOT / "controller3-error.txt").write_text(traceback.format_exc()); raise
finally:
    signal.alarm(0)
    run(["colab", "usage"], check=False)
    r = run(["colab", "stop", "-s", SESSION], timeout=180, check=False)
    if r.returncode:
        run(["colab", "stop", "-s", SESSION], timeout=180, check=False)
    run(["colab", "sessions"], check=False)
    run(["colab", "usage"], check=False)
