"""Run the realistic-dataset T4 experiment and release the assignment in all exit paths.

Costs Colab compute units: run only when explicitly approved."""
import hashlib,json,signal,subprocess,sys,tarfile,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent
SESSION='claude-laya-realistic-20261011'
REMOTE='/content/laya-finetune'
events=[]
TEST_ROWS=len(json.loads((ROOT/'data'/'emails.json').read_text()))
def stop_signal(signum,frame):raise KeyboardInterrupt(f'Signal {signum}')
signal.signal(signal.SIGTERM,stop_signal)
signal.signal(signal.SIGALRM,stop_signal)
signal.alarm(12600)
def run(args,timeout=180,check=True):
    started=time.time()
    try:
        p=subprocess.run(args,cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=timeout)
        event={'args':args,'start':started,'seconds':time.time()-started,'returncode':p.returncode,'output':p.stdout}
    except subprocess.TimeoutExpired as exc:
        event={'args':args,'start':started,'seconds':time.time()-started,'returncode':124,'output':'Local command timeout'}
        p=subprocess.CompletedProcess(args,124,event['output'])
    events.append(event);(ROOT/'session-log.json').write_text(json.dumps(events,indent=2))
    print(p.stdout[-5000:],flush=True)
    if check and p.returncode:raise RuntimeError(f'Failed command: {args}; exit {p.returncode}')
    return p
def remote(code,name,timeout=180,check=True):
    f=ROOT/name;f.write_text(code)
    return run([sys.executable,str(ROOT/'console_exec.py'),SESSION,str(f),str(timeout-15)],timeout=timeout,check=check)
def download(name,check=True,timeout=180):
    return run(['colab','download','-s',SESSION,f'{REMOTE}/{name}',str(ROOT/name)],timeout=timeout,check=check)
def phase(script,args,name,timeout):
    code=f"import subprocess,sys\nfrom pathlib import Path\np=Path({REMOTE!r})\nwith (p/{(name+'.log')!r}).open('w') as log:\n    result=subprocess.run([sys.executable,'-u',str(p/{script!r})]+{args!r},stdout=log,stderr=subprocess.STDOUT,timeout={timeout-90})\nprint((p/{(name+'.log')!r}).read_text()[-4000:])\nsys.exit(result.returncode)\n"
    result=remote(code,f'remote_{name}.py',timeout,check=False)
    download(name+'.log',check=False)
    download('train_report.json' if name=='train' else name+'.json',check=False)
    return result
def collect_checkpoint():
    remote(f"import subprocess,sys\nsubprocess.check_call([sys.executable,{REMOTE+'/pack_checkpoint.py'!r}])\n",'remote_pack.py',timeout=180)
    download('checkpoint_manifest.json')
    manifest=json.loads((ROOT/'checkpoint_manifest.json').read_text())
    with (ROOT/'checkpoint.tar').open('wb') as archive:
        for part in manifest['parts']:
            download(part['name'],timeout=300)
            content=(ROOT/part['name']).read_bytes()
            assert len(content)==part['bytes'] and hashlib.sha256(content).hexdigest()==part['sha256']
            archive.write(content);(ROOT/part['name']).unlink()
    with (ROOT/'checkpoint.tar').open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==manifest['sha256']
    with tarfile.open(ROOT/'checkpoint.tar') as archive:archive.extractall(ROOT,filter='data')
    print('CHECKPOINT_DOWNLOADED_AND_VERIFIED',flush=True)
try:
    run(['colab','usage'])
    run(['colab','new','-s',SESSION,'--gpu','T4'],timeout=240)
    run(['colab','usage'])
    run(['colab','upload','-s',SESSION,str(ROOT/'bundle.tar'),'/content/laya-bundle.tar'])
    remote("import tarfile,subprocess,sys,shutil,json,torch\nfrom pathlib import Path\np=Path('/content/laya-finetune');p.mkdir(exist_ok=True)\nwith tarfile.open('/content/laya-bundle.tar') as t:t.extractall(p,filter='data')\npackages=['laya==0.4.2','transformers==5.19.0','bitsandbytes==0.50.2','accelerate==1.15.0','sentencepiece','jsonschema==4.26.0','lm-format-enforcer @ git+https://github.com/noamgat/lm-format-enforcer.git@817f944fcc8851917c40300a47f62d1defc3ffc3','flash-linear-attention==0.5.2']\ncmd=([shutil.which('uv'),'pip','install','--system','--python',sys.executable] if shutil.which('uv') else [sys.executable,'-m','pip','install'])+packages\nsubprocess.check_call(cmd)\nimport fla\nimport os\nenv=dict(os.environ,TORCH_CUDA_ARCH_LIST='7.5',MAX_JOBS='2',CAUSAL_CONV1D_FORCE_BUILD='TRUE')\nsubprocess.Popen(['nohup',sys.executable,'-m','pip','install','--no-build-isolation','causal-conv1d==1.7.0'],stdout=open(p/'causal_build.log','w'),stderr=subprocess.STDOUT,env=env,start_new_session=True)\nprint(json.dumps({'python':sys.version,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0)}))\n",'remote_setup.py',timeout=1200)
    outcome=phase('train_remote.py',[],'train',5400)
    train=json.loads((ROOT/'train_report.json').read_text())
    if train.get('selected_epoch'):collect_checkpoint()
    assert outcome.returncode==0 and train['status']=='complete',train.get('error')
    run(['colab','usage'])
    # Wait (bounded) for the background causal-conv1d build started during setup.
    remote("import subprocess,time,importlib.util,json\nfrom pathlib import Path\nstart=time.time()\nwhile time.time()-start<1200:\n    busy=subprocess.run(['pgrep','-f','pip install --no-build-isolation causal-conv1d'],capture_output=True).returncode==0\n    if not busy:break\n    time.sleep(20)\nok=importlib.util.find_spec('causal_conv1d') is not None\nPath('/content/laya-finetune/causal_status.json').write_text(json.dumps({'installed':ok,'waited_seconds':time.time()-start,'still_building':busy}))\nprint('CAUSAL_STATUS',ok,busy)\nprint(Path('/content/laya-finetune/causal_build.log').read_text()[-3000:])\n",'remote_wait_causal.py',timeout=1320,check=False)
    download('causal_status.json',check=False);download('causal_build.log',check=False)
    for mode in ['laya_original','laya_trained']:
        outcome=phase('evaluate.py',[mode],mode,300)
        result=json.loads((ROOT/f'{mode}.json').read_text())
        assert outcome.returncode==0 and result['status']=='complete' and len(result['results'])==TEST_ROWS
        run(['colab','usage'])
    for mode in ['qwen']:
        outcome=phase('evaluate.py',[mode],mode,2400 if mode=='qwen' else 300)
        result=json.loads((ROOT/f'{mode}.json').read_text())
        assert outcome.returncode==0 and result['status']=='complete' and len(result['results'])==TEST_ROWS
        run(['colab','usage'])
except BaseException:
    (ROOT/'controller-error.txt').write_text(traceback.format_exc());raise
finally:
    signal.alarm(0)
    run(['colab','usage'],check=False)
    result=run(['colab','stop','-s',SESSION],timeout=180,check=False)
    if result.returncode:run(['colab','stop','-s',SESSION],timeout=180,check=False)
    run(['colab','sessions'],check=False)
    run(['colab','usage'],check=False)
