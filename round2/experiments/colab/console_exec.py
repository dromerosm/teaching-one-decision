"""Run a saved Python script through the authenticated Colab terminal."""
import base64,os,pty,re,select,shlex,subprocess,sys,time
from pathlib import Path
session,filename,limit=sys.argv[1:]
payload=base64.b64encode(Path(filename).read_bytes()).decode()
code="import base64;exec(compile(base64.b64decode("+repr(payload)+"),"+repr(filename)+",'exec'))"
command="python3 -c "+shlex.quote(code)+"; printf '\\nCODEX_RESULT:%s\\n' \"$?\"\n"
master,slave=pty.openpty()
process=subprocess.Popen(['colab','console','-s',session],stdin=slave,stdout=slave,stderr=slave,start_new_session=True)
os.close(slave)
deadline=time.monotonic()+float(limit)
buffer=b'';sent=False;result=None
try:
    while time.monotonic()<deadline:
        if not select.select([master],[],[],1)[0]:
            if process.poll() is not None:break
            continue
        try: chunk=os.read(master,65536)
        except OSError:break
        if not chunk:break
        buffer+=chunk
        if not sent and b'/content' in buffer and b'#' in buffer:
            os.write(master,command.encode());sent=True;buffer=b''
        if sent:
            match=re.search(rb'CODEX_RESULT:(\d+)',buffer)
            if match:
                result=int(match.group(1));os.write(master,b'exit\n');break
    if result is None:raise TimeoutError('Colab terminal did not return a completion marker')
    clean=re.sub(rb'\x1b\[[0-?]*[ -/]*[@-~]',b'',buffer)
    print(clean.decode(errors='replace')[-24000:])
    sys.exit(result)
finally:
    try:process.wait(timeout=5)
    except subprocess.TimeoutExpired:process.terminate();process.wait(timeout=5)
    os.close(master)
