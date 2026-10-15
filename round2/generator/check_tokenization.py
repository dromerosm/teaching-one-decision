"""Check lengths with the pinned Laya tokenizer, without downloading model weights."""
import hashlib,importlib.metadata,json,tempfile,urllib.request
from pathlib import Path
from tokenizers import Tokenizer

ROOT=Path(__file__).resolve().parent
REPO='convaiinnovations/laya-multilingual'
REVISION='1720e3e3357cfe1e281542e223f8273b0890ca34'
path=Path(tempfile.gettempdir())/f'laya-tokenizer-{REVISION}.json'
url=f'https://huggingface.co/{REPO}/resolve/{REVISION}/tokenizer/tokenizer.json'
if not path.exists():
    with urllib.request.urlopen(url,timeout=90) as response:path.write_bytes(response.read())
tokenizer=Tokenizer.from_file(str(path))
tokenizer.no_truncation()
rows=[json.loads(line) for line in (ROOT.parent/"dataset"/"records.jsonl").read_text().splitlines()]
encodings=tokenizer.encode_batch([r['state'] for r in rows])
questions=json.loads((ROOT.parent/"dataset"/"questions.json").read_text())
q_lengths={k:len(tokenizer.encode(json.dumps(v,ensure_ascii=False)).ids) for k,v in questions.items()}
lengths=[len(e.ids) for e in encodings]
assert max(lengths)<=1024
assert max(q_lengths.values())<=256
assert max(lengths)+max(q_lengths.values())+16<=1024
report={'status':'passed','repo':REPO,'revision':REVISION,'tokenizers_version':importlib.metadata.version('tokenizers'),
        'tokenizer_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'state_tokens':{'min':min(lengths),'max':max(lengths),'mean':sum(lengths)/len(lengths)},
        'state_tokens_by_language':{lang:{'max':max(n for r,n in zip(rows,lengths) if r['language']==lang),'mean':sum(n for r,n in zip(rows,lengths) if r['language']==lang)/sum(r['language']==lang for r in rows)} for lang in ['en','es']},
        'serialized_question_tokens':q_lengths,'conservative_pair_max':max(lengths)+max(q_lengths.values())+16,
        'state_limit_checked':1024,'serialized_question_limit_checked':256,
        'scope':'Pinned tokenizer lengths with truncation disabled. Serialized question JSON is a conservative proxy, not the Laya training item renderer. No model load, training, GPU allocation or laya-train dry-run was performed.'}
(ROOT.parent/'dataset'/'tokenizer_check.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
