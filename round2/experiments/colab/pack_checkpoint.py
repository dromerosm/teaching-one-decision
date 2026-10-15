"""Save a checksummed directory (default: checkpoint) in modest chunks for the Contents API.

Usage: pack_checkpoint.py [dirname] [prefix]
"""
import hashlib,json,sys,tarfile
from pathlib import Path
ROOT=Path('/content/laya-finetune')
dirname=sys.argv[1] if len(sys.argv)>1 else 'checkpoint'
prefix=sys.argv[2] if len(sys.argv)>2 else 'checkpoint'
archive=ROOT/f'{prefix}.tar'
with tarfile.open(archive,'w') as tar:tar.add(ROOT/dirname,arcname=dirname)
parts=[]
with archive.open('rb') as source:
    i=0
    while chunk:=source.read(48*1024*1024):
        name=f'{prefix}.part{i:03d}';(ROOT/name).write_bytes(chunk)
        parts.append({'name':name,'bytes':len(chunk),'sha256':hashlib.sha256(chunk).hexdigest()});i+=1
manifest={'archive':f'{prefix}.tar','bytes':archive.stat().st_size,'sha256':hashlib.file_digest(archive.open('rb'),'sha256').hexdigest(),'parts':parts}
(ROOT/f'{prefix}_manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps({'parts':len(parts),'bytes':manifest['bytes'],'sha256':manifest['sha256']}))
