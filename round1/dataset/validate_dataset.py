"""Audit outputs independently of the generator; exit nonzero on failed checks."""
import collections,hashlib,json,re,unicodedata
from pathlib import Path
from jsonschema import Draft202012Validator

ROOT=Path(__file__).resolve().parent
SIZES={'train':2000,'validation':400,'calibration':400,'test':400}
def read_rows(path):return [json.loads(line) for line in path.read_text().splitlines() if line]
def normalize(s):
    s=unicodedata.normalize('NFKC',s).lower()
    s=re.sub(r'\b[0-9a-f]{8}\b','REF',s)
    s=re.sub(r'\d+','NUMBER',s)
    return ' '.join(re.findall(r'\w+',s))
def shingles(s):
    words=normalize(s).split()
    return {tuple(words[i:i+3]) for i in range(len(words)-2)}

schema=json.loads((ROOT/'schema.json').read_text())
Draft202012Validator.check_schema(schema);validator=Draft202012Validator(schema)
questions=json.loads((ROOT/'questions.json').read_text())
manifest=json.loads((ROOT/'manifest.json').read_text())
records=read_rows(ROOT/'records.jsonl')
assert len(records)==3200 and len({r['id'] for r in records})==3200
family_splits=collections.defaultdict(set);groups=collections.defaultdict(list)
exact={};normal={};stats={};id_rows={r['id']:r for r in records}
for r in records:
    validator.validate(r['expected'])
    assert type(r['expected']['needs_reply']) is bool
    assert r['state']==f"Subject: {r['subject']}\nBody: {r['body']}"
    assert r['split'] in SIZES and r['language'] in ['en','es']
    assert '{request' not in r['state'] and '{minutes}' not in r['state']
    assert not re.search(r'(?:podrías|puedes) (?:envía|confirma|explica|indica|propón)\b',r['state'],re.I)
    assert not re.search(r'family_id|counterfactual_group|label_provenance|expected\s*:',r['state'])
    assert set(r['evidence'])==set(r['expected'])
    for evidence in r['evidence'].values():assert evidence and evidence in r['body']
    for mapping,key in [(exact,r['state']),(normal,normalize(r['state']))]:
        assert key not in mapping,('duplicate',r['id'],mapping.get(key))
        mapping[key]=r['id']
    family_splits[r['family_id']].add(r['split'])
    groups[r['counterfactual_group']].append(r)
    # Policy checks use the visible evidence, not a lookup of lifecycle labels.
    priority=r['expected']['priority'];e=r['evidence']['priority'].lower()
    if priority=='high':
        assert re.search(r'\d+ (?:minutes|minutos)|new production requests|solicitudes de producción',e),(r['id'],e)
    elif priority=='low':
        assert re.search(r'complete|closed|finished|accepted|settled|resolved|fulfilled|approved|delivered|no work remains|resuelt|terminad|cerrad|complet|aceptad|atendid|aprob',e),(r['id'],e)
    elif priority=='normal':
        assert re.search(r'week|month|quarter|working days|standard processing|regular work|no same-day|no deadline today|no need to complete|usual queue|semana|mes|trimestre|días laborables|habitual|no hace falta|no hay una fecha límite hoy|sin un plazo que venza hoy',e),(r['id'],e)
    if not r['expected']['needs_reply'] and priority!='low':
        assert 'action_without_reply' in r['tags']
    if 'embedded_instruction' in r['tags']:
        assert '> ' in r['body'] and ('not an instruction' in r['body'] or 'no una instrucción' in r['body'])

assert len(family_splits)==80 and all(len(v)==1 for v in family_splits.values())
assert len(groups)==400
for group in groups.values():
    assert len(group)==8 and len({r['split'] for r in group})==1
    assert collections.Counter(r['language'] for r in group)=={'en':4,'es':4}
    assert len({r['lifecycle'] for r in group})==4
    for mode in {r['lifecycle'] for r in group}:
        pair=[r for r in group if r['lifecycle']==mode]
        assert pair[0]['expected']==pair[1]['expected']
    assert {r['expected']['priority'] for r in group}=={'low','normal','high'}
    assert {r['expected']['needs_reply'] for r in group}=={True,False}

for split,size in SIZES.items():
    rs=[r for r in records if r['split']==split]
    train_rows=read_rows(ROOT/f'{split}.jsonl');inputs=read_rows(ROOT/f'{split}_inputs.jsonl');labels=read_rows(ROOT/f'{split}_labels.jsonl')
    assert len(rs)==len(train_rows)==len(inputs)==len(labels)==size
    assert collections.Counter(r['language'] for r in rs)=={'en':size//2,'es':size//2}
    assert all(v==size//5 for v in collections.Counter(r['expected']['category'] for r in rs).values())
    assert collections.Counter(r['expected']['priority'] for r in rs)=={'low':size//4,'normal':size*3//8,'high':size*3//8}
    assert collections.Counter(r['expected']['needs_reply'] for r in rs)=={True:size//2,False:size//2}
    for canonical,training,inp,label in zip(rs,train_rows,inputs,labels):
        assert set(training)=={'state','questions','expected'}
        assert training['state']==canonical['state'] and training['expected']==canonical['expected']
        assert training['questions']==questions==inp['questions']
        assert set(inp)=={'id','state','questions'}
        assert inp['id']==label['id']==canonical['id'] and inp['state']==canonical['state']
        assert label['expected']==canonical['expected']
    words=[len(r['body'].split()) for r in rs]
    stats[split]={'rows':size,'families':len({r['family_id'] for r in rs}),
                 'language':dict(collections.Counter(r['language'] for r in rs)),
                 'category':dict(collections.Counter(r['expected']['category'] for r in rs)),
                 'priority':dict(collections.Counter(r['expected']['priority'] for r in rs)),
                 'needs_reply':dict(collections.Counter(str(r['expected']['needs_reply']).lower() for r in rs)),
                 'tags':dict(collections.Counter(tag for r in rs for tag in r['tags'])),
                 'body_words':{'min':min(words),'max':max(words),'mean':sum(words)/len(words)}}

for file,digest in manifest['files_sha256'].items():
    assert hashlib.sha256((ROOT/file).read_bytes()).hexdigest()==digest,file

# Inverted index makes cross-split lexical similarity auditable without a vector service.
ss=[shingles(r['state']) for r in records]
index=collections.defaultdict(list)
for i,sh in enumerate(ss):
    for token in sh:index[token].append(i)
nearest=[]
for i,r in enumerate(records):
    counts=collections.Counter()
    for token in ss[i]:
        for j in index[token]:
            if j>i and records[j]['split']!=r['split'] and records[j]['language']==r['language']:counts[j]+=1
    if counts:
        j,score=max(((j,n/(len(ss[i])+len(ss[j])-n)) for j,n in counts.items()),key=lambda item:item[1])
        nearest.append({'left':r['id'],'right':records[j]['id'],'jaccard_3gram':score})
nearest.sort(key=lambda x:x['jaccard_3gram'],reverse=True)
assert not nearest or nearest[0]['jaccard_3gram']<0.80,('cross-split near duplicate',nearest[0])

reference=ROOT/'smoke-emails.json'  # optional: the 50-email smoke set used before round 1 (not published)
reference_check={'available':reference.exists()}
if reference.exists():
    old=json.loads(reference.read_text());old_states={normalize(f"Subject: {r['subject']}\nBody: {r['body']}") for r in old}
    overlap=set(normal)&old_states
    assert not overlap
    reference_check.update(rows=len(old),normalized_exact_overlap=0,sha256=hashlib.sha256(reference.read_bytes()).hexdigest())

report={'status':'passed','checks':['schema validity','native booleans','unique IDs','exact and normalized text uniqueness','family separation','translation and counterfactual grouping','balanced label counts','source evidence spans','export/label alignment','input-only exports exclude gold labels','file hashes','cross-split lexical near-duplicate threshold'],
        'counts':stats,'cross_split_family_overlap':0,'exact_duplicates':0,'normalized_duplicates':0,
        'cross_split_similarity':{'method':'word-trigram Jaccard after case, reference and number normalization; same-language pairs only','rejection_threshold':0.80,'maximum':nearest[0]['jaccard_3gram'],'top_20_pairs':nearest[:20]},
        'prior_smoke_reference':reference_check,
        'review_limits':'Checks establish structure, grouping and selected policy evidence. They do not certify every semantic label or production realism. Shared generic context/quote templates remain across splits.'}
(ROOT/'validation_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':'passed','rows':len(records),'families':len(family_splits),'counts':{s:x['rows'] for s,x in stats.items()},'max_cross_split_similarity':nearest[0]['jaccard_3gram'],'prior_smoke_reference':reference_check},indent=2))
