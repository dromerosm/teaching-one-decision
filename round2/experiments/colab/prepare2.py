"""Package the Qwen fairness run (prompt v2, few-shot, thinking subset, QLoRA). Local only.

- data/fewshot.json: 8 training rows (two counterfactual twin pairs, one by received date and
  one by SLA, plus a notice, an action without reply, a reply with a question to a colleague in
  copy, and an incident with customer impact). Shortest eligible rows, fixed choice.
- data/think_subset.json: 100 test ids, a fixed hash-ordered sample used for the thinking run.
- bundle2.tar: data + scripts for the remote side.
"""
import hashlib, json, tarfile
from pathlib import Path
from qwen_common import target_json

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
rows = [json.loads(l) for l in open(DATA / "records.jsonl")]
train = [r for r in rows if r["split"] == "train"]
by_id = {r["id"]: r for r in train}
L = lambda r: len(r["state"])


def pair(change, cat):
    c = [(by_id[r["twin_of"]], r) for r in train if r["twin_change"] == change and r["twin_of"] in by_id and r["expected"]["category"] == cat]
    return min(c, key=lambda p: L(p[0]))


picks = list(pair("received_date", "scheduling")) + list(pair("sla", "billing"))
used = {r["id"] for r in picks} | {r["twin_of"] for r in picks if r["twin_of"]}
cands = [r for r in train if r["id"] not in used and not r["twin_of"]]
rules = [lambda r: r["lifecycle"] == "notice" and r["expected"]["priority"] == "low" and r["expected"]["category"] == "information",
         lambda r: r["lifecycle"] == "action" and r["expected"]["priority"] == "normal" and not r["spec"]["deadline"] and r["expected"]["category"] == "sales",
         lambda r: r["lifecycle"] == "reply" and r["spec"]["quoted"] != "none" and r["expected"]["category"] == "billing",
         lambda r: r["spec"]["incident_ref"] and r["expected"]["priority"] == "high" and r["lifecycle"] != "notice"]
for rule in rules:
    r = min((x for x in cands if rule(x) and x["id"] not in {p["id"] for p in picks}), key=L)
    picks.append(r)
fewshot = [{"id": r["id"], "state": r["state"], "answer": target_json(r["expected"]), "why": r["twin_change"] or r["lifecycle"]} for r in picks]
(DATA / "fewshot.json").write_text(json.dumps(fewshot, ensure_ascii=False, indent=1) + "\n")
test_ids = sorted((r["id"] for r in rows if r["split"] == "test"), key=lambda i: hashlib.sha256(i.encode()).hexdigest())[:100]
(DATA / "think_subset.json").write_text(json.dumps(test_ids) + "\n")
with tarfile.open(ROOT / "bundle2.tar", "w") as tar:
    for p in sorted(DATA.iterdir()):
        tar.add(p, arcname=f"data/{p.name}")
    for name in ["evaluate.py", "train_qlora.py", "qwen_common.py", "pack_checkpoint.py", "protocol.json"]:
        tar.add(ROOT / name, arcname=name)
print(json.dumps({"fewshot": [(x["id"], x["why"], json.loads(x["answer"])) for x in fewshot],
                  "fewshot_chars": sum(len(x["state"]) for x in fewshot), "think_subset": len(test_ids),
                  "bundle2_sha256": hashlib.sha256((ROOT / "bundle2.tar").read_bytes()).hexdigest()}, indent=1, ensure_ascii=False))
