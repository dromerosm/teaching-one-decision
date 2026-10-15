"""Assemble generated text + specs into model inputs and split files (Laya format).

The model input (`state`) is the header a real triage system would show plus the email.
Twins reuse their source's subject and body with their own header.
"""
import datetime as dt
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / "dataset"
SPLITS = ("train", "validation", "calibration", "test")
INC_STATE = {"open_impact": "status: investigating; customer impact: yes",
             "monitoring": "status: fix deployed, monitoring; customer impact: no",
             "resolved": "status: resolved; customer impact: no"}

QUESTIONS = {
    "category": {"type": "choice", "criteria": {k: None for k in ["billing", "technical", "sales", "scheduling", "information"]},
                 "instructions": "Which team must act on the latest message? Decide by what the sender asks for, not by topics mentioned in passing, the subject line or quoted history. billing = invoices, charges, refunds, payment details; technical = faults, errors, incidents, access, API or security help; sales = buying, quotes, upgrades, pricing, commercial renewals; scheduling = arranging, moving or cancelling meetings, whatever the meeting is about; information = documents, policies and announcements to read, file or share. For a notice that asks for nothing, choose its area: payments or invoices = billing; systems, access or service health = technical; purchases, quotes, orders or contracts = sales; a specific meeting, visit or session = scheduling; company news, policies and documents = information."},
    "priority": {"type": "choice", "criteria": {k: None for k in ["low", "normal", "high"]},
                 "instructions": "How urgent is the latest message, using the header? high = it concerns an incident that the header lists with customer impact: yes; or it sets a deadline that falls on the same day it was received; or the account SLA is Premium and the sender says their work is blocked. low = it asks for no action (notice, receipt, closed item, marketing). normal = any other request. Tone, capital letters, the subject line and quoted history do not change priority."},
    "needs_reply": {"type": "noul",
                    "instructions": "Does the latest message ask the recipient to write back, confirm or send information? Generic courtesy lines such as 'let me know if you have any questions' do not count. Asking only for something to be done, with no written answer expected, is false. Questions addressed to someone else, such as a colleague in copy, and questions in quoted or forwarded history do not count."},
}


def header(s):
    r = dt.datetime.fromisoformat(s["received"])
    domain = s["account"].lower().replace(" ", "-").replace("í", "i").replace("é", "e").replace("ó", "o").replace("á", "a") + ".example"
    user = s["sender"].lower().replace(" ", ".")
    for a, b in [("á", "a"), ("é", "e"), ("í", "i"), ("ó", "o"), ("ú", "u"), ("ñ", "n")]:
        user = user.replace(a, b)
    inc = "; ".join(f'{i["id"]} {i["system"]} ({INC_STATE[i["state"]]})' for i in s["incidents"]) or "none"
    return (f"Received: {r:%a %d %b %Y, %H:%M}\n"
            f"From: {s['sender']} <{user}@{domain}>, {s['role']}, {s['account']}\n"
            f"Account SLA: {s['sla']}\n"
            f"Open incidents for this account: {inc}")


def main():
    specs = [json.loads(l) for l in open(ROOT / "specs.jsonl")]
    text = {}
    if not (ROOT / "gen").exists() and (ROOT / "generation_cache.jsonl").exists():
        for line in open(ROOT / "generation_cache.jsonl"):
            rec = json.loads(line)
            text[rec["id"]] = rec
    for s in specs:
        p = ROOT / "gen" / f"{s['id']}.json"
        if p.exists():
            text[s["id"]] = json.loads(p.read_text())
    rows, missing = [], 0
    for s in specs:
        src = s["twin_of"] or s["id"]
        if src not in text:
            missing += 1
            continue
        g = text[src]
        state = header(s) + f"\nSubject: {g['subject']}\n\n{g['body']}"
        rows.append({"id": s["id"], "split": s["split"], "account": s["account"], "language": s["language"],
                     "group": src, "twin_of": s["twin_of"], "twin_change": s["twin_change"],
                     "lifecycle": s["lifecycle"], "state": state, "expected": s["labels"],
                     "spec": {k: s[k] for k in ["topic", "received", "deadline", "deadline_phrase", "sla", "blocked",
                                                 "incident_ref", "incidents", "confuser", "misleading_subject",
                                                 "urgency_tone", "quoted", "register", "length"]},
                     "label_provenance": "deterministic from spec; text by gpt-6.1-sol, accepted after programmatic and LLM fidelity checks"})
    OUT.mkdir(exist_ok=True)
    (OUT / "questions.json").write_text(json.dumps(QUESTIONS, ensure_ascii=False, indent=2) + "\n")
    (OUT / "records.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    for sp in SPLITS:
        sel = [r for r in rows if r["split"] == sp]
        (OUT / f"{sp}.jsonl").write_text("".join(json.dumps({"state": r["state"], "questions": QUESTIONS, "expected": r["expected"]}, ensure_ascii=False) + "\n" for r in sel))
    made = ["questions.json", "records.jsonl"] + [f"{sp}.jsonl" for sp in SPLITS]
    sums = {n: hashlib.sha256((OUT / n).read_bytes()).hexdigest() for n in sorted(made)}
    (OUT / "ASSEMBLED.sha256").write_text("".join(f"{v}  {k}\n" for k, v in sums.items()))
    print(f"{len(rows)} rows assembled, {missing} specs without text yet")


if __name__ == "__main__":
    main()
