"""Label-consistency check: can a strong model recover the labels from the written policy alone?

gpt-6.1-sol receives only the model input (header + email) and the three field
instructions from questions.json, never the spec. High agreement means the labels
follow from the visible text and policy; low agreement would mean ambiguity or noise.
Usage: python3 ceiling.py --n 100
"""
import argparse
import collections
import concurrent.futures as cf
import json
import random
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "generator"))
import generate

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--split", default=None)
    a = ap.parse_args()
    q = json.loads((ROOT.parent / "dataset" / "questions.json").read_text())
    rows = [json.loads(l) for l in open(ROOT.parent / "dataset" / "records.jsonl")]
    if a.split:
        rows = [r for r in rows if r["split"] == a.split]
    rng = random.Random(a.seed)
    sample = rng.sample(rows, min(a.n, len(rows))) if a.n else rows
    schema = {"type": "object", "additionalProperties": False, "required": ["category", "priority", "needs_reply"],
              "properties": {"category": {"type": "string", "enum": list(q["category"]["criteria"])},
                             "priority": {"type": "string", "enum": list(q["priority"]["criteria"])},
                             "needs_reply": {"type": "boolean"}}}
    system = ("You triage emails received by the shared inbox of Helix Systems, a B2B software vendor. "
              "Apply these definitions exactly.\n\n" + "\n\n".join(f"{k}: {v['instructions']}" for k, v in q.items()))

    def one(r):
        out, _ = generate.call([{"role": "system", "content": system}, {"role": "user", "content": r["state"]}], schema, "triage")
        return r, out

    res = []
    with cf.ThreadPoolExecutor(12) as ex:
        for r, out in ex.map(one, sample):
            res.append({"id": r["id"], "expected": r["expected"], "predicted": out,
                        "twin": bool(r["twin_of"]), "lifecycle": r["lifecycle"]})
    f = ["category", "priority", "needs_reply"]
    acc = {k: sum(x["expected"][k] == x["predicted"][k] for x in res) / len(res) for k in f}
    allf = sum(all(x["expected"][k] == x["predicted"][k] for k in f) for x in res) / len(res)
    conf = {k: collections.Counter(f"{x['expected'][k]}->{x['predicted'][k]}" for x in res if x["expected"][k] != x["predicted"][k]) for k in f}
    summary = {"n": len(res), "all_fields": round(allf, 4), **{k: round(v, 4) for k, v in acc.items()},
               "errors": {k: dict(v) for k, v in conf.items()}, "usd_spent_this_run": round(generate.SPEND["usd"], 4)}
    (ROOT / f"ceiling{'_' + a.split if a.split else ''}{'_all' if not a.n else ''}.json").write_text(json.dumps({"summary": summary, "rows": res}, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
