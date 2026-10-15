"""Package the realistic dataset for the Colab run. Local only; no GPU, no API.

- data/: split files, records, questions, assignment, and schema.json built from the
  same field instructions Laya uses, so Qwen receives an identical written policy.
- data/emails.json: the test rows evaluate.py reads (id, language, state, expected, twin_of).
- protocol.json: recipe, data hashes and the lexical baselines measured before training.
- bundle.tar: everything the remote side needs.
"""
import hashlib
import json
import shutil
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
R2 = ROOT.parent.parent  # round2/
DATA = ROOT / "data"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    DATA.mkdir(exist_ok=True)
    for name in ["train.jsonl", "validation.jsonl", "calibration.jsonl", "test.jsonl", "records.jsonl", "questions.json"]:
        shutil.copy2(R2 / "dataset" / name, DATA / name)
    shutil.copy2(R2 / "generator" / "assignment.json", DATA / "assignment.json")
    q = json.loads((DATA / "questions.json").read_text())
    schema = {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object",
              "properties": {
                  "category": {"type": "string", "enum": list(q["category"]["criteria"]), "description": q["category"]["instructions"]},
                  "priority": {"type": "string", "enum": list(q["priority"]["criteria"]), "description": q["priority"]["instructions"]},
                  "needs_reply": {"type": "boolean", "description": q["needs_reply"]["instructions"]}},
              "required": ["category", "priority", "needs_reply"], "additionalProperties": False}
    (DATA / "schema.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2) + "\n")
    rows = [json.loads(l) for l in open(DATA / "records.jsonl")]
    test = [{"id": r["id"], "language": r["language"], "state": r["state"], "expected": r["expected"],
             "twin_of": r["twin_of"], "account": r["account"]} for r in rows if r["split"] == "test"]
    (DATA / "emails.json").write_text(json.dumps(test, ensure_ascii=False, indent=1) + "\n")
    audit = json.loads((R2 / "audit" / "audit.json").read_text())
    protocol = {
        "base_laya_revision": "1720e3e3357cfe1e281542e223f8273b0890ca34",
        "qwen_revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        "laya_version": "0.4.2", "epochs": 4, "seed": 42, "loss": "soft-ce", "micro_batch": 8, "grad_accum": 4,
        "full_finetune": True, "recipe_identical_to": "laya-t4-finetune-20261011",
        "model_selection": "validation exact-all-fields then per-decision accuracy; earlier epoch on ties",
        "calibration": "explicit calibration split only",
        "split_unit": "account (50/10/10/10); asks and notices also split-disjoint",
        "row_counts": {s: sum(r["split"] == s for r in rows) for s in ["train", "validation", "calibration", "test"]},
        "qwen_changes_vs_previous_run": ["flash-linear-attention and causal-conv1d installed (fast kernels asserted)",
                                         "SDPA attention instead of eager",
                                         "schema descriptions are the same policy text Laya receives"],
        "pre_training_baselines": {k: audit[k] for k in audit if k.startswith(("naive_bayes", "keyword", "header", "majority"))},
        "data_sha256": {p.name: sha(p) for p in sorted(DATA.iterdir()) if p.is_file()},
    }
    (ROOT / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n")
    with tarfile.open(ROOT / "bundle.tar", "w") as tar:
        for p in sorted(DATA.iterdir()):
            tar.add(p, arcname=f"data/{p.name}")
        for name in ["train_remote.py", "evaluate.py", "pack_checkpoint.py", "protocol.json"]:
            tar.add(ROOT / name, arcname=name)
    print(json.dumps({"rows": protocol["row_counts"], "test_rows": len(test), "bundle_sha256": sha(ROOT / "bundle.tar")}, indent=2))


if __name__ == "__main__":
    main()
