"""Evaluation of a Decision 3.0 model on the realistic email test (runs on the DGX Spark). v2.

v2 adds a questions file argument (prompt v2 / few-shot variants), accepts a local fine-tuned
package (revision "local"), and reads yes/no answers from their single probability ({"noul": p}).

Same inputs as every other model: the 398 test states (header + email) and the three
questions from questions.json, whose instructions are the written policy. Output has the
same shape as evaluate.py results, so analyze.py can read it.

Usage: python d3_eval2.py <hub id or dir> <revision|local> <out.json> <data dir> [questions file name]
"""
import hashlib, importlib.metadata, importlib.util, json, os, statistics, sys, time, traceback
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import torch
from transformers import AutoModel

repo, revision, out_path = sys.argv[1], sys.argv[2], Path(sys.argv[3])
data_dir = Path(sys.argv[4]) if len(sys.argv) > 4 else Path(__file__).resolve().parent / "data"
emails = json.loads((data_dir / "emails.json").read_text())
qfile = sys.argv[5] if len(sys.argv) > 5 else "questions.json"
questions = json.loads((data_dir / qfile).read_text())
FIELDS = ["category", "priority", "needs_reply"]
meta = {"model": repo, "revision": revision, "n": len(emails), "started_at": time.time(),
        "gpu": torch.cuda.get_device_name(0), "versions": {p: importlib.metadata.version(p) for p in ["torch", "transformers"]},
        "fla_installed": importlib.util.find_spec("fla") is not None,
        "dataset_sha256": hashlib.sha256((data_dir / "emails.json").read_bytes()).hexdigest(),
        "questions_file": qfile, "questions_sha256": hashlib.sha256((data_dir / qfile).read_bytes()).hexdigest(), "results": []}


def save():
    out_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str))


def pick(answer):
    if "noul" in answer and answer["noul"] is not None:
        p = float(answer["noul"])
        return ("true" if p >= 0.5 else "false"), max(p, 1 - p)
    probs = answer.get("probabilities") or {}
    if not probs:
        return None, None
    key = max(probs, key=probs.get)
    return key, probs[key]


try:
    t = time.perf_counter()
    kw = {} if revision == "local" else {"revision": revision}
    model = AutoModel.from_pretrained(repo, trust_remote_code=True, device="cuda:0", **kw)
    torch.cuda.synchronize(); meta["load_seconds"] = time.perf_counter() - t
    meta["resident_allocated_bytes"] = torch.cuda.memory_allocated()
    t = time.perf_counter()
    model.system_one(state=emails[0]["state"], questions=questions)  # warm-up (kernel compilation)
    torch.cuda.synchronize(); meta["warmup_seconds"] = time.perf_counter() - t
    torch.cuda.reset_peak_memory_stats()
    for d in emails:
        torch.cuda.synchronize(); t = time.perf_counter()
        out = model.system_one(state=d["state"], questions=questions)
        torch.cuda.synchronize(); el = time.perf_counter() - t
        pred, conf, raw = {}, {}, out.get("answers", {})
        for f in FIELDS:
            key, p = pick(raw.get(f, {}))
            pred[f] = (key == "true") if f == "needs_reply" and key is not None else key
            conf[f] = p
        meta["results"].append({"id": d["id"], "language": d["language"], "twin_of": d.get("twin_of"),
                                "expected": d["expected"], "prediction": pred, "confidence": conf,
                                "seconds": el, "input_tokens": out.get("usage", {}).get("input_tokens"),
                                "raw": raw})
        if len(meta["results"]) % 50 == 0:
            save(); print(f"{repo}: {len(meta['results'])}/{len(emails)}", flush=True)
    r = meta["results"]
    meta["accuracy"] = {f: sum(x["prediction"][f] == x["expected"][f] for x in r) / len(r) for f in FIELDS}
    meta["accuracy"]["all_fields"] = sum(all(x["prediction"][f] == x["expected"][f] for f in FIELDS) for x in r) / len(r)
    meta["inference_call_seconds"] = sum(x["seconds"] for x in r)
    meta["latency_median_s"] = statistics.median(x["seconds"] for x in r)
    meta["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    meta["status"] = "complete"; meta["finished_at"] = time.time(); save()
    print(json.dumps({k: meta[k] for k in ["model", "accuracy", "inference_call_seconds", "latency_median_s", "load_seconds"]}), flush=True)
except Exception:
    meta["status"] = "failed"; meta["error"] = traceback.format_exc(); save(); print(meta["error"], flush=True); sys.exit(1)
