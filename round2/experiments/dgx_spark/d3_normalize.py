"""Normalise d3 results copied from the DGX into the evaluate.py shape used by analyze.py.

d3 answers a yes/no ("noul") question with a single probability of "true" ({"noul": p}),
not with per-option probabilities. d3_eval.py v1 read only per-option probabilities, so its
needs_reply predictions were empty. The raw answers were saved, so this script recomputes
needs_reply as p >= 0.5 and the accuracy fields. Category and priority are unchanged.

Usage: python3 d3_normalize.py  (reads d3_results/*.json, writes d3_<name>.json next to it)
"""
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FIELDS = ["category", "priority", "needs_reply"]

for src in sorted((ROOT / "d3_results").glob("*.json")):
    d = json.loads(src.read_text())
    if d.get("status") != "complete":
        print("skip (not complete):", src.name)
        continue
    for x in d["results"]:
        p = x["raw"]["needs_reply"].get("noul")
        x["prediction"]["needs_reply"] = None if p is None else p >= 0.5
        x["confidence"]["needs_reply"] = None if p is None else max(p, 1 - p)
    r = d["results"]
    d["accuracy"] = {f: sum(x["prediction"][f] == x["expected"][f] for x in r) / len(r) for f in FIELDS}
    d["accuracy"]["all_fields"] = sum(all(x["prediction"][f] == x["expected"][f] for f in FIELDS) for x in r) / len(r)
    d["valid_outputs"] = sum(all(x["prediction"][f] is not None for f in FIELDS) for x in r)
    d["latency_median_s"] = statistics.median(x["seconds"] for x in r)
    d["normalisation"] = "needs_reply recomputed from raw noul probability (>= 0.5 is true) by d3_normalize.py"
    name = src.stem.replace("-", "_")
    (ROOT / f"{name}.json").write_text(json.dumps(d, ensure_ascii=False, indent=1))
    print(name, {k: round(v, 4) for k, v in d["accuracy"].items()}, "median_s", round(d["latency_median_s"], 4),
          "peak_GiB", round(d.get("peak_allocated_bytes", 0) / 2**30, 2), "load_s", round(d.get("load_seconds", 0), 1))
