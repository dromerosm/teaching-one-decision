"""Collect every result of the realistic email-triage benchmark into one record.

Reads the per-row prediction files and training reports already on disk and writes:
  results_all.json  machine-readable record (one entry per model and variant)
  RESULTS.md        tables for the article: accuracy with account-bootstrap intervals, per-field
                    accuracy, high-priority recall and false alarms, twin-pair consistency, training
                    time and data seen, inference latency, load time, peak GPU memory, hardware.
Run again whenever new results arrive. Missing files are listed, never invented.
"""
import collections
import json
import random
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent          # round2/results
DATA = ROOT.parent / "dataset"
PRED = ROOT / "predictions"
TRAIN = ROOT / "training"
FIELDS = ["category", "priority", "needs_reply"]
recs = {json.loads(l)["id"]: json.loads(l) for l in open(DATA / "records.jsonl")}
think_ids = set(json.loads((DATA / "think_subset.json").read_text()))

T4, GB10 = "NVIDIA T4 (Colab)", "NVIDIA GB10 (DGX Spark)"
D3_SIZES = {"d3-edge": "0.59B", "d3-lite": "0.85B", "d3-nano": "2.2B", "d3-mini": "4.5B", "d3-flash": "8.4B", "d3": "26.1B"}

# key: (label, family, size, variant, file, hardware, training-report file or None)
ENTRIES = [
    ("Laya, as published", "Laya", "0.32B", "zero-shot", "laya_original.json", T4, None),
    ("Laya, fine-tuned", "Laya", "0.32B", "full fine-tune, 4 epochs", "laya_trained.json", T4, "laya_train_report.json"),
    ("Qwen3.5-9B, original prompt", "Qwen", "9B (4-bit)", "zero-shot", "qwen.json", T4, None),
    ("Qwen3.5-9B, prompt v2", "Qwen", "9B (4-bit)", "prompt v2", "qwen_prompt.json", T4, None),
    ("Qwen3.5-9B, prompt v2 + 8 examples", "Qwen", "9B (4-bit)", "few-shot", "qwen_fewshot.json", T4, None),
    ("Qwen3.5-9B, prompt v2 + thinking", "Qwen", "9B (4-bit)", "thinking (partial)", "qwen_think.json", T4, None),
    ("Qwen3.5-9B, QLoRA 448 emails", "Qwen", "9B (4-bit)", "QLoRA, 0.23 epoch", "qwen_qlora.json", T4, "qwen_qlora_448_report.json"),
    ("Qwen3.5-9B, QLoRA full epoch", "Qwen", "9B (4-bit)", "QLoRA, 1 epoch", "qwen_qlora_full.json", T4, "qwen_qlora_full_report.json"),
]
for m, size in D3_SIZES.items():
    mode = "full fine-tune, 4 epochs" if m in ("d3-edge", "d3-lite", "d3-nano") else "LoRA, 1 epoch"
    ENTRIES += [
        (f"{m}, as published", "Decision 3.0", size, "zero-shot", f"d3/{m}.json", GB10, None),
        (f"{m}, prompt v2", "Decision 3.0", size, "prompt v2", f"d3/{m}-v2.json", GB10, None),
        (f"{m}, prompt v2 + 8 examples", "Decision 3.0", size, "few-shot", f"d3/{m}-fewshot.json", GB10, None),
        (f"{m}, fine-tuned", "Decision 3.0", size, mode, f"d3/{m}-ft.json", GB10, f"{m}_train_report.json"),
    ]


def fix_d3(row):
    """d3 answers yes/no with one probability ({"noul": p}); d3_eval.py v1 did not read it."""
    raw = row.get("raw", {}).get("needs_reply", {})
    if row["prediction"].get("needs_reply") is None and "noul" in raw:
        row["prediction"]["needs_reply"] = raw["noul"] >= 0.5


def ok(r):
    return all(r["prediction"].get(f) == r["expected"][f] for f in FIELDS)


def bootstrap(rows, n=5000, seed=42):
    by = collections.defaultdict(list)
    for r in rows:
        by[recs[r["id"]]["account"]].append(r)
    accs = sorted(by); rng = random.Random(seed); vals = []
    for _ in range(n):
        sample = [r for _ in accs for r in by[rng.choice(accs)]]
        vals.append(sum(map(ok, sample)) / len(sample))
    vals.sort()
    return [round(vals[int(0.025 * n)], 4), round(vals[int(0.975 * n)], 4)]


def training_info(path):
    p = TRAIN / path
    if not p.exists():
        return None
    t = json.loads(p.read_text())
    if "training_with_validation_seconds" in t:  # Laya
        return {"seconds": t["training_with_validation_seconds"], "data_seen": f"{t['row_counts']['train']} emails x 3 fields x 4 epochs",
                "peak_gib": t["peak_training_gpu_bytes"] / 2**30, "selected_epoch": t["selected_epoch"], "includes_validation": True}
    if "examples_seen" in t:  # Qwen QLoRA
        return {"seconds": t.get("training_seconds"), "data_seen": f"{t['examples_seen']} emails x 1 pass",
                "peak_gib": (t.get("peak_training_gpu_bytes") or 0) / 2**30, "trainable_parameters": t.get("trainable_parameters"),
                "status": t.get("status"), "includes_validation": False}
    if t.get("mode"):  # d3
        return {"seconds": t.get("training_seconds"), "data_seen": f"{t.get('train_rows')} emails, {t.get('train_items')} items per epoch x {len(t.get('epochs', []))} epoch(s)",
                "peak_gib": (t.get("peak_training_gpu_bytes") or 0) / 2**30, "trainable_parameters": t.get("trainable_parameters"),
                "selected_epoch": t.get("selected_epoch"), "status": t.get("status"), "includes_validation": True,
                "self_check_max_abs_diff": t.get("self_check_max_abs_diff"), "stopped_early_by_budget": t.get("stopped_early_by_budget")}
    return None


SKIPPED = {
    "d3/d3-ft.json": "Not run, by decision: the 26B model already scores 98.5% zero-shot (95% interval 97.5-99.5%), "
    "at the label-consistency ceiling (99.5%). A fine-tune could add at most ~1.5 points on 398 emails, inside the "
    "interval, so no gain could be shown, and it would take about 5 GPU hours (likely a partial epoch under the cap).",
}
out, missing = [], []
for label, family, size, variant, fname, hw, trep in ENTRIES:
    p = PRED / fname
    if not p.exists():
        if fname not in SKIPPED:
            missing.append(fname)
        continue
    d = json.loads(p.read_text())
    rows = d["results"]
    if family == "Decision 3.0":
        for r in rows:
            fix_d3(r)
    hi = [r for r in rows if r["expected"]["priority"] == "high"]
    ids = {r["id"]: r for r in rows}
    pairs = [(recs[i]["twin_of"], i) for i in ids if recs[i]["twin_of"] in ids]
    e = {"label": label, "family": family, "size": size, "variant": variant, "file": fname, "hardware": hw,
         "n": len(rows), "complete": d.get("status") == "complete" or len(rows) == 398,
         "all_fields": round(sum(map(ok, rows)) / len(rows), 4),
         "ci95_account_bootstrap": bootstrap(rows) if len(rows) >= 300 else None,
         **{f: round(sum(r["prediction"].get(f) == r["expected"][f] for r in rows) / len(rows), 4) for f in FIELDS},
         "high_caught": f"{sum(r['prediction'].get('priority') == 'high' for r in hi)} of {len(hi)}",
         "high_false_alarms": sum(r["prediction"].get("priority") == "high" and r["expected"]["priority"] != "high" for r in rows),
         "twin_pairs_both_priority_correct": round(sum(ids[a]["prediction"].get("priority") == ids[a]["expected"]["priority"] and
                                                        ids[b]["prediction"].get("priority") == ids[b]["expected"]["priority"] for a, b in pairs) / max(1, len(pairs)), 4) if pairs else None,
         "median_seconds_per_email": round(statistics.median(r["seconds"] for r in rows), 4),
         "load_seconds": round(d.get("load_seconds", 0) or 0, 1),
         "peak_inference_gib": round((d.get("peak_allocated_bytes") or 0) / 2**30, 2) or None,
         "training": training_info(trep) if trep else None}
    sub = [r for r in rows if r["id"] in think_ids]
    e["on_thinking_subset_44"] = None
    out.append(e)

(ROOT / "results_all.json").write_text(json.dumps({"entries": out, "missing": missing, "skipped": SKIPPED}, indent=1, ensure_ascii=False) + "\n")


def pct(x):
    return "—" if x is None else f"{100 * x:.1f}%"


def mins(s):
    return "—" if not s else (f"{s / 60:.0f} min" if s >= 60 else f"{s:.1f} s")


lines = ["# Results record: realistic email-triage benchmark", "",
         "Generated by `collect_results.py` from per-row predictions and training reports. Test: 398 emails, 10 held-out accounts. "
         "Intervals: 95% bootstrap over test accounts (5,000 resamples). Latency: median per email after loading, one email at a time. "
         "Timings on different hardware are not directly comparable.", "",
         "## Accuracy", "", "| Model | Variant | Hardware | n | All three | 95% interval | Category | Priority | Needs reply | High caught | False alarms | Twin pairs |",
         "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
for e in out:
    ci = "—" if not e["ci95_account_bootstrap"] else f"{100 * e['ci95_account_bootstrap'][0]:.1f}–{100 * e['ci95_account_bootstrap'][1]:.1f}%"
    lines.append(f"| {e['label']} | {e['variant']} | {e['hardware']} | {e['n']} | {pct(e['all_fields'])} | {ci} | {pct(e['category'])} | {pct(e['priority'])} | "
                 f"{pct(e['needs_reply'])} | {e['high_caught']} | {e['high_false_alarms']} | {pct(e['twin_pairs_both_priority_correct'])} |")
lines += ["", "## Cost: training and inference", "", "| Model | Variant | Hardware | Training time | Data seen | Trainable params | Peak train GiB | Latency per email | Load | Peak inference GiB |",
          "|---|---|---|---:|---|---:|---:|---:|---:|---:|"]
for e in out:
    t = e["training"] or {}
    tp = t.get("trainable_parameters")
    lines.append(f"| {e['label']} | {e['variant']} | {e['hardware']} | {mins(t.get('seconds'))} | {t.get('data_seen', '—')} | "
                 f"{'—' if not tp else f'{tp / 1e6:.0f}M'} | {'—' if not t.get('peak_gib') else round(t['peak_gib'], 1)} | "
                 f"{e['median_seconds_per_email']} s | {e['load_seconds']} s | {e['peak_inference_gib'] or '—'} |")
lines += ["", "## Not run, by decision", ""] + [f"- `{k}`: {v}" for k, v in SKIPPED.items()]
lines += ["", "## Not yet available", ""] + ([f"- `{m}`" for m in missing] or ["- none"])
(ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n")
print(f"{len(out)} entries, {len(missing)} missing")
