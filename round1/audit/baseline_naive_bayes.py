"""No-GPU lexical baseline for the email-triage benchmark.

Multinomial Naive Bayes per field on word unigrams + bigrams of the same
`state` text the models received (digits normalised to N, add-one smoothing,
tokens unseen in training ignored). Fitted on train.jsonl only; no tuning.

Also reports a "keyword rule" variant where priority=high is set by a single
regex for the urgency phrasing used by the generator.

Usage: python3 -I baseline_naive_bayes.py  -> writes baseline_naive_bayes.json
"""
import collections
import json
import math
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIELDS = ["category", "priority", "needs_reply"]
HIGH_RULE = re.compile(r"\b\d+ (minutes|minutos)\b|live service|servicio real")


def toks(text):
    w = re.findall(r"\w+", re.sub(r"\d+", "N", text.lower()))
    return w + [a + "_" + b for a, b in zip(w, w[1:])]


class NB:
    def fit(self, texts, labels):
        self.prior = collections.Counter(labels)
        self.words = collections.defaultdict(collections.Counter)
        self.vocab = set()
        for t, y in zip(texts, labels):
            for w in toks(t):
                self.words[y][w] += 1
                self.vocab.add(w)
        self.total = {y: sum(self.words[y].values()) for y in self.prior}
        return self

    def predict(self, text):
        ts = [w for w in toks(text) if w in self.vocab]
        v = len(self.vocab)
        return max(self.prior, key=lambda y: math.log(self.prior[y]) + sum(
            math.log((self.words[y][w] + 1) / (self.total[y] + v)) for w in ts))


def load(split):
    return [json.loads(l) for l in open(HERE.parent / "dataset" / f"{split}.jsonl")]


def score(rows, preds):
    exp = [{f: str(r["expected"][f]) for f in FIELDS} for r in rows]
    return {
        "all_fields_accuracy": sum(p == e for p, e in zip(preds, exp)) / len(rows),
        "per_field_accuracy": {f: sum(p[f] == e[f] for p, e in zip(preds, exp)) / len(rows) for f in FIELDS},
    }


def main():
    train = load("train")
    models = {f: NB().fit([r["state"] for r in train], [str(r["expected"][f]) for r in train]) for f in FIELDS}
    out = {"method": __doc__.strip().splitlines()[0], "fit_on": "train.jsonl", "results": {}}
    for split in ["validation", "test"]:
        rows = load(split)
        preds = [{f: models[f].predict(r["state"]) for f in FIELDS} for r in rows]
        ruled = []
        for r, p in zip(rows, preds):
            q = dict(p)
            if HIGH_RULE.search(r["state"]):
                q["priority"] = "high"
            elif q["priority"] == "high":
                q["priority"] = "normal"
            ruled.append(q)
        out["results"][split] = {"naive_bayes": score(rows, preds), "naive_bayes_plus_high_rule": score(rows, ruled)}
    all_rows = [r for s in ["train", "validation", "calibration", "test"] for r in load(s)]
    hits = [bool(HIGH_RULE.search(r["state"])) for r in all_rows]
    high = [r["expected"]["priority"] == "high" for r in all_rows]
    out["high_rule_check"] = {
        "rows": len(all_rows),
        "rule_matches": sum(hits),
        "matches_that_are_high": sum(h and y for h, y in zip(hits, high)),
        "high_rows_total": sum(high),
    }
    (HERE / "baseline_naive_bayes.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
