"""Difficulty and leakage audit for the assembled dataset.

1. Naive Bayes word-count baseline (same method as the earlier benchmark's baseline).
   With full splits: fit on train, score validation and test.
   With --cv (small pilots): leave-one-account-out cross-validation over all rows.
2. Keyword heuristics written without looking at labels: urgency words, request cues,
   topic keywords. No training.
3. Counterfactual twins: share of rows whose twin has identical text but another label.
4. Template overlap: share of test sentences that appear verbatim in train.
"""
import argparse
import collections
import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FIELDS = ["category", "priority", "needs_reply"]
spec = importlib.util.spec_from_file_location("nb", ROOT.parent.parent / "round1" / "audit" / "baseline_naive_bayes.py")
nb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nb)

URGENT = re.compile(r"\b(urgent|urgente|asap|immediately|inmediatamente|today|hoy|blocked|bloquead\w*|cannot proceed|can't proceed|no podemos (avanzar|continuar|seguir)|outage|caída|down|incident|incidencia|INC-\d+|critical|crítico)\b", re.I)
REQUEST = re.compile(r"\b(please|could you|can you|would you|kindly|por favor|podrías|podéis|puedes|necesitamos que|os pido|te pido)\b|\?", re.I)
REPLY = re.compile(r"\?|\b(confirm|let me know|reply|respond|send (me|us)|get back|advise|confirma\w*|indica\w*|respond\w*|envia\w*|envía\w*|dime|decidme|hacedme saber|házmelo saber)\b", re.I)
NO_REPLY = re.compile(r"no need to (reply|respond|write|confirm)|don't need a reply|no reply (is )?needed|no hace falta (responder|contestar|confirmar)|sin necesidad de responder|no es necesario (responder|contestar)", re.I)
TOPIC = {
    "billing": r"invoice|factura|refund|reembolso|charge|cargo|payment|pago|vat|iva|credit note|abono|billing|facturaci",
    "technical": r"error|bug|login|api|outage|caída|crash|certificate|certificado|timeout|webhook|access|acceso|sso|password|contraseña|fallo",
    "sales": r"quote|presupuesto|pricing|precio|upgrade|renewal|renovaci|licen|discount|descuento|contract|contrato|plan",
    "scheduling": r"meeting|reunión|call|llamada|reschedule|calendar|calendario|session|sesión|workshop|taller|slot|invite|invitaci",
    "information": r"policy|política|document|documento|announcement|anuncio|newsletter|guideline|guía|organi[sz]ation chart|organigrama|file|archiv",
}


def body_only(state):
    return state.split("\nSubject: ", 1)[1]


def heuristic(state):
    t = body_only(state)
    latest = t.split("\n>", 1)[0]
    scores = {c: len(re.findall(p, latest, re.I)) for c, p in TOPIC.items()}
    cat = max(scores, key=scores.get) if any(scores.values()) else "information"
    if URGENT.search(latest):
        pri = "high"
    elif REQUEST.search(latest):
        pri = "normal"
    else:
        pri = "low"
    reply = bool(REPLY.search(latest)) and not NO_REPLY.search(latest)
    return {"category": cat, "priority": pri, "needs_reply": str(reply)}


def score(rows, preds):
    exp = [{f: str(r["expected"][f]) for f in FIELDS} for r in rows]
    return {"n": len(rows),
            "all_fields": round(sum(p == e for p, e in zip(preds, exp)) / len(rows), 4),
            **{f: round(sum(p[f] == e[f] for p, e in zip(preds, exp)) / len(rows), 4) for f in FIELDS}}


def nb_fit_predict(train, test):
    models = {f: nb.NB().fit([r["state"] for r in train], [str(r["expected"][f]) for r in train]) for f in FIELDS}
    return [{f: models[f].predict(r["state"]) for f in FIELDS} for r in test]


def sentences(t):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", t) if len(s.strip()) > 25]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv", action="store_true")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(ROOT.parent / "dataset" / "records.jsonl")]
    out = {"rows": len(rows), "label_distribution": {f: dict(collections.Counter(str(r["expected"][f]) for r in rows)) for f in FIELDS}}
    majority = {f: collections.Counter(str(r["expected"][f]) for r in rows).most_common(1)[0][0] for f in FIELDS}
    out["majority_class"] = score(rows, [dict(majority) for _ in rows])
    out["keyword_heuristic"] = score(rows, [heuristic(r["state"]) for r in rows])
    out["header_heuristic"] = score(rows, [header_heuristic(r["state"]) for r in rows])
    if a.cv:
        preds = {}
        for acc in sorted({r["account"] for r in rows}):
            tr = [r for r in rows if r["account"] != acc]
            te = [r for r in rows if r["account"] == acc]
            for r, p in zip(te, nb_fit_predict(tr, te)):
                preds[r["id"]] = p
        out["naive_bayes_leave_one_account_out"] = score(rows, [preds[r["id"]] for r in rows])
    else:
        tr = [r for r in rows if r["split"] == "train"]
        for sp in ["validation", "test"]:
            te = [r for r in rows if r["split"] == sp]
            out[f"naive_bayes_{sp}"] = score(te, nb_fit_predict(tr, te))
            out[f"keyword_heuristic_{sp}"] = score(te, [heuristic(r["state"]) for r in te])
            out[f"header_heuristic_{sp}"] = score(te, [header_heuristic(r["state"]) for r in te])
        train_sents = set(s for r in tr for s in sentences(body_only(r["state"])))
        test_sents = [s for r in rows if r["split"] == "test" for s in sentences(body_only(r["state"]))]
        out["test_sentences_verbatim_in_train"] = round(sum(s in train_sents for s in test_sents) / max(1, len(test_sents)), 4)
    ids = {r["id"]: r for r in rows}
    twins = [r for r in rows if r["twin_of"] and r["twin_of"] in ids]
    out["twin_pairs"] = len(twins)
    out["twin_changes"] = dict(collections.Counter(r["twin_change"] for r in twins))
    out["max_text_only_priority_accuracy_on_twin_rows"] = 0.5 if twins else None
    (ROOT / ("audit_cv.json" if a.cv else "audit.json")).write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(out, indent=2, ensure_ascii=False))



WD = {"en": ["monday", "tuesday", "wednesday", "thursday", "friday"], "es": ["lunes", "martes", "miércoles", "jueves", "viernes"]}
MON = {"en": ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
       "es": ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]}
BLOCKED = re.compile(r"blocked|cannot (proceed|continue|move forward)|can't (proceed|continue|move forward)|on hold until|stuck|bloquead|no podemos (avanzar|continuar|seguir)|paralizad|parad[oa]s?", re.I)


def header_heuristic(state):
    """Surface rules that mimic the priority policy: header flags + same-day word matching."""
    h, t = state.split("\nSubject: ", 1)
    latest = t.split("\n>", 1)[0].lower()
    m = re.search(r"Received: (\w+) (\d+) (\w+)", h)
    wd, day, mon = m.group(1).lower(), int(m.group(2)), m.group(3).lower()
    wd_i = ["mon", "tue", "wed", "thu", "fri"].index(wd[:3]) if wd[:3] in ["mon", "tue", "wed", "thu", "fri"] else -1
    p = heuristic(state)
    high = False
    for inc in re.finditer(r"(INC-\d+) ([^(;]+) \(status: [^;]+; customer impact: yes\)", h):
        if inc.group(1).lower() in latest or inc.group(2).strip().lower().split()[0] in latest:
            high = True
    if re.search(r"\b(today|hoy|end of day|close of business|cierre de hoy)\b", latest):
        high = True
    if wd_i >= 0 and re.search(rf"\b({WD['en'][wd_i]}|{WD['es'][wd_i]})\b", latest):
        high = True
    mi = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"].index(mon[:3])
    if re.search(rf"\b{day} (de )?({MON['en'][mi]}\w*|{MON['es'][mi]})\b", latest):
        high = True
    if "SLA: Premium" in h and BLOCKED.search(latest):
        high = True
    if high:
        p["priority"] = "high"
    elif p["priority"] == "high":
        p["priority"] = "normal"
    return p



if __name__ == "__main__":
    main()
