"""Independent review of the round-two dataset generation.

Does not import the generator's label code. Every check re-derives what it needs from the
published files (records.jsonl, specs.jsonl, generation_cache.jsonl, scenarios.py) and from the
model input text itself. Writes generation_review.json and exits non-zero if a hard check fails.

Checks
  A. Labels: priority and needs_reply recomputed from the header as the model sees it, plus the
     spec; category traced back to the hand-written request and notice lists.
  B. Header vs spec: received time, SLA and incidents in every input match the spec.
  C. Deadline phrases: present verbatim; weekday, day, month and time agree with the spec deadline;
     relative phrases agree with the received date.
  D. Twins: identical subject and body, header differs only in the declared field, labels differ
     only in priority, same split; absolute deadlines only, within 1-6 days.
  E. Fidelity: every email comes from an accepted generation whose programmatic and LLM checks passed.
  F. Splits: accounts, requests and notices disjoint; no exact duplicate text across groups;
     cross-split near-duplicates by word 5-gram Jaccard.
  G. Balance: label distributions by split and language.
"""
import collections
import datetime as dt
import importlib.util
import json
import re
import sys
from pathlib import Path

R2 = Path(__file__).resolve().parent.parent
recs = [json.loads(l) for l in open(R2 / "dataset" / "records.jsonl")]
specs = {json.loads(l)["id"]: json.loads(l) for l in open(R2 / "generator" / "specs.jsonl")}
cache = {json.loads(l)["id"]: json.loads(l) for l in open(R2 / "generator" / "generation_cache.jsonl")}
assignment = json.loads((R2 / "generator" / "assignment.json").read_text())
sc_spec = importlib.util.spec_from_file_location("scenarios", R2 / "generator" / "scenarios.py")
sc = importlib.util.module_from_spec(sc_spec); sc_spec.loader.exec_module(sc)

out = {"rows": len(recs), "checks": {}, "hard_failures": [], "notes": []}


def check(name, failures, hard=True, detail=None):
    out["checks"][name] = {"failures": len(failures), "examples": failures[:5], **({"detail": detail} if detail else {})}
    if failures and hard:
        out["hard_failures"].append(name)


MON = {m: i + 1 for i, m in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])}
INC = re.compile(r"(INC-\d+) (.+?) \(status: ([^;]+); customer impact: (yes|no)\)")


def parse_header(state):
    head, rest = state.split("\nSubject: ", 1)
    m = re.search(r"Received: \w{3} (\d{2}) (\w{3}) (\d{4}), (\d{2}):(\d{2})", head)
    received = dt.datetime(int(m.group(3)), MON[m.group(2)], int(m.group(1)), int(m.group(4)), int(m.group(5)))
    sla = re.search(r"Account SLA: (\w+)", head).group(1)
    line = re.search(r"Open incidents for this account: (.*)", head).group(1)
    incidents = [] if line == "none" else [{"id": a, "system": b, "status": c, "impact": d == "yes"} for a, b, c, d in INC.findall(line)]
    subject, body = rest.split("\n\n", 1) if "\n\n" in rest else (rest, "")
    return received, sla, incidents, subject, body


# ---- A. labels -----------------------------------------------------------------
fail_pri, fail_reply, fail_cat, fail_cat_trace = [], [], [], []
for r in recs:
    s = specs[r["id"]]
    received, sla, incidents, _, _ = parse_header(r["state"])
    impacted = {i["system"] for i in incidents if i["impact"]}
    about_impacted = s["incident_ref"] and s["incidents"] and s["incidents"][0]["system"] in impacted
    if about_impacted:
        pri = "high"
    elif s["lifecycle"] == "notice":
        pri = "low"
    elif s["deadline"] and dt.datetime.fromisoformat(s["deadline"]).date() == received.date():
        pri = "high"
    elif s["blocked"] and sla == "Premium":
        pri = "high"
    else:
        pri = "normal"
    if pri != r["expected"]["priority"]:
        fail_pri.append({"id": r["id"], "expected": r["expected"]["priority"], "recomputed": pri})
    if (s["lifecycle"] == "reply") != r["expected"]["needs_reply"]:
        fail_reply.append(r["id"])
    if s["category"] != r["expected"]["category"]:
        fail_cat.append(r["id"])
    bank = sc.NOTICES if s["lifecycle"] == "notice" else sc.ASKS
    owners = {c for c, items in bank.items() for it in items
              if re.fullmatch(re.escape(it).replace(re.escape("{system}"), ".+"), s["topic"])}
    if s["topic"].startswith("status update on the ongoing problem"):
        owners = {"technical"}
    if owners != {s["category"]}:
        fail_cat_trace.append({"id": r["id"], "topic": s["topic"], "owners": sorted(owners)})
check("A1 priority recomputed from header and spec", fail_pri)
check("A2 needs_reply equals reply lifecycle", fail_reply)
check("A3 category equals spec category", fail_cat)
check("A4 topic traces to exactly one team's hand-written list", fail_cat_trace)

# ---- B. header vs spec ---------------------------------------------------------
fail_hdr = []
for r in recs:
    s = specs[r["id"]]
    received, sla, incidents, _, _ = parse_header(r["state"])
    want = [{"id": i["id"], "system": i["system"], "impact": i["state"] == "open_impact"} for i in s["incidents"]]
    got = [{"id": i["id"], "system": i["system"], "impact": i["impact"]} for i in incidents]
    if received != dt.datetime.fromisoformat(s["received"]) or sla != s["sla"] or want != got:
        fail_hdr.append(r["id"])
    if received.weekday() >= 5:
        fail_hdr.append(r["id"] + " weekend")
check("B1 header matches spec (received, SLA, incidents)", fail_hdr)

# ---- C. deadline phrases --------------------------------------------------------
WD_EN = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
WD_ES = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
MON_EN = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]
MON_ES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
fail_verbatim, fail_sem, imprecise_next_week = [], [], []
for r in recs:
    s = specs[r["id"]]
    if not s["deadline_phrase"]:
        continue
    _, _, _, subject, body = parse_header(r["state"])
    phrase = s["deadline_phrase"]
    if phrase not in body:
        fail_verbatim.append(r["id"])
    dl = dt.datetime.fromisoformat(s["deadline"])
    rec = dt.datetime.fromisoformat(specs[r["twin_of"] or r["id"]]["received"])  # phrase was written for the source's header
    p = phrase.lower()
    problems = []
    for names in (WD_EN, WD_ES):
        named = [i for i, n in enumerate(names) if re.search(rf"\b{n}\b", p)]
        if named and named != [dl.weekday()]:
            problems.append("weekday")
    for names in (MON_EN, MON_ES):
        for i, n in enumerate(names):
            if re.search(rf"\b{n}\b|\b{n[:3]}\b", p) and i + 1 != dl.month:
                problems.append("month")
    months = "|".join(MON_EN + MON_ES + [n[:3] for n in MON_EN])
    m = (re.search(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:de\s+)?(?:{months})\b", p)
         or re.search(r"\bthe (\d{1,2})(?:st|nd|rd|th)\b|\bel día (\d{1,2})\b", p))
    if m and int(next(g for g in m.groups() if g)) != dl.day:
        problems.append("day")
    m = re.search(r"\b(\d{1,2})/(\d{1,2})\b", p)
    if m and (int(m.group(1)), int(m.group(2))) != (dl.day, dl.month):
        problems.append("d/m")
    times = re.findall(r"\b(\d{1,2}):(\d{2})\b(?!\s*(?:am|pm))", p)
    if times and (int(times[0][0]), int(times[0][1])) != (dl.hour, dl.minute):
        problems.append("time")
    m = re.search(r"\b(\d{1,2})(?::(\d{2}))? (am|pm)\b", p)
    if m:
        h = int(m.group(1)) % 12 + (12 if m.group(3) == "pm" else 0)
        if (h, int(m.group(2) or 0)) != (dl.hour, dl.minute):
            problems.append("12h time")
    if "noon" in p and (dl.hour, dl.minute) != (12, 0):
        problems.append("noon")
    days = (dl.date() - rec.date()).days
    if re.search(r"\btoday\b|\bhoy\b", p) and days != 0:
        problems.append("today")
    if re.search(r"\btomorrow\b|\bmañana\b", p) and days != 1:
        problems.append("tomorrow")
    if "next week" in p or "semana que viene" in p:
        if days < 2:
            problems.append("next week too soon")
        elif (rec + dt.timedelta(days=days)).isocalendar()[1] == rec.isocalendar()[1]:
            imprecise_next_week.append(r["id"])
    if problems:
        fail_sem.append({"id": r["id"], "phrase": phrase, "deadline": s["deadline"], "received": str(rec), "problems": problems})
check("C1 deadline phrase appears verbatim in the body", fail_verbatim)
check("C2 deadline phrase agrees with the spec deadline", fail_sem)
out["checks"]["C3 'next week' phrases whose deadline falls in the same week"] = {
    "count": len(imprecise_next_week), "label_effect": "none: these deadlines are never on the received day, so priority stays normal",
    "examples": imprecise_next_week[:5]}

# ---- D. twins --------------------------------------------------------------------
by_id = {r["id"]: r for r in recs}
fail_twin = []
for r in recs:
    if not r["twin_of"]:
        continue
    a, b = by_id.get(r["twin_of"]), r
    if a is None:
        fail_twin.append({"id": b["id"], "problem": "source missing"}); continue
    ha, hb = parse_header(a["state"]), parse_header(b["state"])
    sa, sb = specs[a["id"]], specs[b["id"]]
    probs = []
    if ha[3:] != hb[3:]:
        probs.append("text differs")
    if a["split"] != b["split"]:
        probs.append("split differs")
    if a["expected"]["priority"] == b["expected"]["priority"]:
        probs.append("priority equal")
    if (a["expected"]["category"], a["expected"]["needs_reply"]) != (b["expected"]["category"], b["expected"]["needs_reply"]):
        probs.append("other labels differ")
    change = b["twin_change"]
    diff = {"received_date": ha[0] != hb[0], "sla": ha[1] != hb[1], "incident_state": ha[2] != hb[2]}
    if [k for k, v in diff.items() if v] != [change]:
        probs.append(f"header change {diff} vs {change}")
    if change == "received_date":
        dl = dt.datetime.fromisoformat(sb["deadline"])
        if not sa["absolute_deadline"]:
            probs.append("relative phrase on a date twin")
        for h in (ha, hb):
            gap = (dl.date() - h[0].date()).days
            if not 0 <= gap <= 6 or (gap == 0 and h[0] >= dl):
                probs.append(f"deadline gap {gap}")
    if probs:
        fail_twin.append({"id": b["id"], "problems": probs})
check("D1 twin pairs are clean counterfactuals", fail_twin, detail={"pairs": sum(1 for r in recs if r["twin_of"])})

# ---- E. fidelity -----------------------------------------------------------------
REL = re.compile(r"\b(today|tonight|this (morning|afternoon|evening)|end of (the )?day|EOD|hoy|esta (mañana|tarde|noche)|fin del día)\b", re.I)
LABELW = re.compile(r"high priority|low priority|alta prioridad|baja prioridad|needs_reply|prioridad alta|prioridad baja", re.I)
fail_fid, fail_rel, fail_lbl = [], [], []
for r in recs:
    src = r["twin_of"] or r["id"]
    g = cache.get(src)
    s = specs[src]
    if g is None or not g.get("check") or not all(v for k, v in g["check"].items() if k != "problems"):
        fail_fid.append(r["id"]); continue
    _, _, _, subject, body = parse_header(r["state"])
    if subject != g["subject"] or body != g["body"]:
        fail_fid.append(r["id"] + " text differs from cache")
    avoid = bool(s["deadline"] and s["absolute_deadline"]) or s["lifecycle"] == "notice" or s["incident_ref"] or s["blocked"]
    if avoid and REL.search(subject + " " + body) and s.get("date_mention") not in ("today", "hoy"):
        fail_rel.append(r["id"])
    if LABELW.search(subject + " " + body):
        fail_lbl.append(r["id"])
retries = collections.Counter(len(cache[r["id"]]["attempts"]) for r in recs if not r["twin_of"])
check("E1 every email comes from an accepted generation with all checks passed", fail_fid,
      detail={"attempts_per_accepted_email": dict(sorted(retries.items())), "specs_dropped": len(specs) - len(recs)})
check("E2 no relative day words where the brief forbade them", fail_rel)
check("E3 no label words in the text", fail_lbl)

# ---- F. splits --------------------------------------------------------------------
fail_split = []
for kind in ("accounts",):
    seen = collections.defaultdict(set)
    for sp, items in assignment[kind].items():
        for it in items:
            seen[it].add(sp)
    fail_split += [k for k, v in seen.items() if len(v) > 1]
for kind in ("asks", "notices"):
    for cat, per in assignment[kind].items():
        seen = collections.defaultdict(set)
        for sp, items in per.items():
            for it in items:
                seen[it].add(sp)
        fail_split += [k for k, v in seen.items() if len(v) > 1]
acc_split = collections.defaultdict(set)
for r in recs:
    acc_split[r["account"]].add(r["split"])
fail_split += [a for a, v in acc_split.items() if len(v) > 1]
check("F1 accounts, requests and notices are split-disjoint", fail_split)
texts = collections.defaultdict(set)
for r in recs:
    texts[parse_header(r["state"])[3:]].add(r["group"])
check("F2 no identical email text outside a twin group", [str(len(v)) for v in texts.values() if len(v) > 1])


def grams(t, n=5):
    w = re.findall(r"\w+", t.lower())
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


train = [(r["id"], grams(parse_header(r["state"])[4])) for r in recs if r["split"] == "train"]
index = collections.defaultdict(set)
for i, (rid, g) in enumerate(train):
    for x in g:
        index[x].add(i)
worst = []
for r in recs:
    if r["split"] != "test":
        continue
    g = grams(parse_header(r["state"])[4])
    counts = collections.Counter(j for x in g for j in index.get(x, ()))
    if counts:  # exact maximum Jaccard over every training body sharing a 5-gram (order-independent)
        worst.append(round(max(n / len(g | train[j][1]) for j, n in counts.items()), 3))
worst.sort(reverse=True)
check("F3 no test body is a near-duplicate of a training body (5-gram Jaccard >= 0.5)", [w for w in worst if w >= 0.5],
      detail={"max_jaccard": worst[0] if worst else 0, "p99": worst[len(worst) // 100] if worst else 0})

# ---- G. balance -------------------------------------------------------------------
bal = {}
for sp in ("train", "validation", "calibration", "test"):
    rows = [r for r in recs if r["split"] == sp]
    bal[sp] = {f: dict(collections.Counter(str(r["expected"][f]) for r in rows)) for f in ("category", "priority", "needs_reply")}
    bal[sp]["language"] = dict(collections.Counter(r["language"] for r in rows))
out["balance"] = bal

(Path(__file__).resolve().parent / "generation_review.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
for k, v in out["checks"].items():
    print(f"{'FAIL' if k in out['hard_failures'] else 'ok  '} {k}: {v.get('failures', v.get('count'))}")
sys.exit(1 if out["hard_failures"] else 0)
