"""Build deterministic email specs and their labels. No API calls here.

Labels come only from the spec. The LLM later writes text that must express the spec;
a fidelity check rejects text that does not. Priority depends on the header (received
time, SLA, incidents), so the same body can carry different labels (counterfactual twins).
"""
import datetime as dt
import hashlib
import json
import random
from pathlib import Path

from scenarios import ACCOUNTS, ASKS, NOTICES, CONFUSERS, FIRST_NAMES, LAST_NAMES, ROLES, REGISTERS

ROOT = Path(__file__).resolve().parent
SEED = 20261012
SPLITS = ("train", "validation", "calibration", "test")
ACCOUNT_SPLIT = {"train": 50, "validation": 10, "calibration": 10, "test": 10}
ASK_SPLIT = {"train": 18, "validation": 4, "calibration": 4, "test": 4}
NOTICE_SPLIT = {"train": 6, "validation": 2, "calibration": 2, "test": 2}
PER_ACCOUNT = 40
CATEGORIES = ["billing", "technical", "sales", "scheduling", "information"]
EOD = dt.time(18, 0)
# Asks that are requests for information; they always expect a written answer.
REPLY_ONLY = ("explain", "send ", "provide", "confirm", "check", "find out", "get ", "price ", "ask about", "ask for")

WD_EN = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
WD_ES = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
MON_EN = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]
MON_ES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
          "septiembre", "octubre", "noviembre", "diciembre"]


def split_list(items, sizes, rng):
    items = list(items)
    rng.shuffle(items)
    out, i = {}, 0
    for s in SPLITS:
        out[s] = items[i:i + sizes[s]]
        i += sizes[s]
    assert i == len(items), (i, len(items))
    return out


def t12(t):
    h = t.hour % 12 or 12
    return f"{h} pm" if t.minute == 0 and t.hour >= 12 else (
        f"{h} am" if t.minute == 0 else f"{h}:{t.minute:02d} {'pm' if t.hour >= 12 else 'am'}")


def absolute_phrase(deadline, lang, rng):
    """Deadline phrase that stays true whatever day the email is received (used by twins)."""
    d, t = deadline.date(), deadline.time()
    if lang == "en":
        return rng.choice([
            f"by {WD_EN[d.weekday()]} at {t12(t)}",
            f"before {t:%H:%M} on {WD_EN[d.weekday()]} {d.day} {MON_EN[d.month - 1]}",
            f"by {d.day} {MON_EN[d.month - 1][:3]}, {t:%H:%M}",
            f"no later than {t12(t)} on {WD_EN[d.weekday()]}",
            f"by {t:%H:%M} on the {d.day}{'th' if 11 <= d.day <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(d.day % 10, 'th')}",
            f"before {'noon' if t.hour == 12 and t.minute == 0 else t12(t)} on {WD_EN[d.weekday()]}",
            f"{WD_EN[d.weekday()]} {d.day}/{d.month}, {t:%H:%M} at the latest",
        ])
    return rng.choice([
        f"antes del {WD_ES[d.weekday()]} a las {t:%H:%M}",
        f"antes de las {t:%H:%M} del {WD_ES[d.weekday()]} {d.day} de {MON_ES[d.month - 1]}",
        f"para el {d.day} de {MON_ES[d.month - 1]} a las {t:%H:%M}",
        f"como tarde el {WD_ES[d.weekday()]} a las {t:%H:%M}",
        f"el día {d.day} antes de las {t:%H:%M}",
        f"no más tarde de las {t:%H:%M} del {WD_ES[d.weekday()]}",
        f"el {WD_ES[d.weekday()]} {d.day}/{d.month} a las {t:%H:%M} como máximo",
    ])


def relative_phrase(received, deadline, lang, rng):
    """Deadline phrase relative to the received time; only valid for that header."""
    days = (deadline.date() - received.date()).days
    t = deadline.time()
    if days == 0:
        if t == EOD:
            return rng.choice(["by end of day today", "before close of business today"]) if lang == "en" \
                else rng.choice(["hoy antes de que acabe el día", "antes del cierre de hoy"])
        return rng.choice([f"by {t12(t)} today", f"today before {t:%H:%M}"]) if lang == "en" \
            else rng.choice([f"hoy antes de las {t:%H:%M}", f"hoy como tarde a las {t:%H:%M}"])
    if days == 1:
        return rng.choice(["tomorrow morning", f"by tomorrow at {t12(t)}"]) if lang == "en" \
            else rng.choice(["mañana por la mañana", f"mañana antes de las {t:%H:%M}"])
    return rng.choice(["by the end of next week", "sometime next week"]) if lang == "en" \
        else rng.choice(["antes de que acabe la semana que viene", "en algún momento de la semana que viene"])


def business_days(start, end):
    d = start
    while d <= end:
        if d.weekday() < 5:
            yield d
        d += dt.timedelta(days=1)


DAYS = list(business_days(dt.date(2026, 10, 5), dt.date(2026, 11, 27)))


def rand_received(rng, latest=dt.time(17, 30)):
    d = rng.choice(DAYS)
    minutes = rng.randint(7 * 60 + 30, latest.hour * 60 + latest.minute)
    return dt.datetime.combine(d, dt.time(minutes // 60, minutes % 60))


def make_deadline(received, same_day, rng):
    if same_day:
        start = received + dt.timedelta(minutes=60)
        end = dt.datetime.combine(received.date(), EOD)
        m = rng.randint(0, max(0, int((end - start).total_seconds() // 1800)))
        dl = (start + dt.timedelta(minutes=30 * m)).replace(second=0)
        dl = dl.replace(minute=0 if dl.minute < 30 else 30)
        return min(max(dl, received + dt.timedelta(minutes=30)), end)
    for _ in range(50):
        d = received.date() + dt.timedelta(days=rng.randint(1, 6))
        if d.weekday() < 5:
            return dt.datetime.combine(d, dt.time(rng.choice([9, 10, 11, 12, 13, 14, 15, 16, 17]), rng.choice([0, 30])))
    raise RuntimeError("no deadline")


def date_mention(received, lang, rng, allow_relative):
    """A time reference that is NOT a deadline. Often names the received day itself."""
    r = received.date()
    kind = rng.choices(["same_weekday", "same_date", "other_day", "today"], [0.35, 0.2, 0.3, 0.15 if allow_relative else 0])[0]
    if kind == "today":
        return "today" if lang == "en" else "hoy"
    if kind == "same_weekday":
        d = r
    elif kind == "same_date":
        return f"{r.day} {MON_EN[r.month - 1]}" if lang == "en" else f"{r.day} de {MON_ES[r.month - 1]}"
    else:
        d = r + dt.timedelta(days=rng.choice([-7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7]))
    return WD_EN[d.weekday()] if lang == "en" else WD_ES[d.weekday()]


def priority_label(s):
    if s["lifecycle"] == "notice":
        return "high" if s["incident_ref"] and s["incident_state"] == "open_impact" else "low"
    if s["incident_ref"] and s["incident_state"] == "open_impact":
        return "high"
    if s["deadline"] and dt.datetime.fromisoformat(s["deadline"]).date() == dt.datetime.fromisoformat(s["received"]).date():
        return "high"
    if s["blocked"] and s["sla"] == "Premium":
        return "high"
    return "normal"


def labels(s):
    return {"category": s["category"], "priority": priority_label(s), "needs_reply": s["lifecycle"] == "reply"}


def base_spec(rng, account, split, lang, category, asks, notices):
    name, sector, systems = account
    s = {
        "account": name, "sector": sector, "split": split, "language": lang, "category": category,
        "sender": f"{rng.choice(FIRST_NAMES[lang])} {rng.choice(LAST_NAMES[lang])}",
        "role": rng.choice(ROLES), "register": rng.choice(REGISTERS),
        "length": rng.choices(["short", "medium", "long"], [0.35, 0.45, 0.2])[0],
        "sla": rng.choice(["Premium", "Standard"]),
        "received": None, "deadline": None, "deadline_phrase": None, "absolute_deadline": False,
        "blocked": False, "incident_ref": False, "incident_state": None, "incidents": [],
        "confuser": None, "misleading_subject": False, "urgency_tone": "neutral",
        "quoted": rng.choices(["none", "prior_thread", "old_urgent", "forwarded"], [0.45, 0.25, 0.2, 0.1])[0],
        "courtesy_line": False, "twin_of": None, "twin_change": None,
        "date_mention": None, "negated_blocker": False, "explicit_no_reply": True,
        "reply_style": None, "cc_question": False,
    }
    system = rng.choice(systems)
    s["lifecycle"] = rng.choices(["reply", "action", "notice"], [0.36, 0.30, 0.34])[0]
    if s["lifecycle"] == "notice":
        s["topic"] = rng.choice(notices[category]).format(system=system)
        s["courtesy_line"] = rng.random() < 0.4
        if category == "technical" and rng.random() < 0.2:
            s["topic"] = f"status update on the ongoing problem with the {system}: still being investigated, next update to follow"
            s["incident_ref"], s["incident_state"] = True, "open_impact"
    else:
        s["topic"] = rng.choice(asks[category]).format(system=system)
        r = rng.random()
        if category == "technical" and "{system}" in "".join(ASKS["technical"]) and r < 0.28 and system in s["topic"]:
            s["incident_ref"] = True
            s["incident_state"] = rng.choice(["open_impact", "open_impact", "monitoring"])
        else:
            driver = rng.choices(["deadline_today", "deadline_later", "blocked", "none"], [0.24, 0.26, 0.2, 0.3])[0]
            if driver in ("deadline_today", "deadline_later"):
                same = driver == "deadline_today"
                s["received"] = rand_received(rng, dt.time(16, 0) if same else dt.time(17, 30)).isoformat()
                rec = dt.datetime.fromisoformat(s["received"])
                dl = make_deadline(rec, same, rng)
                s["deadline"] = dl.isoformat()
                s["absolute_deadline"] = rng.random() < 0.6
                s["deadline_phrase"] = absolute_phrase(dl, lang, rng) if s["absolute_deadline"] else relative_phrase(rec, dl, lang, rng)
            elif driver == "blocked":
                s["blocked"] = True
    if s["received"] is None:
        s["received"] = rand_received(rng).isoformat()
    allow_relative = not ((s["deadline"] and s["absolute_deadline"]) or s["lifecycle"] == "notice" or s["incident_ref"] or s["blocked"])
    if rng.random() < 0.55:
        s["date_mention"] = date_mention(dt.datetime.fromisoformat(s["received"]), lang, rng, allow_relative)
    if s["lifecycle"] != "notice" and not s["blocked"]:
        s["negated_blocker"] = rng.random() < 0.2
    if s["lifecycle"] == "reply":
        s["reply_style"] = rng.choice(["direct question", "imperative request to confirm or send", "indirect, no question mark"])
    else:
        s["cc_question"] = rng.random() < 0.25
    if s["lifecycle"] == "action":
        s["explicit_no_reply"] = rng.random() < 0.5
        s["courtesy_line"] = rng.random() < 0.3
        if s["topic"].startswith(REPLY_ONLY):
            # The ask itself is a request for information, so it cannot be "do it, no reply".
            # Converted without consuming randomness, so other specs stay unchanged.
            h = int(hashlib.sha256(s["topic"].encode() + s["received"].encode()).hexdigest(), 16)
            s.update(lifecycle="reply", explicit_no_reply=True, cc_question=False, courtesy_line=False,
                     reply_style=["direct question", "imperative request to confirm or send", "indirect, no question mark"][h % 3],
                     reply_only_conversion=True)
    if s["incident_ref"]:
        s["incidents"].append({"id": f"INC-{rng.randint(40000, 99999)}", "system": system, "state": s["incident_state"]})
    elif rng.random() < 0.3:
        other = [x for x in systems if x not in s["topic"]] or systems
        s["incidents"].append({"id": f"INC-{rng.randint(40000, 99999)}", "system": rng.choice(other),
                               "state": rng.choice(["open_impact", "monitoring", "resolved"])})
    if rng.random() < 0.5:
        s["confuser"] = rng.choice([c for c in CATEGORIES if c != category])
        s["misleading_subject"] = rng.random() < 0.4
    high = priority_label(s) == "high"
    s["urgency_tone"] = rng.choices(["calm", "neutral"], [0.5, 0.5])[0] if high else \
        rng.choices(["pushy", "neutral"], [0.3, 0.7])[0]
    return s


def make_twin(s, rng):
    """Same text, different header, different priority. Returns None if not eligible."""
    t = dict(s, incidents=[dict(i) for i in s["incidents"]], twin_of=s["id"])
    if s["deadline"] and s["absolute_deadline"]:
        dl = dt.datetime.fromisoformat(s["deadline"])
        same = dl.date() == dt.datetime.fromisoformat(s["received"]).date()
        if same:  # receive it on an earlier business day instead
            days = [d for d in DAYS if 1 <= (dl.date() - d).days <= 6]
            if not days:
                return None
            d = rng.choice(days)
            m = rng.randint(7 * 60 + 30, 17 * 60 + 30)
            t["received"] = dt.datetime.combine(d, dt.time(m // 60, m % 60)).isoformat()
        else:  # receive it on the deadline day, before the deadline
            lo = 7 * 60 + 30
            hi = dl.hour * 60 + dl.minute - 60
            if hi < lo:
                return None
            m = rng.randint(lo, hi)
            t["received"] = dt.datetime.combine(dl.date(), dt.time(m // 60, m % 60)).isoformat()
        t["twin_change"] = "received_date"
    elif s["blocked"]:
        t["sla"] = "Standard" if s["sla"] == "Premium" else "Premium"
        t["twin_change"] = "sla"
    elif s["incident_ref"] and s["lifecycle"] != "notice":
        new = "monitoring" if s["incident_state"] == "open_impact" else "open_impact"
        t["incident_state"] = new
        t["incidents"][0]["state"] = new
        t["twin_change"] = "incident_state"
    else:
        return None
    if priority_label(t) == priority_label(s):
        return None
    t["labels"] = labels(t)
    t["id"] = "rl-" + hashlib.sha256((s["id"] + "-twin").encode()).hexdigest()[:16]
    return t


def build():
    rng = random.Random(SEED)
    acc = split_list(ACCOUNTS, ACCOUNT_SPLIT, rng)
    asks = {c: split_list(ASKS[c], ASK_SPLIT, rng) for c in CATEGORIES}
    notices = {c: split_list(NOTICES[c], NOTICE_SPLIT, rng) for c in CATEGORIES}
    specs = []
    for split in SPLITS:
        for account in acc[split]:
            base, n = [], 0
            cats = (CATEGORIES * 8)[:32]
            rng.shuffle(cats)
            for i, c in enumerate(cats):
                lang = "en" if i % 2 == 0 else "es"
                s = base_spec(rng, account, split, lang, c, {k: v[split] for k, v in asks.items()},
                              {k: v[split] for k, v in notices.items()})
                s["id"] = "rl-" + hashlib.sha256(f"{account[0]}-{i}".encode()).hexdigest()[:16]
                s["labels"] = labels(s)
                base.append(s)
            twins = []
            order = list(base)
            rng.shuffle(order)
            for s in order:
                if len(base) + len(twins) >= PER_ACCOUNT:
                    break
                t = make_twin(s, rng)
                if t:
                    twins.append(t)
            # top up with extra base specs if not enough twins were possible
            i = len(cats)
            while len(base) + len(twins) < PER_ACCOUNT:
                c = rng.choice(CATEGORIES)
                lang = "en" if i % 2 == 0 else "es"
                s = base_spec(rng, account, split, lang, c, {k: v[split] for k, v in asks.items()},
                              {k: v[split] for k, v in notices.items()})
                s["id"] = "rl-" + hashlib.sha256(f"{account[0]}-{i}".encode()).hexdigest()[:16]
                s["labels"] = labels(s)
                base.append(s)
                i += 1
            specs.extend(base + twins)
    return specs, {"asks": asks, "notices": notices, "accounts": {k: [a[0] for a in v] for k, v in acc.items()}}


if __name__ == "__main__":
    import collections
    specs, assignment = build()
    (ROOT / "specs.jsonl").write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in specs))
    (ROOT / "assignment.json").write_text(json.dumps(assignment, ensure_ascii=False, indent=2) + "\n")
    print(len(specs), "specs;", sum(1 for s in specs if not s["twin_of"]), "need text")
    for split in SPLITS:
        ss = [s for s in specs if s["split"] == split]
        print(split, len(ss),
              dict(collections.Counter(s["labels"]["priority"] for s in ss)),
              dict(collections.Counter(s["labels"]["needs_reply"] for s in ss)),
              dict(collections.Counter(s["labels"]["category"] for s in ss)),
              "twins", sum(1 for s in ss if s["twin_of"]))
