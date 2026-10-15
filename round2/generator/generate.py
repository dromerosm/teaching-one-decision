"""Write email text for each spec with gpt-6.1-sol, then check fidelity before accepting it.

Usage:
  python3 generate.py --limit 40 [--split test] [--budget 60]
Results are cached per spec in gen/<id>.json, so reruns resume. Twins reuse their
source's text and need no call. Spend is tracked from API usage and capped by --budget.
"""
import argparse
import concurrent.futures as cf
import datetime as dt
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GEN = ROOT / "gen"
CACHE = ROOT / "generation_cache.jsonl"
MODEL = "gpt-6.1-sol"
PRICE_IN, PRICE_OUT = 2.0 / 1e6, 10.0 / 1e6  # USD per token, published list price (2026-10)
VENDOR = "Helix Systems"
LOCK = threading.Lock()
SPEND = {"usd": 0.0, "calls": 0}

STATE_TXT = {"open_impact": "still under investigation and affecting the customer",
             "monitoring": "fix deployed, being monitored", "resolved": "resolved"}
NO_RELATIVE = re.compile(r"\b(today|tonight|this (morning|afternoon|evening)|end of (the )?day|EOD|hoy|esta (mañana|tarde|noche)|fin del día)\b", re.I)
LABEL_WORDS = re.compile(r"high priority|low priority|alta prioridad|baja prioridad|needs_reply|prioridad alta|prioridad baja", re.I)


def call(messages, schema, name):
    body = {"model": MODEL, "messages": messages,
            "response_format": {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}}}
    for attempt in range(5):
        req = urllib.request.Request("https://api.openai.com/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"],
                                              "Content-Type": "application/json"})
        try:
            r = json.load(urllib.request.urlopen(req, timeout=180))
            u = r["usage"]
            with LOCK:
                SPEND["usd"] += u["prompt_tokens"] * PRICE_IN + u["completion_tokens"] * PRICE_OUT
                SPEND["calls"] += 1
            return json.loads(r["choices"][0]["message"]["content"]), u
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and attempt < 4:
                time.sleep(2 ** attempt * 3)
                continue
            raise RuntimeError(f"HTTP {e.code}: {e.read()[:300]!r}")
        except (urllib.error.URLError, TimeoutError):
            if attempt < 4:
                time.sleep(2 ** attempt * 3)
                continue
            raise


EMAIL_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["subject", "body"],
                "properties": {"subject": {"type": "string"}, "body": {"type": "string"}}}

SYSTEM = f"""You write realistic business emails for a research dataset. Every email is sent by a customer to the shared inbox of {VENDOR}, a B2B software and services vendor. You receive a JSON brief and return the subject and body of ONE email.

Follow the brief exactly; a separate reviewer checks every point. Write like a real person in the stated register: vary greetings, sign-offs, structure and vocabulary; invent plausible names, ticket or order numbers when useful. Never mention datasets, labels, triage, priority levels or the brief. Never write the words "priority", "needs reply" or "urgent" unless the brief's tone allows urgency words.

Rules for the LATEST message (the text above any quoted history):
- It contains exactly one request: the ASK in the brief (or, for a notice, no request at all). Passing mentions must not become requests.
- reply_mode "reply": the sender clearly wants a written answer, confirmation or information sent back. Vary how this is expressed.
- reply_mode "action": the sender wants the task done but does NOT expect an email back (see explicit_no_reply). Ask the recipient no questions.
- reply_mode "notice": the message only informs. It asks for nothing.
- deadline_phrase: if present, include it EXACTLY as written, character for character, as the deadline for the ask. Do not give any other date, weekday or time for the deadline.
- avoid_relative_time true: do not use words like today, tonight, this morning, this afternoon, end of day, hoy, esta mañana, esta tarde, anywhere in the email.
- blocked true: say plainly that the sender's work or operation cannot proceed until this is done. blocked false: do not say they are blocked or stopped.
- incident: if present, the email is about that problem with the named system; describe it in your own words; you may cite the incident ID; do NOT state its current status. If absent, do not mention any incident or outage unless the passing mention requires it.
- passing_mention: weave it in as background only.
- date_mention: if present, use this exact word or date in the latest message for something that is NOT a deadline for the ask (a past event, a payment already made, a call that happened or is planned, a document date). It must not read as a time limit for the ask.
- negated_blocker true: include a natural remark that nothing is blocked or held up on the sender's side.
- explicit_no_reply (reply_mode "action" only): true = say in some natural way that no written answer is needed; false = do not say it; make it clear only implicitly (e.g. they will see the change in the system, or it can simply be done), and ask nothing.
- reply_style (reply_mode "reply" only): how the request for a written answer is phrased. "indirect, no question mark" = no question mark and none of the words confirm/let me know; e.g. "I'd be glad to have the figures back from you".
- cc_question true: the latest message also asks a question or favour of a named colleague of the sender who is in copy (addressed to them by name). That question is for the colleague, not for the recipient, and must not be the ask.
- courtesy_line (any mode): if true, include a generic courtesy offer such as "let me know if anything is unclear"; it is not a real request.
- misleading_subject true: the subject line is about the passing mention, not about the ask.
- urgency_tone "pushy": the sender sounds pressing (ASAP, chasing, exclamation marks) but adds no deadline or blocker beyond the brief. "calm": no urgency words at all. "neutral": ordinary tone.
- length of the latest message, excluding quoted history: short 40-90 words, medium 90-180, long 180-320.

Quoted history (below the latest message, quoted with ">" or as a forwarded block):
- "none": no quoted history.
- "prior_thread": one earlier message in the same thread, from {VENDOR} or the customer, about a related but already settled point.
- "old_urgent": an older message that contained a pressing request or deadline which is now in the past and settled; it must not change the latest request.
- "forwarded": the latest message is a short note forwarding a colleague's email; the ASK is in the note; the forwarded email may contain other requests that are not the ask.

Write everything, including quoted history, in the language given."""


def brief(s):
    b = {
        "language": "English" if s["language"] == "en" else "Spanish (Spain)",
        "sender": f"{s['sender']}, {s['role']} at {s['account']} ({s['sector']})",
        "register": s["register"], "length": s["length"],
        "reply_mode": s["lifecycle"],
        ("notice_topic" if s["lifecycle"] == "notice" else "ask"): s["topic"],
        "avoid_relative_time": bool(s["deadline"] and s["absolute_deadline"]) or s["lifecycle"] == "notice" or s["incident_ref"] or s["blocked"],
        "blocked": s["blocked"],
        "urgency_tone": s["urgency_tone"],
        "quoted_history": s["quoted"],
        "misleading_subject": s["misleading_subject"],
    }
    b["courtesy_line"] = s["courtesy_line"]
    if s.get("date_mention"):
        b["date_mention"] = s["date_mention"]
    b["negated_blocker"] = s.get("negated_blocker", False)
    if s.get("reply_style"):
        b["reply_style"] = s["reply_style"]
    b["cc_question"] = s.get("cc_question", False)
    if s["lifecycle"] == "action":
        b["explicit_no_reply"] = s.get("explicit_no_reply", True)
    if s["deadline_phrase"]:
        b["deadline_phrase"] = s["deadline_phrase"]
        if not s["absolute_deadline"]:
            b["avoid_relative_time"] = False
    if s["incident_ref"]:
        inc = s["incidents"][0]
        b["incident"] = f"{inc['id']}: problem with the {inc['system']}"
    if s["confuser"]:
        from scenarios import CONFUSERS
        b["passing_mention"] = CONFUSERS[s["confuser"]]
    return b


CHECK_SCHEMA = {"type": "object", "additionalProperties": False,
                "required": ["single_request_matches", "reply_mode_matches", "blocked_matches", "incident_matches",
                             "deadline_consistent", "passing_mention_not_a_request", "problems"],
                "properties": {k: {"type": "boolean"} for k in
                               ["single_request_matches", "reply_mode_matches", "blocked_matches", "incident_matches",
                                "deadline_consistent", "passing_mention_not_a_request"]} | {"problems": {"type": "string"}}}

CHECK_SYSTEM = """You audit emails written from a brief. Judge only the LATEST message (ignore quoted or forwarded history except to confirm it does not carry the latest request). Be strict; answer false when in doubt.
- single_request_matches: for an ask, the latest message requests that ask (a paraphrase is fine) and no other substantive request; for a notice, it requests nothing (a generic courtesy offer is allowed).
- reply_mode_matches: "reply" = the sender expects a written answer, confirmation or information back from the recipient; "action" = the sender wants it done and does NOT expect an email back from the recipient, and asks the recipient no question; "notice" = informs only. A question addressed by name to a colleague in copy is not addressed to the recipient.
- blocked_matches: if blocked is true, the sender says their work cannot proceed; if false, they do not claim to be blocked.
- incident_matches: if an incident is given, the email is about that problem and does not state its current status; if none, no incident or outage is presented as the subject of the request.
- deadline_consistent: if a deadline_phrase is given, it is the only deadline for the ask and nothing contradicts it; if none, the latest message sets no deadline or time limit for the ask. A date_mention, if given, must not work as a deadline for the ask.
- passing_mention_not_a_request: the passing mention (if any) is background only; true if there is none.
- problems: one short sentence listing failures, or "" if none."""


def program_checks(s, out):
    errs = []
    text = out["subject"] + "\n" + out["body"]
    if s["deadline_phrase"] and s["deadline_phrase"] not in out["body"]:
        errs.append("deadline phrase not verbatim in body")
    b = brief(s)
    if b["avoid_relative_time"] and NO_RELATIVE.search(text) and b.get("date_mention") not in ("today", "hoy"):
        errs.append(f"relative time word used: {NO_RELATIVE.search(text).group(0)}")
    if LABEL_WORDS.search(text):
        errs.append("label words used")
    words = len(out["body"].split())
    if words < 25:
        errs.append("too short")
    return errs


def generate_one(s, max_tries=3):
    path = GEN / f"{s['id']}.json"
    if path.exists():
        return json.loads(path.read_text())
    b = brief(s)
    feedback, attempts = None, []
    for k in range(max_tries):
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": json.dumps(b, ensure_ascii=False)}]
        if feedback:
            msgs.append({"role": "user", "content": "The previous draft failed review: " + feedback + " Write a new email that fixes this."})
        out, _ = call(msgs, EMAIL_SCHEMA, "email")
        errs = program_checks(s, out)
        verdict = None
        if not errs:
            check_brief = {k2: b.get(k2) for k2 in ["reply_mode", "ask", "notice_topic", "blocked", "incident", "deadline_phrase", "passing_mention", "date_mention", "cc_question"] if k2 in b}
            verdict, _ = call([{"role": "system", "content": CHECK_SYSTEM},
                               {"role": "user", "content": json.dumps({"brief": check_brief, "email": out}, ensure_ascii=False)}],
                              CHECK_SCHEMA, "check")
            errs = [k2 for k2, v in verdict.items() if v is False]
            if errs and verdict.get("problems"):
                errs.append(verdict["problems"])
        attempts.append({"errors": errs})
        if not errs:
            rec = {"id": s["id"], "subject": out["subject"], "body": out["body"], "attempts": attempts,
                   "check": verdict, "model": MODEL, "generated_at": dt.datetime.now().isoformat(timespec="seconds")}
            path.write_text(json.dumps(rec, ensure_ascii=False, indent=1))
            return rec
        feedback = "; ".join(errs)
    rec = {"id": s["id"], "failed": True, "attempts": attempts, "last": out}
    (GEN / f"{s['id']}.failed.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--split", default=None)
    ap.add_argument("--budget", type=float, default=60.0)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    GEN.mkdir(exist_ok=True)
    if CACHE.exists():  # unpack the published cache so nothing is regenerated (no API calls)
        for line in open(CACHE):
            rec = json.loads(line)
            p = GEN / f"{rec['id']}.json"
            if not p.exists():
                p.write_text(json.dumps(rec, ensure_ascii=False, indent=1))
    specs = [json.loads(l) for l in open(ROOT / "specs.jsonl")]
    todo = [s for s in specs if not s["twin_of"] and (a.split is None or s["split"] == a.split)]
    if a.limit:
        # spread the sample over accounts, splits and label types
        todo = sorted(todo, key=lambda s: s["id"])[:: max(1, len(todo) // a.limit)][: a.limit]
    todo = [s for s in todo if not (GEN / f"{s['id']}.json").exists()]
    print(f"{len(todo)} specs to generate, budget ${a.budget}")
    ok = fail = 0
    with cf.ThreadPoolExecutor(a.workers) as ex:
        futs = {}
        for s in todo:
            futs[ex.submit(generate_one, s)] = s
        for f in cf.as_completed(futs):
            try:
                r = f.result()
                fail += bool(r.get("failed"))
                ok += not r.get("failed")
            except Exception as e:
                fail += 1
                print("ERROR", futs[f]["id"], e)
            if SPEND["usd"] > a.budget:
                print("Budget reached; cancelling remaining work")
                for g in futs:
                    g.cancel()
                break
            if (ok + fail) % 25 == 0:
                print(f"{ok} ok, {fail} failed, ${SPEND['usd']:.2f} over {SPEND['calls']} calls", flush=True)
    print(f"done: {ok} ok, {fail} failed, ${SPEND['usd']:.2f} over {SPEND['calls']} calls")
    with open(ROOT / "spend.log", "a") as fh:
        fh.write(json.dumps({"at": dt.datetime.now().isoformat(timespec="seconds"), "ok": ok, "failed": fail,
                             "usd": round(SPEND["usd"], 4), "calls": SPEND["calls"], "limit": a.limit}) + "\n")


if __name__ == "__main__":
    main()
