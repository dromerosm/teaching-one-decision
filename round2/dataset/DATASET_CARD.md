# Dataset card: realistic email triage (round two)

## Summary

The dataset has 3,176 synthetic business emails in English (1,600) and Spanish (1,576). Each one was sent by a fictional customer to the shared inbox of a fictional software vendor, Helix Systems. Every email has three labels:

| Field | Values | Distribution |
|---|---|---|
| `category`: the team that must act | billing, technical, sales, scheduling, information | 699 / 693 / 579 / 597 / 608 |
| `priority` | low, normal, high | 824 / 1,391 / 961 |
| `needs_reply`: the sender expects a written answer | true, false | 1,425 / 1,751 |

| Split | Emails | Customer accounts |
|---|---:|---:|
| train | 1,985 | 50 |
| validation | 395 | 10 |
| calibration | 398 | 10 |
| test | 398 | 10 |

Accounts, request types and notice types are disjoint across splits. The test never shares an account or a request type with training.

## Input format

Each model input (`state`) is a header written by the triage system, followed by the email:

```
Received: Thu 26 Nov 2026, 07:55
From: <fictional name> <user@account.example>, <role>, <account>
Account SLA: Premium | Standard
Open incidents for this account: none | INC-<id> <system> (status: ...; customer impact: yes|no)
Subject: ...

<body, possibly with quoted or forwarded history>
```

Inputs are 132 to 555 tokens long with the Laya tokenizer (mean 278).

## Label policy

The written policy is in [questions.json](questions.json); every model received it.

- **Category** follows what the latest message asks for, not topics mentioned in passing, the subject line or quoted history. A notice that asks for nothing takes the category of its area.
- **Priority** is high if the email concerns an incident listed with "customer impact: yes", sets a deadline falling on the day it was received, or comes from a Premium account whose work is blocked. It is low if the email asks for no action at all, and normal otherwise. Tone, capitals, the subject line and quoted history never change priority.
- **Needs reply** is true only if the latest message asks the recipient to write back, confirm or send information. Asking only for something to be done, questions addressed to a colleague in copy, and generic courtesy lines do not count.

[questions_v2.json](questions_v2.json) holds the same policy written as an explicit procedure. [questions_fewshot.json](questions_fewshot.json) adds the eight training emails in [fewshot.json](fewshot.json).

## Difficulty by design

- **Counterfactual twins.** 620 emails have the same text as another email and a different header (325 by received date, 241 by service level, 54 by incident state), so their priority differs. A text-only model can get at most half of each pair right.
- **Distractors.** Dates that are not deadlines, often the day the email arrived; negated blockers; questions to colleagues in copy; courtesy lines; quoted urgent threads; forwarded chains; pushy tone on routine requests; topics from another team, sometimes in the subject line.
- **Measured difficulty.** A word-count Naive Bayes trained on the training split scores 52.0% on the test, against 84.0% on round one. Keyword rules score 27–28%. Only 8% of test sentences appear word for word in training: greetings, forward headers and job titles.
- **Label consistency.** `gpt-6.1-sol`, given only the written policy, recovers 99.5% of the test labels and 98.2% of the validation labels.

## How it was made

- **Briefs.** Scenarios were written by hand: 80 accounts, 150 request types and 60 notice types. A seeded script turned them into briefs, and each brief fixes the labels.
- **Text.** `gpt-6.1-sol` wrote each email from its brief. A programmatic check confirmed that deadline phrases appear verbatim and that forbidden relative dates and label words are absent. A second `gpt-6.1-sol` call checked that the text matches the brief.
- **Rejections.** Emails that failed both checks were rewritten up to three times or dropped; 24 specs were dropped.
- **Label consistency checks.** The checks over all emails found 11 ambiguous notice topics and 140 requests that cannot sensibly be "done without a reply". Those were fixed and regenerated before any model was evaluated.
- **API calls.** 8,307 generation calls, plus the label consistency checks.

[../generator/generation_cache.jsonl](../generator/generation_cache.jsonl) keeps every accepted email with its checker verdicts. The dataset rebuilds from it byte for byte; see the main README.

## Files

| File | Content |
|---|---|
| `records.jsonl` | All emails with id, split, account, language, twin link, lifecycle, model input, labels, the brief's key fields and provenance |
| `train.jsonl`, `validation.jsonl`, `calibration.jsonl`, `test.jsonl` | Model-ready rows (`state`, `questions`, `expected`) in the Laya training format |
| `emails.json` | The test rows used by the evaluation scripts |
| `questions.json`, `questions_v2.json`, `questions_fewshot.json`, `schema.json` | The policy as Laya / Decision 3.0 questions and as a JSON Schema for generative models |
| `fewshot.json`, `think_subset.json` | The eight few-shot examples; the fixed test sample for the thinking run |
| `assignment.json` | Which accounts, requests and notices belong to each split |
| `ASSEMBLED.sha256`, `tokenizer_check.json` | Checksums of the assembled files; token lengths |

## Limits and intended use

This is a benchmark for comparing ways of teaching a decision under controlled conditions. It is not training data for a production system. The policy is explicit and applied without exceptions. Real decisions are noisier, threads are longer and context is often missing. The text was written by one language model, so it shares that model's style.

All people, companies, email addresses, incidents and events are fictional. Names are random combinations of common first names and surnames; the author's own name appears a few times by chance. Domains use the reserved `.example` suffix.

## Licence

CC BY 4.0. See [../../LICENSE-DATA](../../LICENSE-DATA).
