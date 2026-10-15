# Review of the round-two dataset generation

This review asks whether the labels are right, whether the text says what the labels assume, whether the splits are clean, and whether the generation process left biases. All automated checks are in [review_generation.py](review_generation.py) (output: [generation_review.json](generation_review.json)). That script does not import the generator's label code; it re-derives everything from the published files and from the model input text.

## Summary

| Area | Result |
|---|---|
| Labels recomputed independently from the header and the brief | 3,176 of 3,176 agree |
| Header (received time, SLA, incidents) matches the brief | 3,176 of 3,176 |
| Deadline phrases present verbatim, and weekday, date and time agree with the deadline | 1,151 of 1,151 |
| Counterfactual twins clean (same text, one header field changed, only priority differs) | 620 of 620 |
| Every email from an accepted generation whose checks all passed (1,987 accepted at the first attempt, 530 at the second, 39 at the third); no forbidden relative dates or label words | 3,176 of 3,176 |
| Accounts, requests and notices disjoint across splits; no identical text outside twin groups; no test-to-train near-duplicates | all pass (closest test-to-train pair: 5-gram Jaccard 0.071) |
| Rebuild from the published specs and generation cache | byte-identical |
| `gpt-6.1-sol` labelling from the written policy only | test 99.5%, validation 98.2% |
| Blind relabelling by a different model family (Claude), seeded random sample of 40 test emails | 40 of 40 agree on all three fields |

## How the checks were made trustworthy

- **Independent logic.** Priority is recomputed from the header text as the model sees it: received date, SLA, and incidents with "customer impact: yes". The deadline and incident fields come from the brief. Category is traced back to the hand-written request or notice list it came from, which must belong to exactly one team.
- **Coverage.** Every one of the 1,151 deadline phrases has at least one date component checked (weekday, day and month, or a relative word), and 979 also have their time checked.
- **Mutation tests.** In copies of the dataset I injected four kinds of fault: a flipped priority label, a wrong weekday in a deadline phrase, a twin with changed text, and a header with the wrong SLA. Each was caught by the check designed for it.
- **Bugs found in the checker itself.** A first version of the phrase check reported 295 "mismatches". All were parser errors: it read "11:30 del viernes" as day 30, and "3:30 pm" as 03:30. Both were fixed, the check was re-run and the mutation tests repeated.

## The remaining label disagreements, read one by one

After the label fixes, `gpt-6.1-sol` disagreed with 2 test labels and 7 validation labels. Each was read in full.

| Split | Email | Label | Model | Verdict |
|---|---|---|---|---|
| test | Notice that an SMS reminder error is fixed; the header lists an open incident on a *different* system (the appointment portal) | low | high | Label correct: the incident with customer impact is not what the email is about |
| test | Premium account: "the service account can no longer access the environment", deadline next week | normal | high | Debatable: the sender does not say their work is blocked, but a reader could infer it |
| validation (2) | Notice that catering for a joint planning day is arranged | scheduling | sales | Debatable: the policy says "a specific meeting or session = scheduling", but catering is also a purchase |
| validation (4, two twin texts) | "Could you schedule a call / arrange a visit before 5 pm on Friday?" | needs reply | no reply | Debatable: a polite question that is mainly a request for action |
| validation | Booking interpreters for a workshop | scheduling | sales | Debatable for the same reason as the catering |

At most 1 or 2 of the 398 test labels (0.5 points) are open to reasonable disagreement. They were left unchanged, because the published results were computed on this exact data.

## Design properties to be aware of

- **Tone.** The briefs gave pushy tone only to routine emails and calm tone to some urgent ones, so that urgent words would not mean urgent. Twins carry the text over to the opposite label, which softens this. Urgency words or exclamation marks appear in 9.7% of high-priority emails, 16.0% of normal and 26.9% of low (notices such as "Invoice settled!"). That is a mild inverse correlation, not a shortcut: the word-count baseline reaches 62% on priority.
- **"Next week" phrasing.** In 60 emails the relative phrase "next week" or "la semana que viene" describes a deadline that actually falls later the same week. This does not affect any label, because those deadlines are never on the day the email arrived, so priority stays normal. It is a small realism imprecision in the phrase generator.
- **Dropped briefs.** 24 of 3,200 briefs (0.75%) failed the fidelity check twice and were dropped. 18 of them were technical requests, mostly because the writer mentioned an incident it was told to avoid. Technical emails are still 693 of 3,176 (22%).
- **One writer, one checker.** The same model family (`gpt-6.1-sol`) wrote the text and checked it, and also ran the label-consistency check. The blind relabelling by a different family (Claude, 40 of 40) is the independent check, but it is a small sample: it bounds the error rate below about 9% with 95% confidence (one-sided Clopper-Pearson), rather than proving zero. No human labelled the data.
- **Explicit policy.** The policy is applied without exceptions. Real decisions are noisier, and reviewer disagreement is not represented.
- **Fictional names.** Sender names are random combinations of common first names and surnames, so the author's own name appears a few times by chance. All domains use `.example`.

## Round one

Round one was audited before this review, and its flaws are the reason round two exists; see [../../round1/README.md](../../round1/README.md). Its generator rebuilds every file byte for byte, and `validate_dataset.py` passes: splits, labels, grouping, hashes and cross-split near-duplicates. The flaw was difficulty, not label errors.

## Reproduce

```bash
python3 round2/audit/review_generation.py        # all automated checks; exits non-zero on any hard failure
cat round2/audit/blind_review_claude.json        # blind sample ids, blind labels, comparison
```
