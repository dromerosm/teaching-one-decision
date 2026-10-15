# Round two: a test built to resist shortcuts

This is the test behind the article's conclusions. It replaced round one after an audit showed that round one could be solved by word counts. See [../round1/README.md](../round1/README.md).

## Dataset

The full description, label policy and statistics are in [dataset/DATASET_CARD.md](dataset/DATASET_CARD.md). In short:

- **Size:** 3,176 synthetic emails. There are 1,985 for training, 395 for validation, 398 for calibration and 398 for the test.
- **Split:** by customer account. The test accounts, request types and notice types never appear in training.
- **Header:** each input starts with a header from the triage system (when the email arrived, sender, service level, open incidents), and priority depends on it.
- **Twins:** 620 emails are counterfactual twins, with the same text and a different header that changes the label.
- **Text and labels:** `gpt-6.1-sol` wrote the text from hand-written briefs. Labels come from the briefs, never from a model.

## How it was built

| Step | Script | Output |
|---|---|---|
| Hand-written accounts, requests, notices | [generator/scenarios.py](generator/scenarios.py) | — |
| Deterministic specs and labels (seed 20261012) | [generator/specs.py](generator/specs.py) | `specs.jsonl`, `assignment.json` |
| Email text from each brief, with programmatic and LLM fidelity checks | [generator/generate.py](generator/generate.py) | `generation_cache.jsonl` (every accepted email) |
| Header + email into model inputs and split files | [generator/assemble.py](generator/assemble.py) | `dataset/*.jsonl`, `questions.json` |
| Token lengths with the pinned Laya tokenizer | [generator/check_tokenization.py](generator/check_tokenization.py) | `dataset/tokenizer_check.json` |

`generate.py` reads `OPENAI_API_KEY`. With the published cache it makes no API calls.

## Audits run before any model

[audit/audit.py](audit/audit.py) runs the shortcut baselines: majority class, keyword rules, header-aware rules and a word-count Naive Bayes trained on the training split. It also checks sentence overlap between test and training. [audit/ceiling.py](audit/ceiling.py) asks `gpt-6.1-sol` to label emails from the written policy alone.

| Check (test split) | Result |
|---|---:|
| Naive Bayes, all three fields | 52.01% |
| Keyword rules / header-aware rules | 27.39% / 27.89% |
| Test sentences found word for word in training | 8.2% |
| `gpt-6.1-sol`, policy only (after label fixes) | 99.50% |

[audit/GENERATION_REVIEW.md](audit/GENERATION_REVIEW.md) reports an independent review of the generation: every label recomputed from the header, every deadline phrase checked against its date, every twin pair checked, splits and near-duplicates, mutation tests of the checks themselves, the remaining label disagreements read one by one, and a blind relabelling of 40 test emails by a second model family (40 of 40 agree).

`ceiling_all_before_label_fixes.json` is the first full run of the label check. It exposed 11 ambiguous notice topics and 140 requests that could not sensibly be "done without a reply". Those were fixed and regenerated before any model was evaluated. `ceiling_test_all.json` and `ceiling_validation_all.json` are the checks after the fixes.

## Experiments

| Folder | Models | Hardware | What |
|---|---|---|---|
| [experiments/colab](experiments/colab) | Laya multilingual, Qwen3.5-9B | Colab T4 | Zero-shot, prompt v2, few-shot, thinking, full fine-tuning (Laya), QLoRA (Qwen) |
| [experiments/dgx_spark](experiments/dgx_spark) | Decision 3.0 (six sizes) | DGX Spark GB10 | Zero-shot, prompt v2, few-shot, full fine-tuning (three smallest), LoRA (mini, flash) |

Scripts are kept as they ran. Some carry a version suffix where a fix was needed:

- `d3_eval2.py` reads the single yes/no probability that Decision 3.0 returns. Version 1 did not, and its zero-shot `needs_reply` predictions are recomputed from the saved raw answers in `collect_results.py`.
- `d3_train2.py` uses lower learning rates with warm-up, after the first recipe made d3-edge collapse, and it saves packages safely. `d3_debug*.py` are the diagnostics behind that change.
- `train_qlora_runA.py` is the QLoRA version whose speed planning stopped at 448 emails. `train_qlora.py` is the corrected version used for the full-epoch run.
- The 26B Decision 3.0 model was deliberately not fine-tuned, because it already scored 98.5% without examples (`d3-skip-26b.sh`).

The prompts are shared: `questions.json` holds the written policy, `questions_v2.json` holds the same policy as a procedure, and `questions_fewshot.json` holds prompt v2 plus the eight examples in `fewshot.json`. Qwen receives the same text through `experiments/colab/qwen_common.py`.

## Results

- [results/RESULTS.md](results/RESULTS.md): accuracy with 95% intervals (bootstrap over test accounts), per-field accuracy, high-priority recall and false alarms, twin-pair consistency, training time, data seen, inference latency, load time and memory, for all 31 configurations.
- [results/predictions](results/predictions): every model's answer for every test email, with timings. Decision 3.0 files also keep the raw answer probabilities.
- [results/training](results/training): training reports (loss by update, validation by epoch, schedule, memory, self-checks).
- `python3 results/collect_results.py` rebuilds `RESULTS.md` and `results_all.json` from the predictions.
