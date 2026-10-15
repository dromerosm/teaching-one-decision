# Round one: a result that did not hold up

This folder keeps the first experiment exactly as it ran, because its audit is part of the story.

## What was done

- **Dataset** ([dataset/](dataset)): 3,200 synthetic emails from 80 bilingual topic families, with 40 emails per family and labels from fixed scenario rules. Whole families went to one split: 2,000 for training and 400 each for validation, calibration and test. `build_dataset.py` rebuilds every file byte for byte, and `validate_dataset.py` checks the splits, labels and hashes.
- **Experiment** ([experiment/](experiment)): full fine-tuning of [Laya multilingual](https://huggingface.co/convaiinnovations/laya-multilingual) (four epochs, best validation epoch kept) and an evaluation of Laya before and after training and of Qwen3.5-9B, on a Colab T4.

| Model | All three fields (test) |
|---|---:|
| Laya, as published | 16.75% |
| Laya, fine-tuned | 88.25% |
| Qwen3.5-9B | 80.00% |

## What the audit found

| Check | Result |
|---|---|
| Word-count Naive Bayes trained on the training split ([audit/baseline_naive_bayes.py](audit/baseline_naive_bayes.py)) | 84.00% |
| One regular expression for "in N minutes" or "live service" | Matches all 1,200 high-priority emails and nothing else |
| Distractor paragraphs (quoted "URGENT" thread, injected instruction) | Identical in every split; most of Laya's lead over Qwen came from those emails |
| Qwen's fast kernels | Not installed; the speed comparison was overstated |

Every reported number matched the raw predictions. The test was simply too easy to show what the models had learned, so [round two](../round2) replaced it.

## Files

- `results/`: per-email predictions (`laya_original.json`, `laya_trained.json`, `qwen.json`), `summary.json`, `train_report.json` and the run protocol.
- `audit/`: the word-count baseline and its output.

The dataset generator and the experiment scripts are kept as they ran, apart from two path constants changed for this repository layout. `validate_dataset.py` optionally compares against a 50-email smoke set that is not published; it reports that reference as unavailable and still validates everything else. Running it rewrites `validation_report.json` with that field set to unavailable; the published report is the original run, made with the smoke set present.
