# Teaching one decision

![A stack of reviewed emails feeds a few selected examples into a small glowing glass cube, which sends out three sorted streams; six glass blocks of increasing size stand behind it.](docs/figures/hero-teaching-one-decision.png)

Code, data and every measurement behind the article **[Private company data could make small models much more useful](https://diegoromero.es/en/articles/private-company-data-small-models/)** by Diego Romero.

The question: can a model learn one business decision from examples of that decision? The task is email triage. Each email gets three answers: which team handles it, how urgent it is, and whether the sender expects a written reply. Three families of models were tested: [Laya multilingual](https://huggingface.co/convaiinnovations/laya-multilingual), [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) and the six [Decision 3.0](https://huggingface.co/collections/vllm-sr/decision-30) models. Each was taught the decision in up to five ways: the written policy, a clearer prompt, examples in the prompt, step-by-step thinking and fine-tuning. Everything was run on the same synthetic test.

All data is synthetic. No private company data was used.

## Results in one table

All three fields right, on the 398 test emails of round two. The full record, with confidence intervals, per-field accuracy, timings and memory, is in [round2/results/RESULTS.md](round2/results/RESULTS.md).

| Model | Size | Written policy | Best prompt | Fine-tuned |
|---|---:|---:|---:|---:|
| Laya | 0.32B | 8.5% | — | 92.5% |
| d3-edge | 0.59B | 17.3% | 17.3% | 92.7% |
| d3-lite | 0.85B | 32.7% | 32.7% | 90.5% |
| d3-nano | 2.2B | 56.5% | 57.3% | 92.5% |
| d3-mini | 4.5B | 72.6% | 74.9% | 94.5% (LoRA) |
| d3-flash | 8.4B | 79.4% | 84.2% | 99.2% (LoRA) |
| Qwen3.5-9B | 9B | 53.3% | 61.3% | 99.0% (QLoRA) |
| d3 | 26B | 98.5% | 98.7% | not run (already at the ceiling) |

For reference, a word-count classifier scores 52.0%. `gpt-6.1-sol`, given only the written policy, recovers 99.5% of the labels, which shows that they follow from the input.

## What is in this repository

```
round1/        First dataset and experiment, and the audit that showed the test was too easy
  dataset/     Generator, data and validation for 3,200 emails from 80 topic families
  experiment/  Laya fine-tuning and evaluation scripts (Colab T4)
  results/     Per-email predictions and training report
  audit/       Word-count baseline that scored 84%
round2/        The test used for the article's conclusions
  generator/   Hand-written scenarios, deterministic specs and labels, gpt-6.1-sol writer and checker,
               and the cache of every accepted email, so the dataset rebuilds without API calls
  dataset/     3,176 emails split by customer account, policy files and the few-shot examples
  audit/       Shortcut baselines, overlap checks and the policy-only label-consistency check
  experiments/
    colab/     Laya and Qwen3.5-9B (prompting, thinking, QLoRA) on a Colab T4, exactly as run
    dgx_spark/ Decision 3.0 evaluation and fine-tuning on a DGX Spark (GB10), exactly as run
  results/     Per-email predictions for all 31 configurations, training reports, RESULTS.md
infra/         Templates with placeholders to set up the two GPU environments used
docs/figures/  Hero image, diagrams and their sources
```

## Check the published numbers (no GPU, no API)

Python 3.10 or newer, standard library only.

```bash
shasum -a 256 -c SHA256SUMS                   # every file matches the published checksums
python3 round2/results/collect_results.py     # rebuilds RESULTS.md and results_all.json from the predictions
python3 round2/audit/audit.py                 # baselines and overlap checks (writes audit.json)
python3 round1/audit/baseline_naive_bayes.py  # the round-one word-count baseline (84.00%)
python3 round2/audit/review_generation.py     # independent review of the dataset generation (labels, phrases, twins, splits)
```

To rebuild the round-two dataset from its specs and the generation cache, with no API calls:

```bash
cd round2/generator && python3 specs.py && python3 assemble.py
cd ../dataset && shasum -a 256 -c ASSEMBLED.sha256
```

The round-one dataset rebuilds with `python3 round1/dataset/build_dataset.py`. Both rebuilds were checked to be byte-identical to the published files. How the round-two labels and text were verified, including a blind relabelling by a second model family, is in [round2/audit/GENERATION_REVIEW.md](round2/audit/GENERATION_REVIEW.md).

## Run the experiments again

- **Laya and Qwen on Google Colab:** [infra/colab](infra/colab) has a runner that creates a T4 session with the Colab CLI, uploads a job bundle, installs the pinned stack, runs the job, downloads the outputs and always stops the session. Bundles are built by [round2/experiments/colab/prepare.py](round2/experiments/colab/prepare.py) and `prepare2.py`.
- **Decision 3.0 on a GPU server:** [infra/gpu_server](infra/gpu_server) has SSH helpers, an environment set-up script that needs no sudo, pinned model downloads and the full evaluation and fine-tuning queue. Our runs used a DGX Spark (NVIDIA GB10, 128 GB unified memory). The 26B model needs about 50 GB.

[infra/README.md](infra/README.md) explains both. Model revisions are pinned everywhere. Expect small numeric differences on other hardware, drivers or library versions.

Regenerating the dataset text itself calls the OpenAI API (`OPENAI_API_KEY`). The published cache makes that unnecessary.

## Honest limits

- The data is synthetic, and the labels follow a policy applied without exceptions. Real decisions are noisier.
- Each configuration ran once, with one seed.
- The thinking run covered 44 emails before its time limit.
- Laya and Qwen ran on a T4 and Decision 3.0 on a GB10, so speeds are comparable only within the same hardware.
- Model weights are not included. The fine-tuning scripts and training reports are, and any run can be repeated from them.
- The d3 trainer is our own. The Decision 3.0 models ship inference code only, and the article describes how the trainer was checked against their runtime.

## Licences

- Code: [Apache License 2.0](LICENSE).
- Datasets, results and figures: [CC BY 4.0](LICENSE-DATA).
- The models evaluated keep their own licences; Laya multilingual, Qwen3.5-9B and Decision 3.0 are all published under Apache-2.0 on Hugging Face.

All people, companies, email addresses and events in the datasets are fictional. Names come from random combinations of common first names and surnames, so the author's own name appears a few times by chance. Domains use the reserved `.example` suffix.

## Citation

See [CITATION.cff](CITATION.cff). Codex and Claude Code assisted with the dataset-generation, training and evaluation scripts, the analysis and the environment management. The author designed the experiments and reviewed the figures.
