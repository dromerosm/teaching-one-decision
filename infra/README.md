# GPU infrastructure

Two environments were used. Everything here uses placeholders in angle brackets: fill them in for your own accounts and machines. Nothing in this folder is specific to the machines used for the article.

## Google Colab (Laya and Qwen3.5-9B)

Requirements: the [Colab CLI](https://github.com/googlecolab/google-colab-cli) installed and logged in, and compute units for a T4 runtime.

1. Build a job bundle. For round two:
   ```bash
   cd round2/experiments/colab && python3 prepare.py && python3 prepare2.py
   ```
   This writes `bundle.tar` and `bundle2.tar`, with the data and scripts the remote side needs. The data inside matches what the published runs used.
2. Run a job. The runner creates a session, uploads the bundle, installs the pinned stack ([setup_remote.py](colab/setup_remote.py)), runs your command, downloads the outputs and always stops the session, even after an error:
   ```bash
   python3 infra/colab/run_on_colab.py --session <SESSION_NAME> --gpu T4 \
       --bundle round2/experiments/colab/bundle2.tar \
       --command "python -u evaluate.py qwen_prompt" \
       --download qwen_prompt.json --with-causal-conv1d
   ```

The original orchestrators (`round2/experiments/colab/run_experiment*.py`) are kept exactly as they ran. They chain several jobs in one session, including the QLoRA training (`train_qlora.py`).

Lessons from the runs:

- Build `causal-conv1d` only for the T4 (`TORCH_CUDA_ARCH_LIST=7.5`). A build for every architecture took more than 45 minutes.
- Never build it while a 9B model is loaded or training. In one run the 12.7 GB of host RAM ran out and the machine was lost.
- T4 capacity is sometimes unavailable, and uploads occasionally fail with a broken pipe. The runner retries both.
- Plan training from speed measured after warm-up. Triton compiles kernels during the first steps. One QLoRA run planned from those steps and trained on only 448 of 1,985 emails.

## GPU server over SSH (Decision 3.0)

Our runs used a DGX Spark (NVIDIA GB10, 128 GB unified memory, aarch64), reached over SSH, optionally through a VPN client that exposes a local SOCKS5 proxy. Any Linux machine with a recent NVIDIA GPU and enough memory works; the 26B model needs about 50 GB.

1. `cp infra/gpu_server/config.example.env infra/gpu_server/config.env` and fill in the host, user, optional proxy, known-hosts file and remote directory. `config.env` is git-ignored.
2. Open one SSH master connection and leave it open. You authenticate yourself; the scripts never handle credentials:
   ```bash
   infra/gpu_server/ssh_master.sh
   ```
3. Run read-only checks before changing anything:
   ```bash
   infra/gpu_server/preflight.sh
   ```
4. Copy the scripts and the dataset to the server (checksums are printed on both sides):
   ```bash
   infra/gpu_server/push.sh setup infra/gpu_server/setup_env.sh infra/gpu_server/download_models.sh infra/gpu_server/run_decision3.sh
   infra/gpu_server/push.sh scripts round2/experiments/dgx_spark/d3_eval2.py round2/experiments/dgx_spark/d3_train2.py
   infra/gpu_server/push.sh dataset round2/dataset/*
   ```
5. On the server, inside tmux so that a dropped connection does not stop the work:
   ```bash
   infra/gpu_server/remote.sh 'cd <REMOTE_ROOT>/setup && tmux new -d -s d3 "REMOTE_ROOT=<REMOTE_ROOT> bash setup_env.sh && REMOTE_ROOT=<REMOTE_ROOT> bash download_models.sh && REMOTE_ROOT=<REMOTE_ROOT> bash run_decision3.sh"'
   ```
6. Copy the results back with `scp` through the same master connection, then place them in `round2/results/predictions/d3/` and run `round2/results/collect_results.py`.

Notes:

- No sudo is needed. The Python environment is user-level, on a uv-managed CPython that includes the headers Triton needs. The system Python on our server did not have them.
- The models load with `trust_remote_code=True`. The code was reviewed at the pinned revisions before use: weights load through safetensors, the network is used only for image or video URLs, and there is no subprocess or `exec`.
- If another service shares the GPU, check free memory first. We paused a resident inference service for cleaner timings.
- The original queue scripts (`round2/experiments/dgx_spark/*.sh`) are kept as they ran, with home paths replaced by `$HOME`.

Times measured on the GB10, including validation after each epoch: full fine-tuning of d3-edge took 40 minutes, d3-lite 81 and d3-nano 137; LoRA on d3-mini took 76 minutes and on d3-flash 109. The whole Decision 3.0 programme took about 12 GPU hours, including two failed attempts.
