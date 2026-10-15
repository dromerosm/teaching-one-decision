#!/bin/bash
# Read-only checks before changing anything on the server: identity, disk, memory, GPU, processes, tmux.
source "$(dirname "$0")/common.sh"
"$(dirname "$0")/remote.sh" 'set -u; hostname; whoami; uptime; df -h ~ | tail -1; free -g | sed -n 2p;
nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total,utilization.gpu --format=csv,noheader;
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader; tmux ls 2>&1 | head; python3 --version;
command -v uv >/dev/null && uv --version || echo "uv not installed (see https://docs.astral.sh/uv/)"'
