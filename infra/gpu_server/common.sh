#!/bin/bash
# Shared settings for the GPU-server scripts. Reads infra/gpu_server/config.env.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[ -f "$HERE/config.env" ] || { echo "Create $HERE/config.env from config.example.env first"; exit 2; }
# shellcheck disable=SC1091
source "$HERE/config.env"
CONTROL_PATH="$HERE/.ssh-master.sock"
COMMON_OPTS=(-o "ControlPath=$CONTROL_PATH" -o "UserKnownHostsFile=$KNOWN_HOSTS" -o StrictHostKeyChecking=yes -o ConnectTimeout=15)
if [ -n "${SOCKS_PROXY:-}" ]; then
  COMMON_OPTS+=(-o "ProxyCommand=nc -X 5 -x $SOCKS_PROXY %h %p")
fi
SSH_OPTS=("${COMMON_OPTS[@]}" -p "$SSH_PORT")
SCP_OPTS=("${COMMON_OPTS[@]}" -P "$SSH_PORT")
TARGET="$GPU_USER@$GPU_HOST"
