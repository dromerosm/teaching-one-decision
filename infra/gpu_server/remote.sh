#!/bin/bash
# Run a command on the GPU server through the open master connection.
# Usage: infra/gpu_server/remote.sh 'nvidia-smi; df -h ~'
source "$(dirname "$0")/common.sh"
ssh -O check "${SSH_OPTS[@]}" "$TARGET" 2>/dev/null || { echo "No master connection: run ssh_master.sh first"; exit 3; }
exec ssh "${SSH_OPTS[@]}" -o ControlMaster=no -o BatchMode=yes "$TARGET" "$@"
