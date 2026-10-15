#!/bin/bash
# Open one SSH master connection and keep it open; the other scripts reuse it.
# You authenticate here yourself (password or key); no credential is stored or passed by the scripts.
# Leave this terminal open. Check it from another terminal with: ssh -O check -o ControlPath=... <user>@<host>
source "$(dirname "$0")/common.sh"
exec ssh -M -N "${SSH_OPTS[@]}" -o ServerAliveInterval=30 -o ServerAliveCountMax=3 "$TARGET"
