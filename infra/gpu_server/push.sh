#!/bin/bash
# Copy local files to a directory on the GPU server through the master connection, then print their SHA-256 on both sides.
# Usage: infra/gpu_server/push.sh <REMOTE_SUBDIR> <local files...>
source "$(dirname "$0")/common.sh"
dest="$1"; shift
"$(dirname "$0")/remote.sh" "mkdir -p $REMOTE_ROOT/$dest"
scp -q "${SCP_OPTS[@]}" -o ControlMaster=no "$@" "$TARGET:$REMOTE_ROOT/$dest/"
shasum -a 256 "$@"
"$(dirname "$0")/remote.sh" "cd $REMOTE_ROOT/$dest && sha256sum $(for f in "$@"; do basename "$f"; done | tr '\n' ' ')"
