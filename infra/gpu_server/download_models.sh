#!/bin/bash
# Run ON the GPU server. Downloads the six Decision 3.0 models at the revisions used in the article (~86 GB).
set -euo pipefail
ROOT=${REMOTE_ROOT:?set REMOTE_ROOT}
HF_HUB_DISABLE_TELEMETRY=1 "$ROOT/venv/bin/python" - <<'PY'
from huggingface_hub import snapshot_download
pins = {
    "vllm-sr/d3-edge": "20c745653400d191ecacd99c4eafc0c5900dc15e",
    "vllm-sr/d3-lite": "3c49f5c0bd48a4b28e29b8ec1f38f56e199f3d61",
    "vllm-sr/d3-nano": "9d9b937f5f117f78c030ea7ba677c0b3d328ec58",
    "vllm-sr/d3-mini": "0e78bbdbe3cf480ed1461c185c9e0c4fa9258011",
    "vllm-sr/d3-flash": "f6075bff3c0f7d5704e353d0fb2ebf94888d1691",
    "vllm-sr/d3": "093cffbaa7be6f5dfe14252ae7e80c396c06dbd9",
}
for repo, rev in pins.items():
    print("DOWNLOADED", repo, rev, snapshot_download(repo, revision=rev), flush=True)
PY
