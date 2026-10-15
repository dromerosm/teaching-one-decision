#!/bin/bash
# Zero-shot evaluation of the six Decision 3.0 models, smallest first, one process per model.
set -uo pipefail
ROOT=$HOME/d3-eval-20261012
OPS=$HOME/dgx-ops/d3-20261012
OUT=$OPS/results
mkdir -p "$OUT"
LOG=$HOME/dgx-ops/d3-eval-$(date +%F_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1
echo "D3_EVAL_START $(date -Is) log=$LOG"
declare -A PIN=(
  [vllm-sr/d3-edge]=20c745653400d191ecacd99c4eafc0c5900dc15e
  [vllm-sr/d3-lite]=3c49f5c0bd48a4b28e29b8ec1f38f56e199f3d61
  [vllm-sr/d3-nano]=9d9b937f5f117f78c030ea7ba677c0b3d328ec58
  [vllm-sr/d3-mini]=0e78bbdbe3cf480ed1461c185c9e0c4fa9258011
  [vllm-sr/d3-flash]=f6075bff3c0f7d5704e353d0fb2ebf94888d1691
  [vllm-sr/d3]=093cffbaa7be6f5dfe14252ae7e80c396c06dbd9
)
fails=0
for repo in vllm-sr/d3-edge vllm-sr/d3-lite vllm-sr/d3-nano vllm-sr/d3-mini vllm-sr/d3-flash vllm-sr/d3; do
  name=${repo#vllm-sr/}
  avail=$(free -g | awk '/^Mem:/{print $7}')
  echo "MODEL_START $name $(date -Is) available_GiB=$avail"
  if [ "$name" = d3 ] && [ "$avail" -lt 60 ]; then echo "SKIP $name: only ${avail} GiB available"; continue; fi
  HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 "$ROOT/venv-managed/bin/python" "$OPS/d3_eval.py" "$repo" "${PIN[$repo]}" "$OUT/$name.json" "$OPS/data" \
    || { echo "MODEL_FAILED $name"; fails=$((fails+1)); }
  curl -fsS --max-time 10 http://127.0.0.1:8000/health >/dev/null 2>&1 && echo "QWEN_SERVICE_HEALTHY" || echo "QWEN_SERVICE_CHECK_FAILED"
done
echo "D3_EVAL_EXIT failures=$fails"
