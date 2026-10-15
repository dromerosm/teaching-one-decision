#!/bin/bash
# Run ON the GPU server, inside tmux. Zero-shot, prompt v2 and few-shot evaluation of the six
# Decision 3.0 models, then fine-tuning and evaluation (full fine-tune for the three smallest,
# LoRA for mini and flash; the 26B model is evaluated but not fine-tuned, as in the article).
# Expects $REMOTE_ROOT/dataset (round2/dataset) and $REMOTE_ROOT/scripts (round2/experiments/dgx_spark).
set -uo pipefail
ROOT=${REMOTE_ROOT:?set REMOTE_ROOT}
PY=$ROOT/venv/bin/python; S=$ROOT/scripts; D=$ROOT/dataset; OUT=$ROOT/results; mkdir -p "$OUT" "$ROOT/train" "$ROOT/logs"
LOG=$ROOT/logs/decision3-$(date +%F_%H%M%S).log; exec > >(tee -a "$LOG") 2>&1
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
declare -A PIN=(
  [d3-edge]=20c745653400d191ecacd99c4eafc0c5900dc15e [d3-lite]=3c49f5c0bd48a4b28e29b8ec1f38f56e199f3d61
  [d3-nano]=9d9b937f5f117f78c030ea7ba677c0b3d328ec58 [d3-mini]=0e78bbdbe3cf480ed1461c185c9e0c4fa9258011
  [d3-flash]=f6075bff3c0f7d5704e353d0fb2ebf94888d1691 [d3]=093cffbaa7be6f5dfe14252ae7e80c396c06dbd9 )
for m in d3-edge d3-lite d3-nano d3-mini d3-flash d3; do
  $PY $S/d3_eval2.py vllm-sr/$m ${PIN[$m]} $OUT/$m.json $D questions.json || echo "EVAL_FAILED $m zero-shot"
  $PY $S/d3_eval2.py vllm-sr/$m ${PIN[$m]} $OUT/$m-v2.json $D questions_v2.json || echo "EVAL_FAILED $m v2"
  $PY $S/d3_eval2.py vllm-sr/$m ${PIN[$m]} $OUT/$m-fewshot.json $D questions_fewshot.json || echo "EVAL_FAILED $m fewshot"
done
for m in d3-edge d3-lite d3-nano d3-mini d3-flash; do
  case $m in d3-edge|d3-lite|d3-nano) mode=full;; *) mode=lora;; esac
  if $PY $S/d3_train2.py --repo vllm-sr/$m --revision ${PIN[$m]} --mode $mode --out $ROOT/train/$m --data $D --budget 10800; then
    $PY $S/d3_eval2.py $ROOT/train/$m/package local $OUT/$m-ft.json $D questions.json || echo "FT_EVAL_FAILED $m"
  else
    echo "TRAIN_FAILED $m"
  fi
done
echo "DECISION3_EXIT $(date -Is)"
