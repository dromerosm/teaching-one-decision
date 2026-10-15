#!/bin/bash
# Phase 2: prompt v2 and few-shot variants for all six d3 models, then fine-tuning each model and
# evaluating the fine-tuned package on the test split. Waits for the zero-shot run to finish first.
set -uo pipefail
ROOT=$HOME/d3-eval-20261012
OPS=$HOME/dgx-ops/d3-20261012
PY=$ROOT/venv-managed/bin/python
OUT=$OPS/results; mkdir -p "$OUT" "$OPS/train"
LOG=$HOME/dgx-ops/d3-phase2-$(date +%F_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1
echo "D3_PHASE2_START $(date -Is) log=$LOG"
while ! grep -q D3_EVAL_EXIT $HOME/dgx-ops/d3-eval-*.log; do sleep 30; done
echo "ZERO_SHOT_DONE $(date -Is)"
declare -A PIN=(
  [d3-edge]=20c745653400d191ecacd99c4eafc0c5900dc15e [d3-lite]=3c49f5c0bd48a4b28e29b8ec1f38f56e199f3d61
  [d3-nano]=9d9b937f5f117f78c030ea7ba677c0b3d328ec58 [d3-mini]=0e78bbdbe3cf480ed1461c185c9e0c4fa9258011
  [d3-flash]=f6075bff3c0f7d5704e353d0fb2ebf94888d1691 [d3]=093cffbaa7be6f5dfe14252ae7e80c396c06dbd9 )
MODELS="d3-edge d3-lite d3-nano d3-mini d3-flash d3"
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
for m in $MODELS; do
  for v in v2 fewshot; do
    echo "VARIANT_START $m $v $(date -Is)"
    $PY $OPS/d3_eval2.py vllm-sr/$m ${PIN[$m]} $OUT/$m-$v.json $OPS/data questions_$v.json || echo "VARIANT_FAILED $m $v"
  done
done
for m in $MODELS; do
  case $m in d3-edge|d3-lite|d3-nano) mode=full; budget=10800;; d3) mode=lora; budget=18000;; *) mode=lora; budget=10800;; esac
  echo "TRAIN_START $m $mode $(date -Is) available_GiB=$(free -g | awk '/^Mem:/{print $7}')"
  if $PY $OPS/d3_train.py --repo vllm-sr/$m --revision ${PIN[$m]} --mode $mode --out $OPS/train/$m --data $OPS/data --budget $budget; then
    echo "TRAIN_DONE $m $(date -Is)"
    $PY $OPS/d3_eval2.py $OPS/train/$m/package local $OUT/$m-ft.json $OPS/data questions.json || echo "FT_EVAL_FAILED $m"
  else
    echo "TRAIN_FAILED $m"
  fi
done
echo "D3_PHASE2_EXIT $(date -Is)"
