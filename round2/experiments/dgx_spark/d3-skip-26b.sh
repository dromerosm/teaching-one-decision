#!/bin/bash
# Operator decision (2026-10-12): skip fine-tuning of the 26B d3 model. Its zero-shot test accuracy is
# 98.5% (95% interval 97.5-99.5%), at the label-consistency ceiling (99.5%), so a fine-tune cannot be
# shown to help on a 398-email test, and it would take about 5 GPU hours (likely a partial epoch under
# the 5-hour cap). This watcher stops that one training as soon as phase 3 starts it, then ends phase 3.
LOG=$(ls -t $HOME/dgx-ops/d3-phase3-*.log | head -1)
while ! grep -q "^TRAIN_START d3 lora" "$LOG"; do
  grep -q D3_PHASE3_EXIT "$LOG" && exit 0
  sleep 10
done
sleep 5
pkill -f "[d]3_train2.py --repo vllm-sr/d3 --revision"
sleep 5
tmux kill-session -t d3-phase3
echo "D3_TRAIN_SKIPPED d3 (26B) by operator decision $(date -Is): zero-shot 98.5% at the ceiling; fine-tune not needed" >> "$LOG"
echo "D3_PHASE3_EXIT $(date -Is) (ended after d3-flash; 26B fine-tune skipped)" >> "$LOG"
