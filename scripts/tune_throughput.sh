#!/usr/bin/env bash
set -euo pipefail

# Simple tuner: try from high to low settings and pick the fastest that runs.
# Usage: scripts/tune_throughput.sh [gpu_id]

GPU_ID="${1:-0}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

# SDPA backend: default safe math; allow override via environment before calling
: "${TECGPT_SDPA:=math}"
: "${TECGPT_ENABLE_GC:=0}"
unset TECGPT_NODE_SAFE_CAP  # disable chunk safety cap for max throughput during tuning

BASE_CMD=(python Run.py -dataset GIMtec -mode ori -model TEC_MoLLM -epochs 1 -amp True -accumulate_steps 6 --use_lora True -log_step 10)
WD="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WD/model"

# Attempt combinations high -> low
BATCHES=(64 48)
CHUNKS=(5184 4096 3072)
# Fallback空间（高失败后继续下降，确保至少找到可跑的组合）
FB_BATCHES=(48 40 32 24)
FB_CHUNKS=(2048 1536 1024 768 512 384 256)

mkdir -p "$WD/Output/GIMtec/TEC_MoLLM/tune"
ts() { date +%Y%m%d_%H%M%S; }

kill_existing() {
  # Kill any stale Run.py processes on this GPU to avoid OOM interference
  pkill -f "python .*Run.py" || true
}

measure() {
  local log_file="$1"
  # Count how many training log lines we saw
  local cnt
  cnt=$(grep -c "Train Epoch" "$log_file" || true)
  # crude duration since we wrap with 'time -p'
  local dur
  dur=$(grep -E "^real " "$log_file" | awk '{print $2}' | tail -n1)
  echo "$cnt" "$dur"
}

best_combo=""
best_rate=0
for bs in "${BATCHES[@]}"; do
  for nc in "${CHUNKS[@]}"; do
    echo "[tune] Trying batch_size=$bs node_chunk=$nc (SDPA=$TECGPT_SDPA, GC=$TECGPT_ENABLE_GC)"
    kill_existing
    LOG="$WD/Output/GIMtec/TEC_MoLLM/tune/$(ts)_bs${bs}_nc${nc}.log"
    # Run up to 120s to avoid long stalls; record time using 'time -p'
    set +e
    start_ts=$(date +%s)
    (timeout 120s stdbuf -oL -eL \
      "${BASE_CMD[@]}" -batch_size "$bs" --node_chunk "$nc" 2>&1) | tee "$LOG"
    code=${PIPESTATUS[0]}
    end_ts=$(date +%s)
    echo "real $((end_ts-start_ts))" >> "$LOG"
    set -e
    if grep -qi "out of memory" "$LOG"; then
      echo "[tune] OOM at bs=$bs nc=$nc → fallback"
      continue
    fi
    if grep -qi "invalid configuration argument" "$LOG"; then
      echo "[tune] SDPA kernel error at bs=$bs nc=$nc → fallback"
      continue
    fi
    # Parse throughput proxy: training log lines per second
    read cnt dur < <(measure "$LOG")
    # Fallback if duration missing; assume 120s timeout
    if [[ -z "$dur" ]]; then dur=120; fi
    # Each log roughly every 10 steps; rate = cnt/dur
    rate=$(python - <<PY
cnt=$cnt
dur=float($dur)
print(cnt/dur if dur>0 else 0.0)
PY
)
    echo "[tune] rate(lines/s)=$rate from cnt=$cnt in ${dur}s"
    awk -v r="$rate" -v b="$bs" -v n="$nc" 'BEGIN { printf("[tune] score bs=%s nc=%s → %.6f\n", b, n, r) }'
    # Choose the best by highest rate
    pycmp=$(python - <<PY
cur=$rate
best=$best_rate if "$best_rate"!="" else 0.0
print(1 if cur>best else 0)
PY
)
    if [[ "$pycmp" == "1" ]]; then
      best_rate="$rate"; best_combo="bs=$bs nc=$nc"
    fi
  done
done

if [[ -z "$best_combo" ]]; then
  echo "[tune] Primary space all failed; enabling safety cap and retrying fallback space..."
  export TECGPT_NODE_SAFE_CAP=${TECGPT_NODE_SAFE_CAP:-16384}
  for bs in "${FB_BATCHES[@]}"; do
    for nc in "${FB_CHUNKS[@]}"; do
      echo "[tune] Trying FB batch_size=$bs node_chunk=$nc (SDPA=$TECGPT_SDPA, GC=$TECGPT_ENABLE_GC, SAFE_CAP=$TECGPT_NODE_SAFE_CAP)"
      kill_existing
      LOG="$WD/Output/GIMtec/TEC_MoLLM/tune/$(ts)_FB_bs${bs}_nc${nc}.log"
      set +e
      start_ts=$(date +%s)
      (timeout 120s stdbuf -oL -eL \
        "${BASE_CMD[@]}" -batch_size "$bs" --node_chunk "$nc" 2>&1) | tee "$LOG"
      code=${PIPESTATUS[0]}
      end_ts=$(date +%s)
      echo "real $((end_ts-start_ts))" >> "$LOG"
      set -e
      if grep -qi "out of memory" "$LOG"; then
        echo "[tune] OOM at FB bs=$bs nc=$nc → fallback"
        continue
      fi
      if grep -qi "invalid configuration argument" "$LOG"; then
        echo "[tune] SDPA kernel error at FB bs=$bs nc=$nc → fallback"
        continue
      fi
      read cnt dur < <(measure "$LOG")
      if [[ -z "$dur" ]]; then dur=120; fi
      rate=$(python - <<PY
cnt=$cnt
dur=float($dur)
print(cnt/dur if dur>0 else 0.0)
PY
)
      echo "[tune] FB rate(lines/s)=$rate from cnt=$cnt in ${dur}s"
      pycmp=$(python - <<PY
cur=$rate
best=$best_rate if "$best_rate"!="" else 0.0
print(1 if cur>best else 0)
PY
)
      if [[ "$pycmp" == "1" ]]; then
        best_rate="$rate"; best_combo="bs=$bs nc=$nc (FB)"
      fi
    done
  done
fi

echo "[tune] Best combo: $best_combo with rate=$best_rate"
exit 0
