#!/bin/bash
# Run the 3 static cells of the 6-prefix many-prefixes test at M=64.
# Pair across both GPUs.
set -u
SEED=1
ROOT=/home/belal/150326
LOGS=$ROOT/logs
RESULTS=$ROOT/results
VLLM=$ROOT/venv-phase-2/bin/vllm
MODEL=allenai/OLMoE-1B-7B-0924-Instruct
PROMPTS=$ROOT/prompts/many_prefixes.jsonl

mkdir -p "$LOGS" "$RESULTS"

start_static() {
  local gpu=$1 port=$2 cache=$3 kv=$4 cell=$5
  CUDA_VISIBLE_DEVICES=$gpu setsid "$VLLM" serve "$MODEL" \
    --port $port \
    --expert-offload --expert-cache-size $cache \
    --enable-prefix-caching --enforce-eager --trust-remote-code \
    --max-model-len 4096 --max-num-batched-tokens 1 \
    --no-async-scheduling --attention-backend TRITON_ATTN \
    --block-size 1536 --num-gpu-blocks-override $kv \
    --gpu-memory-utilization 0.6 \
    > "$LOGS/${cell}_seed${SEED}_g${gpu}.log" 2>&1 &
  echo $!
}

wait_for_server() {
  local port=$1 pid=$2
  for i in $(seq 1 240); do
    if curl -sf -m 2 http://127.0.0.1:$port/v1/models >/dev/null 2>&1; then
      echo "[ready] port=$port after ${i}*2s"; return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then echo "[dead]"; return 1; fi
    sleep 2
  done
  echo "[timeout]"; return 1
}

capture_idle_mem() {
  local gpu=$1 cell=$2
  sleep 3
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i $gpu \
    > "$LOGS/${cell}_seed${SEED}_idle_mem.txt"
}

run_phase() {
  local port=$1 outname=$2
  "$VLLM" bench serve --backend vllm --host 127.0.0.1 --port $port \
    --endpoint /v1/completions --model "$MODEL" \
    --dataset-name custom --dataset-path "$PROMPTS" \
    --disable-shuffle --skip-chat-template \
    --custom-output-len 5 --num-prompts 6 \
    --max-concurrency 1 --num-warmups 0 --seed $SEED \
    --result-filename "$RESULTS/$outname" --save-result --trust-remote-code
}

shutdown() {
  local pid=$1
  if [ -z "${pid:-}" ]; then return; fi
  kill -TERM -$pid 2>/dev/null || kill -TERM $pid 2>/dev/null || true
  for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
    if ! kill -0 $pid 2>/dev/null; then return; fi; sleep 1
  done
  kill -KILL -$pid 2>/dev/null || kill -KILL $pid 2>/dev/null || true
}

run_cell() {
  local cell=$1 gpu=$2 port=$3 cache=$4 kv=$5
  local PID
  PID=$(start_static $gpu $port $cache $kv $cell)
  wait_for_server $port $PID || { shutdown $PID; return 1; }
  capture_idle_mem $gpu $cell
  run_phase $port "${cell}_phase1_seed${SEED}.json" \
    > "$LOGS/bench_${cell}_phase1_seed${SEED}_g${gpu}.log" 2>&1
  echo "[${cell} g${gpu}] phase1 (cold) exit: $?"
  run_phase $port "${cell}_phase2_seed${SEED}.json" \
    > "$LOGS/bench_${cell}_phase2_seed${SEED}_g${gpu}.log" 2>&1
  echo "[${cell} g${gpu}] phase2 (warm) exit: $?"
  shutdown $PID
}

echo "=== 6-prefix static cells at M=64 ==="
echo "  Round 1: static C=20 (GPU0) + static C=40 (GPU1) ==="
run_cell manyprefix6_static_C20 0 8000 20 44 &
run_cell manyprefix6_static_C40 1 8001 40 24 &
wait
echo "  Round 2: static C=60 (GPU0) ==="
run_cell manyprefix6_static_C60 0 8000 60 4
echo "=== 6-prefix static cells done ==="
