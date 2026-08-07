#!/bin/bash
# Memory-budget sweep. For each per-layer budget M, run 4 cells:
#   - static at 3 cache splits (cache=C, KV=K with C+K=M)
#   - unified at pool=M, init=middle
#
# Each cell: boot server, capture idle memory (apples-to-apples check),
# run Phase 1 (5 alternating prompts), run Phase 2 (6 random prompts),
# shut down. Pair cells across 2 GPUs for parallelism.
set -u
SEED=1
ROOT=/home/belal/150326
LOGS=$ROOT/logs
RESULTS=$ROOT/results
VLLM=$ROOT/venv-phase-2/bin/vllm
MODEL=allenai/OLMoE-1B-7B-0924-Instruct
PROMPTS=$ROOT/prompts/alternating_prompts.jsonl

mkdir -p "$LOGS" "$RESULTS"

start_static() {
  local gpu=$1 port=$2 cache=$3 kv=$4 cell=$5
  CUDA_VISIBLE_DEVICES=$gpu setsid "$VLLM" serve "$MODEL" \
    --port $port \
    --expert-offload --expert-cache-size $cache \
    --enable-prefix-caching --enforce-eager --trust-remote-code \
    --max-model-len 4096 --max-num-batched-tokens 1 \
    --no-async-scheduling --attention-backend TRITON_ATTN \
    --block-size 1536 \
    --num-gpu-blocks-override $kv \
    --gpu-memory-utilization 0.6 \
    > "$LOGS/${cell}_seed${SEED}_g${gpu}.log" 2>&1 &
  echo $!
}

start_unified() {
  local gpu=$1 port=$2 cache=$3 pool=$4 cell=$5
  CUDA_VISIBLE_DEVICES=$gpu setsid "$VLLM" serve "$MODEL" \
    --port $port \
    --expert-offload --expert-unified-pool --expert-cache-size $cache \
    --enable-prefix-caching --enforce-eager --trust-remote-code \
    --max-model-len 4096 --max-num-batched-tokens 1 \
    --no-async-scheduling --attention-backend TRITON_ATTN \
    --block-size 1536 --num-gpu-blocks-override $pool \
    > "$LOGS/${cell}_seed${SEED}_g${gpu}.log" 2>&1 &
  echo $!
}

wait_for_server() {
  local port=$1 pid=$2
  for i in $(seq 1 240); do
    if curl -sf -m 2 http://127.0.0.1:$port/v1/models >/dev/null 2>&1; then
      echo "[ready] port=$port after ${i}*2s"; return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "[dead] server pid=$pid port=$port"; return 1
    fi
    sleep 2
  done
  echo "[timeout] port=$port"; return 1
}

capture_idle_mem() {
  local gpu=$1 cell=$2
  sleep 3
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i $gpu \
    > "$LOGS/${cell}_seed${SEED}_idle_mem.txt"
}

run_phase1() {
  local port=$1 outname=$2
  "$VLLM" bench serve --backend vllm --host 127.0.0.1 --port $port \
    --endpoint /v1/completions --model "$MODEL" \
    --dataset-name custom --dataset-path "$PROMPTS" \
    --disable-shuffle --skip-chat-template \
    --custom-output-len 20 --num-prompts 5 \
    --max-concurrency 1 --num-warmups 1 --seed $SEED \
    --result-filename "$RESULTS/$outname" --save-result --trust-remote-code
}

run_phase2() {
  local port=$1 outname=$2
  "$VLLM" bench serve --backend vllm --host 127.0.0.1 --port $port \
    --endpoint /v1/completions --model "$MODEL" \
    --dataset-name random \
    --random-input-len 256 --random-output-len 80 \
    --random-range-ratio 0 --random-prefix-len 0 \
    --num-prompts 6 \
    --max-concurrency 1 --num-warmups 0 --seed $SEED \
    --result-filename "$RESULTS/$outname" --save-result --trust-remote-code
}

shutdown() {
  local pid=$1
  if [ -z "${pid:-}" ]; then return; fi
  kill -TERM -$pid 2>/dev/null || kill -TERM $pid 2>/dev/null || true
  for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
    if ! kill -0 $pid 2>/dev/null; then return; fi
    sleep 1
  done
  kill -KILL -$pid 2>/dev/null || kill -KILL $pid 2>/dev/null || true
  wait $pid 2>/dev/null || true
}

# args: cell, gpu, port, mode, cache, kv_or_pool
run_cell() {
  local cell=$1 gpu=$2 port=$3 mode=$4 cache=$5 kv_or_pool=$6
  local PID
  if [ "$mode" = "static" ]; then
    PID=$(start_static $gpu $port $cache $kv_or_pool $cell)
  else
    PID=$(start_unified $gpu $port $cache $kv_or_pool $cell)
  fi
  wait_for_server $port $PID || { shutdown $PID; return 1; }
  capture_idle_mem $gpu $cell

  run_phase1 $port "${cell}_phase1_seed${SEED}.json" \
    > "$LOGS/bench_${cell}_phase1_seed${SEED}_g${gpu}.log" 2>&1
  echo "[${cell} g${gpu}] phase1 exit: $?"
  run_phase2 $port "${cell}_phase2_seed${SEED}.json" \
    > "$LOGS/bench_${cell}_phase2_seed${SEED}_g${gpu}.log" 2>&1
  echo "[${cell} g${gpu}] phase2 exit: $?"
  shutdown $PID
}

# Budget table: M -> (cache_low, cache_mid, cache_high, unified_init)
# Min cache = 8 (OLMoE has top_k=8; cache<8 fails to forward).
# KV for static = M - cache. M=16 only has 2 viable splits.
declare -A BUDGETS=(
  [16]="8 12 8"
  [24]="8 12 16 12"
  [32]="8 16 24 16"
  [48]="16 24 40 24"
  [64]="20 40 64 40"
)

# Run order (sweep tight → comfortable so failures show up early)
ORDER="16 24 32 48 64"

for M in $ORDER; do
  vals=(${BUDGETS[$M]})
  if [ "$M" = "16" ]; then
    # M=16: only 2 static splits + 1 unified
    CL=${vals[0]} CH=${vals[1]} UI=${vals[2]}
    KL=$((M - CL)) KH=$((M - CH))
    echo ""
    echo "############################################"
    echo "### Budget M=$M  (static splits: ($CL,$KL)/($CH,$KH); unified init=$UI)"
    echo "############################################"
    echo "=== Round 1: static cache=$CL (GPU0) + static cache=$CH (GPU1) ==="
    run_cell budget${M}_static_C${CL} 0 8000 static $CL $KL &
    run_cell budget${M}_static_C${CH} 1 8001 static $CH $KH &
    wait
    echo "=== Round 2: unified pool=$M (GPU0) ==="
    run_cell budget${M}_unified_init${UI} 0 8000 unified $UI $M
    echo "=== Budget M=$M done ==="
  else
    CL=${vals[0]} CM=${vals[1]} CH=${vals[2]} UI=${vals[3]}
    KL=$((M - CL)) KM=$((M - CM)) KH=$((M - CH))
    echo ""
    echo "############################################"
    echo "### Budget M=$M  (static splits: ($CL,$KL)/($CM,$KM)/($CH,$KH); unified init=$UI)"
    echo "############################################"
    echo "=== Round 1: static cache=$CL (GPU0) + static cache=$CM (GPU1) ==="
    run_cell budget${M}_static_C${CL} 0 8000 static $CL $KL &
    run_cell budget${M}_static_C${CM} 1 8001 static $CM $KM &
    wait
    echo "=== Round 2: static cache=$CH (GPU0) + unified pool=$M (GPU1) ==="
    run_cell budget${M}_static_C${CH} 0 8000 static $CH $KH &
    run_cell budget${M}_unified_init${UI} 1 8001 unified $UI $M &
    wait
    echo "=== Budget M=$M done ==="
  fi
done

echo "=== Memory-budget sweep complete ==="
