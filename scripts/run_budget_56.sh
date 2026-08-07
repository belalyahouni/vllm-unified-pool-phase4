#!/bin/bash
# Add M=56 to the budget sweep to localise the cliff between M=48 and M=64.
# Reuses helper logic from run_budget_sweep.sh.
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
    --block-size 1536 --num-gpu-blocks-override $kv \
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

# M=56 splits: (16,40), (24,32), (40,16); unified init=28
M=56
echo ""
echo "############################################"
echo "### Budget M=$M  (static splits: (16,40)/(24,32)/(40,16); unified init=28)"
echo "############################################"
echo "=== Round 1: static cache=16 (GPU0) + static cache=24 (GPU1) ==="
run_cell budget${M}_static_C16 0 8000 static 16 40 &
run_cell budget${M}_static_C24 1 8001 static 24 32 &
wait
echo "=== Round 2: static cache=40 (GPU0) + unified pool=$M (GPU1) ==="
run_cell budget${M}_static_C40 0 8000 static 40 16 &
run_cell budget${M}_unified_init28 1 8001 unified 28 $M &
wait
echo "=== Budget M=$M done ==="
