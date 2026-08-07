#!/bin/bash
# Many-prefixes test with 7 prompts (instead of 10) — unified only.
# 7 distinct prefixes × 2 blocks each = 14 prefix blocks total.
# This matches the max simultaneous prefix-cache capacity we observed at
# pool=64 init=40 in the 10-prefix run. So Phase 2 SHOULD hit cache cleanly
# at 7 prefixes if the LRU race only loses prefix above ~14 blocks.
#
# Both passes (latency on GPU 0, trace on GPU 1) in parallel.
set -u
SEED=1
ROOT=/home/belal/150326
LOGS=$ROOT/logs
RESULTS=$ROOT/results
VLLM=$ROOT/venv-phase-2/bin/vllm
MODEL=allenai/OLMoE-1B-7B-0924-Instruct
PROMPTS=$ROOT/prompts/many_prefixes.jsonl   # has 10 entries; we use --num-prompts 7

mkdir -p "$LOGS" "$RESULTS"

start_unified() {
  local gpu=$1 port=$2 trace=$3 cell=$4
  local logsuffix=""
  local env_prefix=""
  if [ "$trace" = "1" ]; then
    env_prefix="VLLM_UNIFIED_POOL_TRACE=1"
    logsuffix="_trace"
  fi
  env $env_prefix CUDA_VISIBLE_DEVICES=$gpu setsid "$VLLM" serve "$MODEL" \
    --port $port \
    --expert-offload --expert-unified-pool --expert-cache-size 40 \
    --enable-prefix-caching --enforce-eager --trust-remote-code \
    --max-model-len 4096 --max-num-batched-tokens 1 \
    --no-async-scheduling --attention-backend TRITON_ATTN \
    --block-size 1536 --num-gpu-blocks-override 64 \
    > "$LOGS/${cell}_seed${SEED}_g${gpu}${logsuffix}.log" 2>&1 &
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
    --custom-output-len 5 --num-prompts 7 \
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

echo "=== 7-prefix unified test (pool=64 init=40) — latency (GPU0) + trace (GPU1) ==="
PID0=$(start_unified 0 8000 0 manyprefix7_unified_init40)
PID1=$(start_unified 1 8001 1 manyprefix7_unified_init40)
wait_for_server 8000 $PID0 || { shutdown $PID0; shutdown $PID1; exit 1; }
wait_for_server 8001 $PID1 || { shutdown $PID0; shutdown $PID1; exit 1; }
capture_idle_mem 0 manyprefix7_unified_init40

# Latency pass on GPU 0 (kept JSONs); trace pass on GPU 1 (discard JSONs, keep server log).
run_phase 8000 manyprefix7_unified_init40_phase1_seed${SEED}.json \
  > "$LOGS/bench_manyprefix7_unified_init40_phase1_seed${SEED}_g0.log" 2>&1 &
B0=$!
run_phase 8001 _discard_manyprefix7_unified_trace_phase1_seed${SEED}.json \
  > "$LOGS/bench_manyprefix7_unified_trace_phase1_seed${SEED}_g1.log" 2>&1 &
B1=$!
wait $B0; echo "[g0] phase1 (cold) exit: $?"
wait $B1; echo "[g1] phase1 trace exit: $?"

run_phase 8000 manyprefix7_unified_init40_phase2_seed${SEED}.json \
  > "$LOGS/bench_manyprefix7_unified_init40_phase2_seed${SEED}_g0.log" 2>&1 &
B0=$!
run_phase 8001 _discard_manyprefix7_unified_trace_phase2_seed${SEED}.json \
  > "$LOGS/bench_manyprefix7_unified_trace_phase2_seed${SEED}_g1.log" 2>&1 &
B1=$!
wait $B0; echo "[g0] phase2 (warm) exit: $?"
wait $B1; echo "[g1] phase2 trace exit: $?"

shutdown $PID0; shutdown $PID1
rm -f "$RESULTS/_discard_manyprefix7_unified_trace_phase1_seed${SEED}.json" \
      "$RESULTS/_discard_manyprefix7_unified_trace_phase2_seed${SEED}.json"
echo "=== 7-prefix unified test done ==="
