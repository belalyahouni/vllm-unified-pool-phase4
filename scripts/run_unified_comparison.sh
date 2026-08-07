#!/bin/bash
# Re-run unified-from-bad with matched prompt counts so we have apples-to-apples
# comparison data for the static-sweep doc.
#
# Workload A: 5 alternating prompts (matches sweep). cache=64 init.
# Workload B: 8 random prompts (already matches existing 1B). cache=16 init.
#
# Each cell: latency pass on GPU 0, trace pass (level 1) on GPU 1, in parallel.
set -u
SEED=1
ROOT=/home/belal/150326
LOGS=$ROOT/logs
RESULTS=$ROOT/results
VLLM=$ROOT/venv-phase-2/bin/vllm
MODEL=allenai/OLMoE-1B-7B-0924-Instruct
PROMPTS=$ROOT/prompts/alternating_prompts.jsonl
UNIFIED_BLOCKS="${UNIFIED_BLOCKS:-68}"
TRACE_LEVEL="${TRACE_LEVEL:-1}"

mkdir -p "$LOGS" "$RESULTS"

start_unified() {
  local gpu=$1 port=$2 cache=$3 cell=$4 trace=$5
  local logsuffix=""
  local env_prefix=""
  if [ "$trace" = "1" ]; then
    env_prefix="VLLM_UNIFIED_POOL_TRACE=$TRACE_LEVEL"
    logsuffix="_trace"
  fi
  env $env_prefix CUDA_VISIBLE_DEVICES=$gpu setsid "$VLLM" serve "$MODEL" \
    --port $port \
    --expert-offload --expert-unified-pool --expert-cache-size $cache \
    --enable-prefix-caching --enforce-eager --trust-remote-code \
    --max-model-len 4096 --max-num-batched-tokens 1 \
    --no-async-scheduling --attention-backend TRITON_ATTN \
    --block-size 1536 --num-gpu-blocks-override $UNIFIED_BLOCKS \
    > "$LOGS/${cell}_seed${SEED}_g${gpu}${logsuffix}.log" 2>&1 &
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

run_workload_A() {
  local port=$1 outname=$2
  "$VLLM" bench serve --backend vllm --host 127.0.0.1 --port $port \
    --endpoint /v1/completions --model "$MODEL" \
    --dataset-name custom --dataset-path "$PROMPTS" \
    --disable-shuffle --skip-chat-template \
    --custom-output-len 20 --num-prompts 5 \
    --max-concurrency 1 --num-warmups 1 --seed $SEED \
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

echo "=== Unified comparison: workload A (5 prompts, cache=64 init) — latency (GPU0) + trace (GPU1) ==="
PID0=$(start_unified 0 8000 64 sweep_unified_from_bad_1A 0)
PID1=$(start_unified 1 8001 64 sweep_unified_from_bad_1A 1)
wait_for_server 8000 $PID0 || { shutdown $PID0; shutdown $PID1; exit 1; }
wait_for_server 8001 $PID1 || { shutdown $PID0; shutdown $PID1; exit 1; }
run_workload_A 8000 sweep_unified_from_bad_1A_seed${SEED}.json > "$LOGS/bench_sweep_unified_1A_latency_seed${SEED}.log" 2>&1 &
B0=$!
run_workload_A 8001 _discard_sweep_unified_1A_trace_seed${SEED}.json > "$LOGS/bench_sweep_unified_1A_trace_seed${SEED}.log" 2>&1 &
B1=$!
wait $B0; B0_EX=$?
wait $B1; B1_EX=$?
echo "[unified 1A] bench exits: latency=$B0_EX trace=$B1_EX"
shutdown $PID0; shutdown $PID1
rm -f "$RESULTS/_discard_sweep_unified_1A_trace_seed${SEED}.json"

echo "=== Unified comparison done ==="
