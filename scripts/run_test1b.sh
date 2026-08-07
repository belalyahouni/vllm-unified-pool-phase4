#!/bin/bash
set -u
SEED=1
ROOT=/home/belal/150326
LOGS=$ROOT/logs
RESULTS=$ROOT/results
VLLM=$ROOT/venv-phase-2/bin/vllm
MODEL=allenai/OLMoE-1B-7B-0924-Instruct
UNIFIED_BLOCKS="${UNIFIED_BLOCKS:-68}"
TRACE_LEVEL="${TRACE_LEVEL:-1}"

mkdir -p "$LOGS" "$RESULTS"

start_static() {
  local gpu=$1 port=$2 cache=$3 cell=$4
  CUDA_VISIBLE_DEVICES=$gpu setsid "$VLLM" serve "$MODEL" \
    --port $port \
    --expert-offload --expert-cache-size $cache \
    --enable-prefix-caching --enforce-eager --trust-remote-code \
    --max-model-len 4096 --max-num-batched-tokens 1 \
    --no-async-scheduling --attention-backend TRITON_ATTN \
    --block-size 1536 --gpu-memory-utilization 0.3105 \
    > "$LOGS/${cell}_seed${SEED}_g${gpu}.log" 2>&1 &
  echo $!
}

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
      echo "[dead] server pid=$pid (port=$port) exited during boot"; return 1
    fi
    sleep 2
  done
  echo "[timeout] port=$port"; return 1
}

run_bench_1B() {
  local port=$1 outname=$2
  "$VLLM" bench serve --backend vllm --host 127.0.0.1 --port $port \
    --endpoint /v1/completions --model "$MODEL" \
    --dataset-name random \
    --random-input-len 256 --random-output-len 80 \
    --random-range-ratio 0 --random-prefix-len 0 \
    --num-prompts 8 \
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

ROUND="${1:-static}"

if [ "$ROUND" = "static" ] || [ "$ROUND" = "all" ]; then
  echo "=== 1B Round static: 1B-static-bad (GPU0, cache=16) + 1B-static-good (GPU1, cache=64) ==="
  PID0=$(start_static 0 8000 16 test1B_static_bad)
  PID1=$(start_static 1 8001 64 test1B_static_good)
  wait_for_server 8000 $PID0 || { shutdown $PID0; shutdown $PID1; exit 1; }
  wait_for_server 8001 $PID1 || { shutdown $PID0; shutdown $PID1; exit 1; }
  run_bench_1B 8000 test1B_static_bad_seed${SEED}.json > "$LOGS/bench_test1B_static_bad_seed${SEED}.log" 2>&1 &
  B0=$!
  run_bench_1B 8001 test1B_static_good_seed${SEED}.json > "$LOGS/bench_test1B_static_good_seed${SEED}.log" 2>&1 &
  B1=$!
  wait $B0; B0_EX=$?
  wait $B1; B1_EX=$?
  echo "[1B-static] bench exits: g0=$B0_EX g1=$B1_EX"
  shutdown $PID0; shutdown $PID1
  echo "=== 1B Round static done ==="
fi

if [ "$ROUND" = "unified" ] || [ "$ROUND" = "all" ]; then
  echo "=== 1B Round unified: latency (GPU0) + trace (GPU1, lvl=$TRACE_LEVEL) ==="
  PID0=$(start_unified 0 8000 16 test1B_unified_from_bad 0)
  PID1=$(start_unified 1 8001 16 test1B_unified_from_bad 1)
  wait_for_server 8000 $PID0 || { shutdown $PID0; shutdown $PID1; exit 1; }
  wait_for_server 8001 $PID1 || { shutdown $PID0; shutdown $PID1; exit 1; }
  run_bench_1B 8000 test1B_unified_from_bad_seed${SEED}.json > "$LOGS/bench_test1B_unified_latency_seed${SEED}.log" 2>&1 &
  B0=$!
  run_bench_1B 8001 _discard_trace_seed${SEED}.json > "$LOGS/bench_test1B_unified_trace_seed${SEED}.log" 2>&1 &
  B1=$!
  wait $B0; B0_EX=$?
  wait $B1; B1_EX=$?
  echo "[1B-unified] bench exits: latency=$B0_EX trace=$B1_EX"
  shutdown $PID0; shutdown $PID1
  rm -f "$RESULTS/_discard_trace_seed${SEED}.json"
  echo "=== 1B Round unified done ==="
fi

echo "=== Test 1B complete ==="
