#!/bin/bash
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

# 2B Phase 1: random, 6 prompts, --num-warmups 1
run_phase1() {
  local port=$1 outname=$2
  "$VLLM" bench serve --backend vllm --host 127.0.0.1 --port $port \
    --endpoint /v1/completions --model "$MODEL" \
    --dataset-name random \
    --random-input-len 256 --random-output-len 80 \
    --random-range-ratio 0 --random-prefix-len 0 \
    --num-prompts 6 \
    --max-concurrency 1 --num-warmups 1 --seed $SEED \
    --result-filename "$RESULTS/$outname" --save-result --trust-remote-code
}

# 2B Phase 2: alternating prefixes, 10 prompts, --num-warmups 0 (preserve pool state)
run_phase2() {
  local port=$1 outname=$2
  "$VLLM" bench serve --backend vllm --host 127.0.0.1 --port $port \
    --endpoint /v1/completions --model "$MODEL" \
    --dataset-name custom --dataset-path "$PROMPTS" \
    --disable-shuffle --skip-chat-template \
    --custom-output-len 20 --num-prompts 10 \
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
  local cell=$1 gpu=$2 port=$3 cache=$4 mode=$5 trace=${6:-0}
  local PID
  if [ "$mode" = "static" ]; then
    PID=$(start_static $gpu $port $cache $cell)
  else
    PID=$(start_unified $gpu $port $cache $cell $trace)
  fi
  wait_for_server $port $PID || { shutdown $PID; return 1; }

  local p1_out p2_out
  if [ "$trace" = "1" ]; then
    p1_out="_discard_${cell}_phase1_trace_seed${SEED}.json"
    p2_out="_discard_${cell}_phase2_trace_seed${SEED}.json"
  else
    p1_out="${cell}_phase1_seed${SEED}.json"
    p2_out="${cell}_phase2_seed${SEED}.json"
  fi
  run_phase1 $port "$p1_out" > "$LOGS/bench_${cell}_phase1_seed${SEED}_g${gpu}.log" 2>&1
  echo "[${cell} g${gpu}] phase1 exit: $?"
  run_phase2 $port "$p2_out" > "$LOGS/bench_${cell}_phase2_seed${SEED}_g${gpu}.log" 2>&1
  echo "[${cell} g${gpu}] phase2 exit: $?"
  shutdown $PID
  if [ "$trace" = "1" ]; then
    rm -f "$RESULTS/$p1_out" "$RESULTS/$p2_out"
  fi
}

ROUND="${1:-all}"

if [ "$ROUND" = "static" ] || [ "$ROUND" = "all" ]; then
  echo "=== 2B static cells ==="
  echo "  Round A: prefix-tuned (cache=20, GPU0) + middle (cache=40, GPU1) ==="
  run_cell test2B_static_prefix_tuned 0 8000 20 static &
  run_cell test2B_static_middle 1 8001 40 static &
  wait
  echo "  Round B: expert-tuned (cache=64, GPU0) ==="
  run_cell test2B_static_expert_tuned 0 8000 64 static
  echo "=== 2B static cells done ==="
fi

if [ "$ROUND" = "unified-middle" ] || [ "$ROUND" = "all" ]; then
  echo "=== 2B unified-from-middle (cache=40 init) — latency (GPU0) + trace (GPU1) ==="
  run_cell test2B_unified_from_middle 0 8000 40 unified 0 &
  run_cell test2B_unified_from_middle 1 8001 40 unified 1 &
  wait
  echo "=== 2B unified-from-middle done ==="
fi

if [ "$ROUND" = "unified-prefix" ] || [ "$ROUND" = "all" ]; then
  echo "=== 2B unified-from-prefix (cache=20 init, ablation) — latency (GPU0) + trace (GPU1) ==="
  run_cell test2B_unified_from_prefix 0 8000 20 unified 0 &
  run_cell test2B_unified_from_prefix 1 8001 20 unified 1 &
  wait
  echo "=== 2B unified-from-prefix done ==="
fi

if [ "$ROUND" = "unified-expert" ] || [ "$ROUND" = "all" ]; then
  echo "=== 2B unified-from-expert (cache=64 init, ablation) — latency (GPU0) + trace (GPU1) ==="
  run_cell test2B_unified_from_expert 0 8000 64 unified 0 &
  run_cell test2B_unified_from_expert 1 8001 64 unified 1 &
  wait
  echo "=== 2B unified-from-expert done ==="
fi

echo "=== Test 2B complete ==="
