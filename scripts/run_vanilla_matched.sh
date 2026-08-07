#!/bin/bash
# Vanilla vLLM at MATCHED memory budget (gpu-mem-util=0.3105, same as static).
# All 64 experts resident in the model's own weights — no offload, no cache.
# Block size 1536 (same as static/unified cells).
#
# Hypothesis: at the matched budget, the residual after expert weights is
# near-zero, so KV is tiny or zero. 1A should be unable to fit a 3072-token
# prefix in cache and lose the prefix-cache hit advantage entirely. May also
# fail to boot if the KV allocator can't get its minimum.
set -u
SEED=1
ROOT=/home/belal/150326
LOGS=$ROOT/logs
RESULTS=$ROOT/results
VLLM=$ROOT/venv-phase-2/bin/vllm
MODEL=allenai/OLMoE-1B-7B-0924-Instruct
ALT_PROMPTS=$ROOT/prompts/alternating_prompts.jsonl
GPU_UTIL="${GPU_UTIL:-0.3105}"

mkdir -p "$LOGS/test1A/server" "$LOGS/test1A/bench"
mkdir -p "$LOGS/test1B/server" "$LOGS/test1B/bench"
mkdir -p "$RESULTS/test1A" "$RESULTS/test1B"

start_vanilla() {
  local gpu=$1 port=$2 cell=$3 test=$4
  CUDA_VISIBLE_DEVICES=$gpu setsid "$VLLM" serve "$MODEL" \
    --port $port \
    --enable-prefix-caching --enforce-eager --trust-remote-code \
    --max-model-len 4096 --max-num-batched-tokens 1 \
    --no-async-scheduling --attention-backend TRITON_ATTN \
    --block-size 1536 --gpu-memory-utilization $GPU_UTIL \
    > "$LOGS/${test}/server/${cell}_seed${SEED}_g${gpu}.log" 2>&1 &
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

capture_idle_mem() {
  local gpu=$1 cell=$2 test=$3
  sleep 3
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i $gpu \
    > "$LOGS/${test}/server/${cell}_seed${SEED}_idle_mem.txt"
}

run_bench_1A() {
  local port=$1 outname=$2
  "$VLLM" bench serve --backend vllm --host 127.0.0.1 --port $port \
    --endpoint /v1/completions --model "$MODEL" \
    --dataset-name custom --dataset-path "$ALT_PROMPTS" \
    --disable-shuffle --skip-chat-template \
    --custom-output-len 20 --num-prompts 20 \
    --max-concurrency 1 --num-warmups 1 --seed $SEED \
    --result-filename "$RESULTS/test1A/$outname" --save-result --trust-remote-code
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
    --result-filename "$RESULTS/test1B/$outname" --save-result --trust-remote-code
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

echo "=== Vanilla MATCHED-BUDGET: 1A (GPU0) + 1B (GPU1), gpu-mem-util=$GPU_UTIL ==="
PID0=$(start_vanilla 0 8000 test1A_vanilla_matched test1A)
PID1=$(start_vanilla 1 8001 test1B_vanilla_matched test1B)
W0=0; W1=0
wait_for_server 8000 $PID0; W0=$?
wait_for_server 8001 $PID1; W1=$?
if [ $W0 -ne 0 ] || [ $W1 -ne 0 ]; then
  echo "[boot] failure(s): g0=$W0 g1=$W1 — see server logs for the OOM / KV-allocator error."
  shutdown $PID0; shutdown $PID1
  exit 1
fi
capture_idle_mem 0 test1A_vanilla_matched test1A
capture_idle_mem 1 test1B_vanilla_matched test1B

run_bench_1A 8000 test1A_vanilla_matched_seed${SEED}.json \
  > "$LOGS/test1A/bench/bench_test1A_vanilla_matched_seed${SEED}.log" 2>&1 &
B0=$!
run_bench_1B 8001 test1B_vanilla_matched_seed${SEED}.json \
  > "$LOGS/test1B/bench/bench_test1B_vanilla_matched_seed${SEED}.log" 2>&1 &
B1=$!
wait $B0; B0_EX=$?
wait $B1; B1_EX=$?
echo "[vanilla-matched] bench exits: 1A=$B0_EX 1B=$B1_EX"

shutdown $PID0; shutdown $PID1
echo "=== Vanilla MATCHED-BUDGET done ==="
