#!/bin/bash
# Trace pass for unified pool=64 cell of the many-prefixes test.
# Same workload + seed as the latency pass; emits VLLM_UNIFIED_POOL_TRACE=1 lines
# for per-step pool-composition analysis.
set -u
SEED=1
ROOT=/home/belal/150326
LOGS=$ROOT/logs
RESULTS=$ROOT/results
VLLM=$ROOT/venv-phase-2/bin/vllm
MODEL=allenai/OLMoE-1B-7B-0924-Instruct
PROMPTS=$ROOT/prompts/many_prefixes.jsonl
GPU="${GPU:-0}"
PORT="${PORT:-8000}"
TRACE_LEVEL="${TRACE_LEVEL:-1}"

mkdir -p "$LOGS" "$RESULTS"

VLLM_UNIFIED_POOL_TRACE=$TRACE_LEVEL CUDA_VISIBLE_DEVICES=$GPU setsid "$VLLM" serve "$MODEL" \
  --port $PORT \
  --expert-offload --expert-unified-pool --expert-cache-size 40 \
  --enable-prefix-caching --enforce-eager --trust-remote-code \
  --max-model-len 4096 --max-num-batched-tokens 1 \
  --no-async-scheduling --attention-backend TRITON_ATTN \
  --block-size 1536 --num-gpu-blocks-override 64 \
  > "$LOGS/manyprefix_unified_init40_seed${SEED}_g${GPU}_trace.log" 2>&1 &
PID=$!

for i in $(seq 1 240); do
  if curl -sf -m 2 http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1; then
    echo "[ready] port=$PORT after ${i}*2s"; break
  fi
  if ! kill -0 "$PID" 2>/dev/null; then echo "[dead]"; exit 1; fi
  sleep 2
done

run_phase() {
  local outname=$1
  "$VLLM" bench serve --backend vllm --host 127.0.0.1 --port $PORT \
    --endpoint /v1/completions --model "$MODEL" \
    --dataset-name custom --dataset-path "$PROMPTS" \
    --disable-shuffle --skip-chat-template \
    --custom-output-len 5 --num-prompts 10 \
    --max-concurrency 1 --num-warmups 0 --seed $SEED \
    --result-filename "$RESULTS/$outname" --save-result --trust-remote-code
}

run_phase "_discard_manyprefix_unified_trace_phase1_seed${SEED}.json" \
  > "$LOGS/bench_manyprefix_unified_trace_phase1_seed${SEED}_g${GPU}.log" 2>&1
echo "[unified-trace g${GPU}] phase1 (cold) exit: $?"
run_phase "_discard_manyprefix_unified_trace_phase2_seed${SEED}.json" \
  > "$LOGS/bench_manyprefix_unified_trace_phase2_seed${SEED}_g${GPU}.log" 2>&1
echo "[unified-trace g${GPU}] phase2 (warm) exit: $?"

kill -TERM -$PID 2>/dev/null || kill -TERM $PID 2>/dev/null || true
for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
  if ! kill -0 $PID 2>/dev/null; then break; fi; sleep 1
done
kill -KILL -$PID 2>/dev/null || kill -KILL $PID 2>/dev/null || true
rm -f "$RESULTS/_discard_manyprefix_unified_trace_phase1_seed${SEED}.json" \
      "$RESULTS/_discard_manyprefix_unified_trace_phase2_seed${SEED}.json"
echo "=== unified trace done ==="
