#!/bin/bash
set -u
ROOT=/home/belal/150326
LOGS=$ROOT/logs
RESULTS=$ROOT/results
VLLM=$ROOT/venv-phase-2/bin/vllm
MODEL=allenai/OLMoE-1B-7B-0924-Instruct
PROMPTS=$ROOT/prompts/alternating_prompts.jsonl
SEED=1

CUDA_VISIBLE_DEVICES=1 setsid "$VLLM" serve "$MODEL" \
    --port 8001 \
    --expert-offload --expert-unified-pool --expert-cache-size 64 \
    --enable-prefix-caching --enforce-eager --trust-remote-code \
    --max-model-len 4096 --max-num-batched-tokens 1 \
    --no-async-scheduling --attention-backend TRITON_ATTN \
    --block-size 1536 --num-gpu-blocks-override 68 \
    > "$LOGS/test1A_unified_from_bad_seed${SEED}_g1.log" 2>&1 &
PID=$!

# Wait for server
for i in $(seq 1 240); do
  if curl -sf -m 2 http://127.0.0.1:8001/v1/models >/dev/null 2>&1; then
    echo "[ready] port=8001 after ${i}*2s"; break
  fi
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "[dead] server pid=$PID exited during boot"; exit 1
  fi
  sleep 2
done

"$VLLM" bench serve --backend vllm --host 127.0.0.1 --port 8001 \
    --endpoint /v1/completions --model "$MODEL" \
    --dataset-name custom --dataset-path "$PROMPTS" \
    --disable-shuffle --skip-chat-template \
    --custom-output-len 20 --num-prompts 20 \
    --max-concurrency 1 --num-warmups 1 --seed $SEED \
    --result-filename "$RESULTS/test1A_unified_from_bad_seed${SEED}.json" \
    --save-result --trust-remote-code \
    > "$LOGS/bench_test1A_unified_latency_seed${SEED}.log" 2>&1
B_EX=$?
echo "[unified_g1] bench exit: $B_EX"

kill -TERM -$PID 2>/dev/null || kill -TERM $PID 2>/dev/null
for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
  if ! kill -0 $PID 2>/dev/null; then break; fi
  sleep 1
done
kill -KILL -$PID 2>/dev/null || kill -KILL $PID 2>/dev/null || true
echo "=== unified_g1 done ==="
