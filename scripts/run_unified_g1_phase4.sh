#!/bin/bash
# Phase 4 (fine-grained pages) smoke / equivalence run, adapted from
# run_unified_g1.sh. Serves OLMoE with the unified pool at a *tunable*
# page size and runs the 20-prompt latency bench.
#
# Phase 3 pinned one page == one whole expert (1536 tokens). Phase 4 sets
# the page with --expert-pool-page-tokens; an expert then spans a super-
# block of F = EXPERT_TOKENS / PAGE_TOKENS contiguous pages. We scale
# --num-gpu-blocks-override by F so the total KV byte budget matches
# Phase 3 exactly, which makes token outputs directly comparable.
#
# Bring-up ladder (set PAGE_TOKENS):
#   1536 -> F=1   : must be byte-identical to Phase 3 (the M1 gate).
#    384 -> F=4   : matches the report figure; easy to eyeball in traces.
#     16 -> F=96  : the target (kernel-minimum page, max fragmentation win).
#
# Verify correctness on the GPU box (single NVIDIA L40, as in the report):
#   1. Server boots and prints "UnifiedPool warm-up sanity check passed"
#      (the F-strided super-block view + super-block DMA round-trip).
#   2. Run with PARANOID=1 -> "Phase-4 paranoid: L0 forward 0 verified ...".
#   3. Output equivalence: run this at PAGE_TOKENS=1536 and again at 16 with
#      the SAME --seed; the completion tokens must match Phase 3's run.
#   4. Fragmentation win: TRACE_LEVEL=1 and compare the "UNIFIED CACHE" line
#      and the relocation count in the shutdown "UnifiedPool: relocation=on
#      ... relocated=N" log against a PAGE_TOKENS=1536 run.
#   5. A/B: set RELOCATE=0 to disable relocation (evict-only) and compare
#      hit-rate / fragmentation against the default (relocation on).

set -u
ROOT="${ROOT:-/home/belal/150326-phase-4}"
LOGS="${LOGS:-$ROOT/logs}"
RESULTS="${RESULTS:-$ROOT/results}"
VLLM="${VLLM:-$ROOT/venv-phase-2/bin/vllm}"
MODEL="${MODEL:-allenai/OLMoE-1B-7B-0924-Instruct}"
PROMPTS="${PROMPTS:-$ROOT/prompts/alternating_prompts.jsonl}"
SEED="${SEED:-1}"

# Tunables.
PAGE_TOKENS="${PAGE_TOKENS:-16}"     # unified-pool page size in tokens
EXPERT_TOKENS="${EXPERT_TOKENS:-1536}"  # OLMoE expert footprint in tokens (L40 ref)
BASE_BLOCKS="${BASE_BLOCKS:-68}"     # Phase-3 num-gpu-blocks at F=1
PARANOID="${PARANOID:-0}"
TRACE_LEVEL="${TRACE_LEVEL:-0}"
RELOCATE="${RELOCATE:-1}"

if (( EXPERT_TOKENS % PAGE_TOKENS != 0 )); then
  echo "ERROR: EXPERT_TOKENS ($EXPERT_TOKENS) must be divisible by PAGE_TOKENS ($PAGE_TOKENS)"
  exit 1
fi
F=$(( EXPERT_TOKENS / PAGE_TOKENS ))
GPU_BLOCKS=$(( BASE_BLOCKS * F ))
mkdir -p "$LOGS" "$RESULTS"
echo "[phase4] PAGE_TOKENS=$PAGE_TOKENS F=$F GPU_BLOCKS=$GPU_BLOCKS "\
"PARANOID=$PARANOID TRACE=$TRACE_LEVEL RELOCATE=$RELOCATE"

TAG="p4_pt${PAGE_TOKENS}_seed${SEED}"

VLLM_UNIFIED_POOL_TRACE=$TRACE_LEVEL \
VLLM_UNIFIED_POOL_PARANOID=$PARANOID \
VLLM_UNIFIED_POOL_RELOCATE=$RELOCATE \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" setsid "$VLLM" serve "$MODEL" \
    --port 8001 \
    --expert-offload --expert-unified-pool --expert-cache-size 64 \
    --expert-pool-page-tokens "$PAGE_TOKENS" \
    --enable-prefix-caching --enforce-eager --trust-remote-code \
    --max-model-len 4096 --max-num-batched-tokens 1 \
    --no-async-scheduling --attention-backend TRITON_ATTN \
    --block-size "$PAGE_TOKENS" --num-gpu-blocks-override "$GPU_BLOCKS" \
    > "$LOGS/serve_${TAG}.log" 2>&1 &
PID=$!

# Wait for server
for i in $(seq 1 240); do
  if curl -sf -m 2 http://127.0.0.1:8001/v1/models >/dev/null 2>&1; then
    echo "[ready] port=8001 after ${i}*2s"; break
  fi
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "[dead] server pid=$PID exited during boot; see $LOGS/serve_${TAG}.log"
    exit 1
  fi
  sleep 2
done

"$VLLM" bench serve --backend vllm --host 127.0.0.1 --port 8001 \
    --endpoint /v1/completions --model "$MODEL" \
    --dataset-name custom --dataset-path "$PROMPTS" \
    --disable-shuffle --skip-chat-template \
    --custom-output-len 20 --num-prompts 20 \
    --max-concurrency 1 --num-warmups 1 --seed "$SEED" \
    --result-filename "$RESULTS/${TAG}.json" \
    --save-result --trust-remote-code \
    > "$LOGS/bench_${TAG}.log" 2>&1
B_EX=$?
echo "[phase4] bench exit: $B_EX"

kill -TERM -$PID 2>/dev/null || kill -TERM $PID 2>/dev/null
for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
  if ! kill -0 $PID 2>/dev/null; then break; fi
  sleep 1
done
kill -KILL -$PID 2>/dev/null || kill -KILL $PID 2>/dev/null || true
echo "=== phase4 $TAG done ==="
