#!/bin/bash
# Real-code KV-heavy workload, re-run after the victim-search fixes.
#
# Corpus: the OLMoE section 5.3 code text
# (allenai/OLMoE scripts/routing_output/text/github_oss_with_stack_texts.txt,
# sha256 1a8e4b8e...). Real code routes to a narrow expert subset in the
# deeper layers and a broad one early, which is the asymmetry the synthetic
# workloads cannot reproduce.
#
# 8 prompts x ~3072 tokens, cold pass then reverse replay, matching the
# earlier handoff experiment so the numbers are comparable:
#
#   config              cold mean TTFT   replay median   hits   total
#   static C=48              57.9s           0.33s       8/8    465.6s
#   static C=56              50.4s           0.29s       5/8    531.9s
#   static C=64              47.5s          47.5s        1/8    690.4s
#   unified init=56          81.4s           1.19s       8/8    660.1s
#
# Unified lost on total time there, and the suspicion is that much of the
# gap was the O(M * num_blocks) victim search (up to 197 ms per expert
# miss at high KV occupancy, against 22,317 misses) rather than anything
# fundamental. That search is now O(1)-ish, so this re-run tests it.
#
# Trace is OFF throughout: it inflates per-step time and would make the
# comparison apples-to-oranges (see GPU_VALIDATION.md).
#
# Prompts come from scripts/prep_code8.py.
set -u
source /root/env.sh 2>/dev/null || true
cd "$(dirname "$0")/.."

MODEL=${MODEL:-allenai/OLMoE-1B-7B-0924-Instruct}
PORT=${PORT:-8100}
R=${R:-/root/code8_results}
L=${L:-/root/code8_logs}
FWD=${FWD:-/root/code8_fwd.jsonl}
REV=${REV:-/root/code8_rev.jsonl}
mkdir -p "$R" "$L"

log(){ echo "[$(date +%H:%M:%S)] $*"; }
SERVE="python3 -m vllm.entrypoints.openai.api_server"
BENCH="python3 -m vllm.entrypoints.cli.main bench serve"
ENVF="--enable-prefix-caching --enforce-eager --trust-remote-code --max-model-len 4096 --max-num-batched-tokens 1 --no-async-scheduling --attention-backend TRITON_ATTN --block-size 16 --gpu-memory-utilization 0.90"

BOOTPID=""
gpu_clean(){ local i u n p
  for i in $(seq 1 90); do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
    [ "$n" -eq 0 ] && [ "$u" -lt 800 ] && return 0
    for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
      kill -9 "$p" 2>/dev/null
    done
    sleep 2
  done
  log "  WARN gpu not clean"; return 1; }

boot(){ local extra="$1" blog="$2" i
  env VLLM_UNIFIED_POOL_TRACE=0 setsid $SERVE --model "$MODEL" --port "$PORT" \
      $extra $ENVF > "$blog" 2>&1 &
  BOOTPID=$!
  for i in $(seq 1 300); do
    curl -sf -m2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && {
      log "  boot ok (${i}x2s)"; return 0; }
    kill -0 "$BOOTPID" 2>/dev/null || { log "  BOOT DEAD -> $blog"; return 1; }
    sleep 2
  done
  log "  BOOT TIMEOUT -> $blog"; return 1; }

down(){ [ -n "$BOOTPID" ] && {
    kill -TERM -"$BOOTPID" 2>/dev/null || kill -TERM "$BOOTPID" 2>/dev/null
    sleep 8
    kill -KILL -"$BOOTPID" 2>/dev/null || kill -KILL "$BOOTPID" 2>/dev/null
  }
  BOOTPID=""; gpu_clean; }

pass(){ local tag=$1 file=$2 which=$3
  timeout 3600 $BENCH --backend vllm --host 127.0.0.1 --port "$PORT" \
    --endpoint /v1/completions --model "$MODEL" --dataset-name custom \
    --dataset-path "$file" --disable-shuffle --skip-chat-template \
    --custom-output-len 1 --num-prompts 8 --max-concurrency 1 --num-warmups 0 \
    --seed 1 --save-detailed --result-filename "$R/${tag}_${which}.json" \
    --save-result --trust-remote-code > "$L/${tag}_${which}.log" 2>&1
  log "  $tag $which exit $?"; }

cell(){ local tag=$1 flags=$2
  if [ -f "$R/${tag}_replay.json" ]; then log "SKIP $tag (done)"; return; fi
  log "CELL $tag"
  gpu_clean
  if boot "$flags" "$L/${tag}_boot.log"; then
    pass "$tag" "$FWD" cold
    pass "$tag" "$REV" replay
  else
    log "  SKIP $tag (boot fail)"
  fi
  down; }

M67="--num-gpu-blocks-override 6432 --expert-pool-page-tokens 16"

log "===== CODE8 START ====="
# Unified with the adaptive target OFF. The handoff's best unified numbers
# came from this configuration, and its own conclusion was that the
# adaptive target "should not be treated as the solution".
cell uni48_noadapt "--expert-offload --expert-unified-pool $M67 --expert-cache-size 48 --expert-working-set-window 0"
# Static C=48: the tuned oracle for this stationary workload.
cell static48 "--expert-offload --expert-cache-size 48 --num-gpu-blocks-override 1824"
# Unified with the adaptive target ON (the current default), for contrast.
cell uni48_adapt "--expert-offload --expert-unified-pool $M67 --expert-cache-size 48"
# Same unified configuration as uni48_noadapt, re-run with the KV score
# restored to the warmest page (commit 25a42a4's anti-starvation bias).
# uni48_noadapt scored by the oldest page and retained only 2/8 prefixes.
cell uni48_kvprotect "--expert-offload --expert-unified-pool $M67 --expert-cache-size 48 --expert-working-set-window 0"
log "===== CODE8 DONE ====="
echo CODE8_DONE
