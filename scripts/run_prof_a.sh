#!/bin/bash
# Workstream A — mechanism-cost microbenchmark.
#
# Measures what the unified pool's shared address space *costs*, split into
# GPU byte movement (expert HtoD, relocation D2D) and the Python
# victim-selection logic, via VLLM_UNIFIED_POOL_PROFILE=1.
#
# Two pool regimes, because they answer different questions:
#   P (paper, M=67 super-blocks): the regime E1/E2 actually reported. Here
#     GPU_VALIDATION.md predicts the pool self-partitions and relocation
#     never fires — so this cell measures the cost of the mechanism in the
#     configuration the paper's numbers came from.
#   T (tight, M=18): the crafted regime that does force cross-type
#     eviction + relocation, so the per-relocation cost is observable.
#
# Crossed with relocation on/off (VLLM_UNIFIED_POOL_RELOCATE), which is
# the evict-only ablation.
#
# NOTE: profiling adds CPU timing and CUDA events, so latencies from these
# runs are NOT comparable with the trace-off latency runs in results/.
# This script measures the cost *breakdown*; E1/E2 measure end-to-end time.
# Tracing is kept OFF throughout (it inflates per-step time badly).
set -u

export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH:-}
export HF_HOME=${HF_HOME:-/root/hf}
MODEL=allenai/OLMoE-1B-7B-0924-Instruct
PORT=8100
R=${R:-/root/prof_a_results}
L=${L:-/root/prof_a_logs}
mkdir -p "$R" "$L"
PROG=$R/progress.log
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$PROG"; }

SERVE="python3 -m vllm.entrypoints.openai.api_server"
BENCH="python3 -m vllm.entrypoints.cli.main bench serve"
BT="timeout 2400"
ENVF="--enable-prefix-caching --enforce-eager --trust-remote-code --max-model-len 4096 --max-num-batched-tokens 1 --no-async-scheduling --attention-backend TRITON_ATTN --block-size 16 --gpu-memory-utilization 0.90"

# Pool configs. F = 96 pages/super-block at 16 tokens/page, so
# num-gpu-blocks-override = 96 * num_super_blocks.
PAPER="--expert-offload --expert-unified-pool --expert-pool-page-tokens 16 --expert-cache-size 8 --num-gpu-blocks-override 6432"   # M=67
TIGHT="--expert-offload --expert-unified-pool --expert-pool-page-tokens 16 --expert-cache-size 16 --num-gpu-blocks-override 1728"  # M=18

BOOTPID=""
wait_gpu_clean(){ local i used procs p; for i in $(seq 1 90); do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|head -1); procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader|wc -l); [ "$procs" -eq 0 ] && [ "$used" -lt 800 ] && return 0; for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 "$p" 2>/dev/null; done; sleep 2; done; log "  WARN gpu not clean"; return 1; }

# $1 flags, $2 bootlog, $3 relocate(0/1), $4 profile json path
boot(){ local extra="$1" blog="$2" reloc="$3" pjson="$4" i
  env VLLM_UNIFIED_POOL_PROFILE=1 VLLM_UNIFIED_POOL_PROF_JSON="$pjson" \
      VLLM_UNIFIED_POOL_PROF_EVERY=500 VLLM_UNIFIED_POOL_RELOCATE="$reloc" \
      VLLM_UNIFIED_POOL_TRACE=0 HF_HOME="$HF_HOME" \
      setsid $SERVE --model "$MODEL" --port $PORT $extra $ENVF > "$blog" 2>&1 &
  BOOTPID=$!
  for i in $(seq 1 300); do
    curl -sf -m2 http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && { log "  boot ready (${i}x2s)"; return 0; }
    kill -0 "$BOOTPID" 2>/dev/null || { log "  BOOT DEAD -> $blog"; return 1; }
    sleep 2
  done
  log "  BOOT TIMEOUT -> $blog"; return 1; }

shutdown(){ [ -n "$BOOTPID" ] || { wait_gpu_clean; return; }
  kill -TERM -"$BOOTPID" 2>/dev/null || kill -TERM "$BOOTPID" 2>/dev/null || true
  local i; for i in $(seq 1 20); do kill -0 "$BOOTPID" 2>/dev/null || break; sleep 1; done
  kill -KILL -"$BOOTPID" 2>/dev/null || kill -KILL "$BOOTPID" 2>/dev/null || true
  BOOTPID=""; sleep 1; wait_gpu_clean; }

# Expert-heavy: self-contained `random` dataset, drives many expert misses.
bench_exp(){ local tag=$1
  $BT $BENCH --backend vllm --host 127.0.0.1 --port $PORT --endpoint /v1/completions \
    --model "$MODEL" --dataset-name random --random-input-len 256 --random-output-len 80 \
    --random-range-ratio 0 --random-prefix-len 0 --num-prompts 12 --max-concurrency 1 \
    --num-warmups 1 --seed 1 --save-detailed --result-filename "$R/${tag}_exp.json" \
    --save-result --trust-remote-code > "$L/bench_${tag}_exp.log" 2>&1
  log "  $tag exp exit $?"; }

# KV-heavy: long distinct prompts, the workload that forces KV<->expert flex.
bench_kv(){ local tag=$1
  [ -f "$FWD" ] || { log "  SKIP $tag kv (no $FWD)"; return; }
  $BT $BENCH --backend vllm --host 127.0.0.1 --port $PORT --endpoint /v1/completions \
    --model "$MODEL" --dataset-name custom --dataset-path "$FWD" --disable-shuffle \
    --skip-chat-template --custom-output-len 1 --num-prompts 8 --max-concurrency 1 \
    --num-warmups 0 --seed 1 --save-detailed --result-filename "$R/${tag}_kv.json" \
    --save-result --trust-remote-code > "$L/bench_${tag}_kv.log" 2>&1
  log "  $tag kv exit $?"; }

FWD=${FWD:-/root/kv_distinct_fwd.jsonl}

# $1 tag, $2 flags, $3 reloc, $4 workload(exp|kv)
cell(){ local tag=$1 flags=$2 reloc=$3 wl=$4
  local pjson="$R/${tag}_prof.json"
  if [ -f "$pjson" ] && [ -f "$R/${tag}_${wl}.json" ]; then log "SKIP $tag (done)"; return; fi
  log "CELL $tag (reloc=$reloc, workload=$wl)"
  if boot "$flags" "$L/${tag}_boot.log" "$reloc" "$pjson"; then
    bench_$wl "$tag"
  else
    log "  SKIP $tag (boot fail)"
  fi
  shutdown
  [ -f "$pjson" ] && log "  profile -> $pjson" || log "  WARN no profile json for $tag"; }

log "===== PROF A START ====="
wait_gpu_clean

# Expert-heavy first: cheapest and needs no prompt prep.
cell paper_exp_reloc1 "$PAPER" 1 exp
cell paper_exp_reloc0 "$PAPER" 0 exp
cell tight_exp_reloc1 "$TIGHT" 1 exp
cell tight_exp_reloc0 "$TIGHT" 0 exp

# KV-heavy: needs $FWD (scripts/make_kv_distinct.py).
cell tight_kv_reloc1  "$TIGHT" 1 kv
cell tight_kv_reloc0  "$TIGHT" 0 kv
cell paper_kv_reloc1  "$PAPER" 1 kv
cell paper_kv_reloc0  "$PAPER" 0 kv

log "===== PROF A DONE ====="
echo "PROF_A_DONE"
