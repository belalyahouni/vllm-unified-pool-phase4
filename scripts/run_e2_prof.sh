#!/bin/bash
# E2 phase shift with the in-process profiler on, to get policy cost and
# mechanism event counts from a real serving run rather than a synthetic
# harness.
#
# Same configuration as E2's ours48 cell (M=67, init 48) and the same
# Nexp=220 as the reported badness runs, so counts here correspond to the
# runs the headline number comes from. The earlier traced runs used
# Nexp_tr=60, so their relocation counts cannot be divided by a 220-request
# run's duration.
#
# Profiling is far lighter than VLLM_UNIFIED_POOL_TRACE (CUDA events and
# perf_counter, no per-step printing), but it is not free: treat latency
# from this run as indicative only and take end-to-end numbers from the
# existing trace-off runs.
set -u
source /root/env.sh 2>/dev/null || true
cd "$(dirname "$0")/.."

MODEL=${MODEL:-allenai/OLMoE-1B-7B-0924-Instruct}
PORT=${PORT:-8100}
R=${R:-/root/e2prof_results}
L=${L:-/root/e2prof_logs}
FWD=${FWD:-/root/kvd_fwd.jsonl}
REV=${REV:-/root/kvd_rev.jsonl}
NEXP=${NEXP:-220}
mkdir -p "$R" "$L"

log(){ echo "[$(date +%H:%M:%S)] $*"; }
SERVE="python3 -m vllm.entrypoints.openai.api_server"
BENCH="python3 -m vllm.entrypoints.cli.main bench serve"
BT="timeout 6000"
ENVF="--enable-prefix-caching --enforce-eager --trust-remote-code --max-model-len 4096 --max-num-batched-tokens 1 --no-async-scheduling --attention-backend TRITON_ATTN --block-size 16 --gpu-memory-utilization 0.90"
OURS48="--expert-offload --expert-unified-pool --expert-pool-page-tokens 16 --expert-cache-size 48 --num-gpu-blocks-override 6432"

BOOTPID=""
gpu_clean(){ local i u n p
  for i in $(seq 1 120); do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
    [ "$n" -eq 0 ] && [ "$u" -lt 800 ] && return 0
    for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
      kill -9 "$p" 2>/dev/null
    done
    sleep 2
  done
  log "  WARN gpu not clean"; return 1; }

boot(){ local blog="$1" pjson="$2" i
  env VLLM_UNIFIED_POOL_TRACE=0 VLLM_UNIFIED_POOL_PROFILE=1 \
      VLLM_UNIFIED_POOL_PROF_JSON="$pjson" VLLM_UNIFIED_POOL_PROF_EVERY=2000 \
      setsid $SERVE --model "$MODEL" --port "$PORT" $OURS48 $ENVF > "$blog" 2>&1 &
  BOOTPID=$!
  for i in $(seq 1 300); do
    curl -sf -m2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && {
      log "  boot ok"; return 0; }
    kill -0 "$BOOTPID" 2>/dev/null || { log "  BOOT DEAD -> $blog"; return 1; }
    sleep 2
  done
  log "  BOOT TIMEOUT"; return 1; }

down(){ [ -n "$BOOTPID" ] && {
    kill -TERM -"$BOOTPID" 2>/dev/null || kill -TERM "$BOOTPID" 2>/dev/null
    sleep 10
    kill -KILL -"$BOOTPID" 2>/dev/null || kill -KILL "$BOOTPID" 2>/dev/null
  }
  BOOTPID=""; sleep 1; gpu_clean; }

kv(){ local ds=$1 out=$2
  $BT $BENCH --backend vllm --host 127.0.0.1 --port "$PORT" \
    --endpoint /v1/completions --model "$MODEL" --dataset-name custom \
    --dataset-path "$ds" --disable-shuffle --skip-chat-template \
    --custom-output-len 1 --num-prompts 16 --max-concurrency 1 --num-warmups 0 \
    --seed 1 --save-detailed --result-filename "$R/$out" --save-result \
    --trust-remote-code > "$L/${out%.json}.log" 2>&1
  log "    $out exit $?"; }

exp(){ local out=$1
  $BT $BENCH --backend vllm --host 127.0.0.1 --port "$PORT" \
    --endpoint /v1/completions --model "$MODEL" --dataset-name random \
    --random-input-len 256 --random-output-len 80 --random-range-ratio 0 \
    --random-prefix-len 0 --num-prompts "$NEXP" --max-concurrency 1 \
    --num-warmups 0 --seed 1 --save-detailed --result-filename "$R/$out" \
    --save-result --trust-remote-code > "$L/${out%.json}.log" 2>&1
  log "    $out exit $?"; }

log "===== E2 PROF START (Nexp=$NEXP) ====="
gpu_clean

if [ ! -f "$R/kv2exp_prof.json" ]; then
  log "RUN kv2exp"
  if boot "$L/kv2exp_boot.log" "$R/kv2exp_prof.json"; then
    kv "$FWD" kv2exp_kvcold.json
    kv "$REV" kv2exp_kvwarm.json
    exp kv2exp_exp.json
  fi
  down
else
  log "SKIP kv2exp (done)"
fi

if [ ! -f "$R/exp2kv_prof.json" ]; then
  log "RUN exp2kv"
  if boot "$L/exp2kv_boot.log" "$R/exp2kv_prof.json"; then
    exp exp2kv_exp.json
    kv "$FWD" exp2kv_kvcold.json
    kv "$REV" exp2kv_kvwarm.json
  fi
  down
else
  log "SKIP exp2kv (done)"
fi

log "===== E2 PROF DONE ====="
echo E2PROF_DONE
