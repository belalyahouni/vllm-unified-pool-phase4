#!/bin/bash
# Experiment 2 — phase-shift adaptation (M=67). Same E1 workloads back-to-back on a
# persistent server (pool state carries; --num-warmups 0 throughout). Two orders:
# KV->Expert and Expert->KV. Expert phase = 220 requests (tuned so static32 ~ static64
# in total badness; see E2 section of EXPERIMENTS.md). Waits for the E1 notrace rerun.
set -u
export HF_HOME=/workspace/hf
MODEL=allenai/OLMoE-1B-7B-0924-Instruct
PORT=8100
R=/workspace/e2_results
L=/workspace/e2_logs
mkdir -p "$R" "$L"
PROG=/workspace/e2_progress.log; : > "$PROG"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$PROG"; }
SERVE="python3 -m vllm.entrypoints.openai.api_server"
BENCH="python3 -m vllm.entrypoints.cli.main bench serve"
BT="timeout 6000"                 # 100 min/bench (trace-on expert phase can be slow)
ENVF="--enable-prefix-caching --enforce-eager --trust-remote-code --max-model-len 4096 --max-num-batched-tokens 1 --no-async-scheduling --attention-backend TRITON_ATTN --block-size 16 --gpu-memory-utilization 0.90"
FWD=/workspace/kv_distinct_fwd.jsonl
REV=/workspace/kv_distinct_rev.jsonl
NEXP=220                          # badness (no-trace) runs
NEXP_TR=60                        # trace runs (enough to show the swap + plateau)
BOOTPID=""
wait_gpu_clean(){ local i used procs p; for i in $(seq 1 120); do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|head -1); procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader|wc -l); [ "$procs" -eq 0 ] && [ "$used" -lt 800 ] && return 0; for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 "$p" 2>/dev/null; done; sleep 2; done; }
boot(){ local extra="$1" blog="$2" trace="${3:-0}" envp="" i; [ "$trace" = "1" ] && envp="VLLM_UNIFIED_POOL_TRACE=1"; env $envp HF_HOME=/workspace/hf setsid $SERVE --model "$MODEL" --port $PORT $extra $ENVF > "$blog" 2>&1 & BOOTPID=$!; for i in $(seq 1 200); do curl -sf -m2 http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && { log "  boot ready"; return 0; }; kill -0 "$BOOTPID" 2>/dev/null || { log "  BOOT DEAD -> $blog"; return 1; }; sleep 2; done; return 1; }
shutdown(){ [ -n "$BOOTPID" ] || { wait_gpu_clean; return; }; kill -TERM -"$BOOTPID" 2>/dev/null; local i; for i in $(seq 1 15); do kill -0 "$BOOTPID" 2>/dev/null || break; sleep 1; done; kill -KILL -"$BOOTPID" 2>/dev/null; BOOTPID=""; sleep 1; wait_gpu_clean; }
bench_kv(){ local ds=$1 out=$2   # KV phase sub-pass (16 prompts, out=1); pool state carries (num-warmups 0)
  $BT $BENCH --backend vllm --host 127.0.0.1 --port $PORT --endpoint /v1/completions --model "$MODEL" --dataset-name custom --dataset-path "$ds" --disable-shuffle --skip-chat-template --custom-output-len 1 --num-prompts 16 --max-concurrency 1 --num-warmups 0 --seed 1 --save-detailed --result-filename "$R/$out" --save-result --trust-remote-code > "$L/${out%.json}.log" 2>&1; log "    $out exit $?"; }
bench_exp(){ local out=$1 n=$2    # expert phase (n random requests, out=80); num-warmups 0
  $BT $BENCH --backend vllm --host 127.0.0.1 --port $PORT --endpoint /v1/completions --model "$MODEL" --dataset-name random --random-input-len 256 --random-output-len 80 --random-range-ratio 0 --random-prefix-len 0 --num-prompts $n --max-concurrency 1 --num-warmups 0 --seed 1 --save-detailed --result-filename "$R/$out" --save-result --trust-remote-code > "$L/${out%.json}.log" 2>&1; log "    $out exit $?"; }
run_kv2exp(){ local tag=$1 flags=$2 trace=$3 n=$4; log "RUN $tag / KV->Expert (trace=$trace, Nexp=$n)"; if boot "$flags" "$L/${tag}_kv2exp_boot.log" "$trace"; then bench_kv "$FWD" "${tag}_kv2exp_kvcold.json"; bench_kv "$REV" "${tag}_kv2exp_kvwarm.json"; bench_exp "${tag}_kv2exp_exp.json" "$n"; else log "  SKIP"; fi; shutdown; }
run_exp2kv(){ local tag=$1 flags=$2 trace=$3 n=$4; log "RUN $tag / Expert->KV (trace=$trace, Nexp=$n)"; if boot "$flags" "$L/${tag}_exp2kv_boot.log" "$trace"; then bench_exp "${tag}_exp2kv_exp.json" "$n"; bench_kv "$FWD" "${tag}_exp2kv_kvcold.json"; bench_kv "$REV" "${tag}_exp2kv_kvwarm.json"; else log "  SKIP"; fi; shutdown; }

STATIC32="--expert-offload --expert-cache-size 32 --num-gpu-blocks-override 3360"
STATIC48="--expert-offload --expert-cache-size 48 --num-gpu-blocks-override 1824"
STATIC64="--expert-offload --expert-cache-size 64 --num-gpu-blocks-override 288"
OURS48="--expert-offload --expert-unified-pool --expert-pool-page-tokens 16 --expert-cache-size 48 --num-gpu-blocks-override 6432"

log "===== E2 QUEUED — waiting for E1 notrace rerun to finish ====="
while pgrep -f run_ours_notrace >/dev/null; do sleep 30; done
log "===== E2 START (M=67; Nexp=$NEXP) ====="; wait_gpu_clean

# Badness runs (trace OFF, Nexp=220), both orders
run_kv2exp static32 "$STATIC32" 0 $NEXP; run_exp2kv static32 "$STATIC32" 0 $NEXP
run_kv2exp static48 "$STATIC48" 0 $NEXP; run_exp2kv static48 "$STATIC48" 0 $NEXP
run_kv2exp static64 "$STATIC64" 0 $NEXP; run_exp2kv static64 "$STATIC64" 0 $NEXP
run_kv2exp ours48nt "$OURS48"   0 $NEXP; run_exp2kv ours48nt "$OURS48"   0 $NEXP
# Trace runs (trace ON, shorter Nexp for the composition/VRAM figure), both orders
run_kv2exp ours48tr "$OURS48"   1 $NEXP_TR; run_exp2kv ours48tr "$OURS48"   1 $NEXP_TR
log "===== E2 DONE ====="
