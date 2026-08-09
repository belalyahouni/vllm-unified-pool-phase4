#!/bin/bash
# E1 (full, M=67) — adds ours32/ours64 to both workloads. ALL KV-heavy first, then ALL
# expert-heavy. Skips cells whose result already exists (static64_kv, ours48_kv done).
set -u
export HF_HOME=/workspace/hf
MODEL=allenai/OLMoE-1B-7B-0924-Instruct
PORT=8100
R=/workspace/e1full_results
L=/workspace/e1full_logs
mkdir -p "$R" "$L"
PROG=/workspace/e1full_progress.log   # append (keep history)
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$PROG"; }
SERVE="python3 -m vllm.entrypoints.openai.api_server"
BENCH="python3 -m vllm.entrypoints.cli.main bench serve"
BT="timeout 1800"
ENVF="--enable-prefix-caching --enforce-eager --trust-remote-code --max-model-len 4096 --max-num-batched-tokens 1 --no-async-scheduling --attention-backend TRITON_ATTN --block-size 16 --gpu-memory-utilization 0.90"
FWD=/workspace/kv_distinct_fwd.jsonl
REV=/workspace/kv_distinct_rev.jsonl
BOOTPID=""
wait_gpu_clean(){ local i used procs p; for i in $(seq 1 90); do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|head -1); procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader|wc -l); [ "$procs" -eq 0 ] && [ "$used" -lt 800 ] && return 0; for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 "$p" 2>/dev/null; done; sleep 2; done; log "  WARN gpu not clean"; return 1; }
boot(){ local extra="$1" blog="$2" trace="${3:-0}" envp="" i; [ "$trace" = "1" ] && envp="VLLM_UNIFIED_POOL_TRACE=1"; env $envp HF_HOME=/workspace/hf setsid $SERVE --model "$MODEL" --port $PORT $extra $ENVF > "$blog" 2>&1 & BOOTPID=$!; for i in $(seq 1 200); do curl -sf -m2 http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && { log "  boot ready (${i}x2s)"; return 0; }; kill -0 "$BOOTPID" 2>/dev/null || { log "  BOOT DEAD -> $blog"; return 1; }; sleep 2; done; log "  BOOT TIMEOUT"; return 1; }
idle_vram(){ sleep 2; nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 0 > "$L/$1_idle_vram.txt"; }
shutdown(){ [ -n "$BOOTPID" ] || { wait_gpu_clean; return; }; kill -TERM -"$BOOTPID" 2>/dev/null || kill -TERM "$BOOTPID" 2>/dev/null || true; local i; for i in $(seq 1 15); do kill -0 "$BOOTPID" 2>/dev/null || break; sleep 1; done; kill -KILL -"$BOOTPID" 2>/dev/null || kill -KILL "$BOOTPID" 2>/dev/null || true; BOOTPID=""; sleep 1; wait_gpu_clean; }
bench_kv(){ local tag=$1
  $BT $BENCH --backend vllm --host 127.0.0.1 --port $PORT --endpoint /v1/completions --model "$MODEL" --dataset-name custom --dataset-path "$FWD" --disable-shuffle --skip-chat-template --custom-output-len 1 --num-prompts 16 --max-concurrency 1 --num-warmups 0 --seed 1 --save-detailed --result-filename "$R/${tag}_kv_cold.json" --save-result --trust-remote-code > "$L/bench_${tag}_kv_cold.log" 2>&1; log "  $tag kv-cold exit $?"
  $BT $BENCH --backend vllm --host 127.0.0.1 --port $PORT --endpoint /v1/completions --model "$MODEL" --dataset-name custom --dataset-path "$REV" --disable-shuffle --skip-chat-template --custom-output-len 1 --num-prompts 16 --max-concurrency 1 --num-warmups 0 --seed 1 --save-detailed --result-filename "$R/${tag}_kv_warm.json" --save-result --trust-remote-code > "$L/bench_${tag}_kv_warm.log" 2>&1; log "  $tag kv-warm exit $?"; }
bench_exp(){ local tag=$1 s
  for s in 1 2 3; do $BT $BENCH --backend vllm --host 127.0.0.1 --port $PORT --endpoint /v1/completions --model "$MODEL" --dataset-name random --random-input-len 256 --random-output-len 80 --random-range-ratio 0 --random-prefix-len 0 --num-prompts 12 --max-concurrency 1 --num-warmups 1 --seed $s --save-detailed --result-filename "$R/${tag}_exp_seed${s}.json" --save-result --trust-remote-code > "$L/bench_${tag}_exp_seed${s}.log" 2>&1; log "  $tag exp-seed$s exit $?"; done; }
cell_kv(){ local tag=$1 flags=$2 trace=$3; if [ -f "$R/${tag}_kv_warm.json" ]; then log "SKIP $tag / KV-heavy (done)"; return; fi; log "CELL $tag / KV-heavy"; if boot "$flags" "$L/${tag}_kv_boot.log" "$trace"; then idle_vram "${tag}_kv"; bench_kv "$tag"; else log "  SKIP $tag kv (boot fail)"; fi; shutdown; }
cell_exp(){ local tag=$1 flags=$2; if [ -f "$R/${tag}_exp_seed3.json" ]; then log "SKIP $tag / expert (done)"; return; fi; log "CELL $tag / expert-heavy"; if boot "$flags" "$L/${tag}_exp_boot.log" 0; then idle_vram "${tag}_exp"; bench_exp "$tag"; else log "  SKIP $tag exp (boot fail)"; fi; shutdown; }

VANILLA=""
STATIC32="--expert-offload --expert-cache-size 32 --num-gpu-blocks-override 3360"
STATIC48="--expert-offload --expert-cache-size 48 --num-gpu-blocks-override 1824"
STATIC64="--expert-offload --expert-cache-size 64 --num-gpu-blocks-override 288"
OURS8="--expert-offload --expert-unified-pool --expert-pool-page-tokens 16 --expert-cache-size 8 --num-gpu-blocks-override 6432"
OURS32="--expert-offload --expert-unified-pool --expert-pool-page-tokens 16 --expert-cache-size 32 --num-gpu-blocks-override 6432"
OURS48="--expert-offload --expert-unified-pool --expert-pool-page-tokens 16 --expert-cache-size 48 --num-gpu-blocks-override 6432"
OURS64="--expert-offload --expert-unified-pool --expert-pool-page-tokens 16 --expert-cache-size 64 --num-gpu-blocks-override 6432"

log "===== E1 FULL2 START (M=67; +ours32/ours64) ====="; wait_gpu_clean
# ---- ALL KV-heavy first (cold-start/risky unified cells first; done cells auto-skip) ----
cell_kv ours64   "$OURS64"   1   # extreme cold-start (init 64 -> shed to ~29)
cell_kv ours32   "$OURS32"   1   # near-optimal cold-start
cell_kv ours8    "$OURS8"    1   # headline (lean)
cell_kv ours48   "$OURS48"   1   # (done -> skip)
cell_kv static64 "$STATIC64" 0   # (done -> skip)
cell_kv static48 "$STATIC48" 0
cell_kv static32 "$STATIC32" 0
cell_kv vanilla  "$VANILLA"  0
# ---- THEN all expert-heavy ----
cell_exp ours64   "$OURS64"
cell_exp ours32   "$OURS32"
cell_exp ours8    "$OURS8"
cell_exp ours48   "$OURS48"
cell_exp static32 "$STATIC32"
cell_exp static48 "$STATIC48"
cell_exp static64 "$STATIC64"
cell_exp vanilla  "$VANILLA"
log "===== E1 FULL2 DONE ====="
