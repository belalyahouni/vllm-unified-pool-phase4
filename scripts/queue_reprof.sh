#!/bin/bash
# Wait for the in-flight profiled run to finish, then re-profile on the
# current checkout.
#
# Purpose: commit d6ba38d replaced the tier-1 pool scan in
# _select_super_block_for_expert with a holder lookup plus a free-page
# guard. The pre-change run measured select_super_block at 181 us over
# 76,731 calls (13.9 s, 0.68% of a 2057 s run); this run measures the same
# counter on the new code so the appendix can quote a host-side cost.
#
# The relocation figures already in the appendix are unaffected -- the
# change is host-side only, and the fuzzer confirms the same super-blocks
# are chosen -- so this run is only expected to move the CPU counters.
# The new free_pure_scan_skipped / free_pure_scan_ran counters also show
# whether the guard actually fires on a saturated pool, which the fuzzer
# cannot tell us (its pool has far more free space).
#
# Results go to a fresh directory so the pre-change numbers survive for
# comparison.
set -u
cd "$(dirname "$0")/.."

log(){ echo "[$(date +%H:%M:%S)] queue: $*"; }

# 1. Wait for the GPU to go idle. The in-flight run holds it; do not kill
#    it, its exp2kv KV phases are still finishing.
log "waiting for GPU to go idle"
for i in $(seq 1 900); do   # up to ~1h
  n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
  [ "$n" -eq 0 ] && { log "GPU idle after $((i*4))s"; break; }
  sleep 4
done
n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
[ "$n" -ne 0 ] && { log "ABORT: GPU still busy"; exit 1; }
sleep 20

# 2. Take the new code.
log "pulling"
git pull --ff-only 2>&1 | tail -3
log "HEAD $(git log --oneline -1)"

# 3. Re-profile into a fresh directory.
export R=/root/e2prof2_results L=/root/e2prof2_logs
log "starting run_e2_prof.sh (R=$R)"
bash scripts/run_e2_prof.sh
log "done"
