#!/bin/bash
# One-shot setup of the Phase-4 fork on a fresh `vllm/vllm-openai:v0.17.1` pod.
#
# The pod MUST be started with its Container Start Command overridden to
# `sleep infinity` (the image's default entrypoint is `vllm serve`, which
# auto-launches a server that eats the whole GPU).
#
# This reproduces a working install WITHOUT pip's editable install (which
# mis-maps this repo's nested `vllm/` layout as a namespace package). It:
#   - clones the fork,
#   - fetches the matching vLLM 0.17.1 wheel purely for its compiled kernels,
#   - drops those .so into the fork tree (paths line up 1:1),
#   - puts the fork on sys.path via a .pth (so `import vllm` == the fork),
#   - installs the 0.17.1 dist-info + a _version.py so version/platform
#     detection works.
#
# Usage on the pod:
#   curl -fsSL https://raw.githubusercontent.com/belalyahouni/vllm-unified-pool-phase4/main/scripts/pod_setup.sh | bash
set -euo pipefail

FORK=/root/vllm-unified-pool-phase4
SP=$(python3 -c "import site;print(site.getsitepackages()[0])")
echo "[setup] site-packages: $SP"

# 1. git + clone (idempotent)
if ! command -v git >/dev/null; then
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git
fi
if [ ! -d "$FORK/.git" ]; then
  git clone --depth 1 https://github.com/belalyahouni/vllm-unified-pool-phase4.git "$FORK"
fi
echo "[setup] fork at $FORK"

# 2. matching 0.17.1 kernel wheel (only used for its compiled .so)
mkdir -p /root/wheels
WHL=$(ls /root/wheels/vllm-0.17.1*.whl 2>/dev/null | head -1 || true)
if [ -z "$WHL" ]; then
  pip download vllm==0.17.1 --no-deps -d /root/wheels
  WHL=$(ls /root/wheels/vllm-0.17.1*.whl | head -1)
fi
echo "[setup] wheel: $WHL"

# 3. extract the compiled .so from the wheel into the fork tree (paths align:
#    wheel 'vllm/..._C.abi3.so' -> $FORK/vllm/vllm/..._C.abi3.so)
python3 - "$WHL" "$FORK/vllm" <<'PY'
import sys, zipfile
whl, dest = sys.argv[1], sys.argv[2]
z = zipfile.ZipFile(whl)
n = 0
for name in z.namelist():
    if name.endswith(".so"):
        z.extract(name, dest); n += 1
print(f"[setup] extracted {n} kernel .so into {dest}")
PY

# 4. make the fork the active vllm: drop the image's pip install, add a .pth
pip uninstall -y vllm >/dev/null 2>&1 || true
echo "$FORK/vllm" > "$SP/vllm_fork.pth"

# 5. metadata so importlib.metadata + platform detection work
python3 - "$WHL" "$SP" <<'PY'
import sys, zipfile
whl, sp = sys.argv[1], sys.argv[2]
z = zipfile.ZipFile(whl)
n = 0
for name in z.namelist():
    if name.startswith("vllm-0.17.1.dist-info/"):
        z.extract(name, sp); n += 1
print(f"[setup] installed {n} dist-info entries")
PY
cat > "$FORK/vllm/vllm/_version.py" <<'PY'
__version__ = "0.17.1"
__version_tuple__ = (0, 17, 1)
PY

# 6. verify
cd /root
python3 - <<'PY'
import torch, vllm, importlib
print("[verify] vllm", vllm.__version__, "->", vllm.__file__)
for m in ["vllm.config.offload",
          "vllm.engine.arg_utils",
          "vllm.v1.engine.core",
          "vllm.v1.worker.gpu_model_runner",
          "vllm.v1.worker.gpu_worker",
          "vllm.model_executor.layers.fused_moe.unified_pool",
          "vllm.model_executor.layers.fused_moe.expert_cache",
          "vllm.entrypoints.openai.api_server"]:
    importlib.import_module(m)
import vllm._C
from vllm.config.offload import OffloadConfig
assert hasattr(OffloadConfig, "expert_pool_page_tokens"), "phase-4 field missing"
from vllm.platforms import current_platform
print("[verify] platform", type(current_platform).__name__,
      "cuda", current_platform.is_cuda())
print("SETUP_OK")
PY
echo "[setup] done. Serve with:  python3 -m vllm.entrypoints.openai.api_server --model allenai/OLMoE-1B-7B-0924-Instruct ..."
