#!/bin/bash
# Clone TensorRT-LLM on nsc-svg for the CacheSage PoC and wire up the fork remote.
set -euo pipefail

WS=/lustre/fsw/portfolios/coreai/projects/coreai_comparch_trtllm/users/yuehtingc
DST="$WS/TensorRT-LLM-cachesage"
LOG="$WS/cachesage-clone.log"

exec >>"$LOG" 2>&1
echo "=== clone start $(date -Is) on $(hostname) ==="

# Skip LFS during the clone so a flaky LFS endpoint cannot fail the whole
# checkout, then pull the objects explicitly below. Do NOT leave LFS skipped:
# cpp/tensorrt_llm/CMakeLists.txt:135 hard-fails the configure step if
# tensorrt_llm_internal_cutlass_kernels_static.tar.xz is still a pointer file.
export GIT_LFS_SKIP_SMUDGE=1

if [ -d "$DST/.git" ]; then
    echo "already cloned at $DST"
else
    git clone --recurse-submodules git@github.com:NVIDIA/TensorRT-LLM.git "$DST"
fi

cd "$DST"
git lfs install --local
GIT_LFS_SKIP_SMUDGE=0 git lfs pull
CUTLASS_TARBALL=cpp/tensorrt_llm/kernels/internal_cutlass_kernels/$(uname -m)-linux-gnu/tensorrt_llm_internal_cutlass_kernels_static.tar.xz
if [ "$(stat -c %s "$CUTLASS_TARBALL" 2>/dev/null || echo 0)" -lt 1024 ]; then
    echo "FATAL: $CUTLASS_TARBALL is still an LFS pointer; the build would fail at cmake configure"
    exit 4
fi
echo "cutlass tarball OK: $(stat -c %s "$CUTLASS_TARBALL") bytes"
git remote remove fork 2>/dev/null || true
git remote add fork git@github.com:eopXD/TensorRT-LLM.git
git remote -v
git config user.email yuehtingc@nvidia.com
git fetch fork --prune || echo "fork fetch failed (non-fatal)"

echo "HEAD: $(git log --oneline -1)"
echo "=== clone done $(date -Is) rc=0 ==="
