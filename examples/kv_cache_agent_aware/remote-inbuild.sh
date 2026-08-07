#!/bin/bash
# Runs INSIDE the tritondevel container. Repo at /code/tensorrt_llm, Lustre scratch at /lustre_scratch.
set -x
[ -f /etc/shinit_v2 ] && source /etc/shinit_v2
export CCACHE_DIR=/lustre_scratch/.ccache
export PIP_CACHE_DIR=/lustre_scratch/.pip-cache
export HF_HOME=/lustre_scratch/.hf
export XDG_CACHE_HOME=/lustre_scratch/.xdg-cache
export CCACHE_MAXSIZE=80G
mkdir -p "$CCACHE_DIR" "$PIP_CACHE_DIR" "$HF_HOME" "$XDG_CACHE_HOME"
cd /code/tensorrt_llm || exit 3
echo "=== BUILD START $(date -u) nproc=$(nproc) arch=$(uname -m) host=$(hostname) ==="
python3 --version

# Exercise the new pure-Python module before spending an hour on C++: it is
# stdlib-only, so a failure here is a code error, not an environment problem.
python3 tests/unittest/kv_cache_manager_v2_tests/test_agent_aware.py 2>&1 | tail -5
echo "=== agent_aware unit tests rc=$? ==="

# 89-real (Ada) per this user's standing default; 100-real for the B200 this
# wheel will run on.
python3 scripts/build_wheel.py --cuda_architectures "89-real;100-real" --job_count "$(nproc)"
rc=$?
echo "=== BUILD_INNER_RC=$rc $(date -u) ==="
ls -la /code/tensorrt_llm/build/tensorrt_llm-*.whl 2>&1

# Confirm the new eviction policy is reachable and still defaults to lru.
#
# Import the package directly off PYTHONPATH rather than pip-installing the
# wheel: `pip install --no-deps` leaves transformers missing, and
# `import tensorrt_llm` pulls it in via llmapi, so the check fails for a reason
# that has nothing to do with this change. The runtime suite in
# remote-verify.sh is what actually exercises the policy.
#
# Capture $? from python itself, never from a pipeline -- `python | tail` yields
# tail's status and will report success for a failed check.
if [ $rc -eq 0 ]; then
    PYTHONPATH=/code/tensorrt_llm/tensorrt_llm/runtime python3 -c "
from kv_cache_manager_v2._eviction_controller import (
    EVICTION_POLICY_NAME, DepthAwareEvictionPolicy, make_eviction_policy)
print('policy default =', EVICTION_POLICY_NAME)
print('factory ->', type(make_eviction_policy()).__name__)
assert EVICTION_POLICY_NAME == 'lru', 'default must stay lru'
print('POST_BUILD_IMPORT_OK')
"
    echo "=== POST_BUILD_RC=$? ==="
fi
exit $rc
