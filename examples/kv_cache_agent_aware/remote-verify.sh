#!/bin/bash
# Runs INSIDE the container on a GPU node. Exercises the new eviction policy
# against the real KV cache manager v2 test suite, under both policy settings.
#
# Note on exit codes: every check below captures $? from the command itself, not
# from a pipeline. `cmd | tail` yields tail's status and will mask a failure.
set -x
[ -f /etc/shinit_v2 ] && source /etc/shinit_v2
export XDG_CACHE_HOME=/lustre_scratch/.xdg-cache
cd /code/tensorrt_llm || exit 3

echo "=== VERIFY START $(date -u) host=$(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# rawref is a C extension the pure-Python package imports at load time.
( cd tensorrt_llm/runtime/kv_cache_manager_v2/rawref && python3 setup.py build_ext --inplace )
rawref_rc=$?
echo "=== RAWREF_RC=$rawref_rc ==="

python3 -c "import llist; print('llist OK')" || python3 -m pip install -q llist
# The tritondevel image carries the runtime deps but not the test-only ones.
# kernels.py needs cuda.core for the fill/check kernels FakeEngine uses to
# validate page contents, and the suite imports FakeEngine at module scope, so
# a missing cuda-core stops the whole file from loading.
python3 -c "import parameterized; print('parameterized OK')" || python3 -m pip install -q parameterized
python3 -c "import cuda.core; print('cuda.core OK')" || python3 -m pip install -q cuda-core

SUITE=tests/unittest/kv_cache_manager_v2_tests/test_kv_cache_manager_v2.py
export PYTHONPATH=/code/tensorrt_llm/tensorrt_llm/runtime

# 1. Default policy must still be lru, and the suite must pass unchanged.
unset TLLM_KV_EVICTION_POLICY
python3 -c "
from kv_cache_manager_v2._eviction_controller import EVICTION_POLICY_NAME, make_eviction_policy
print('DEFAULT_POLICY=' + EVICTION_POLICY_NAME)
print('FACTORY=' + type(make_eviction_policy()).__name__)
assert EVICTION_POLICY_NAME == 'lru', 'default must stay lru'
"
default_rc=$?
echo "=== DEFAULT_CHECK_RC=$default_rc ==="

python3 "$SUITE" > /lustre_scratch/cachesage-suite-lru.txt 2>&1
lru_rc=$?
tail -5 /lustre_scratch/cachesage-suite-lru.txt
echo "=== SUITE_LRU_RC=$lru_rc ==="

# 2. Same suite with the new policy engaged. This is what exercises
#    Page.eviction_ordinal and DepthAwareEvictionPolicy on real pages.
export TLLM_KV_EVICTION_POLICY=depth
python3 -c "
from kv_cache_manager_v2._eviction_controller import EVICTION_POLICY_NAME, make_eviction_policy
print('SELECTED_POLICY=' + EVICTION_POLICY_NAME)
print('FACTORY=' + type(make_eviction_policy()).__name__)
assert EVICTION_POLICY_NAME == 'depth'
"
depth_rc=$?
echo "=== DEPTH_CHECK_RC=$depth_rc ==="

python3 "$SUITE" > /lustre_scratch/cachesage-suite-depth.txt 2>&1
depth_suite_rc=$?
tail -5 /lustre_scratch/cachesage-suite-depth.txt
echo "=== SUITE_DEPTH_RC=$depth_suite_rc ==="

# 3. Reject an unknown policy name rather than silently falling back.
TLLM_KV_EVICTION_POLICY=bogus python3 -c "
import kv_cache_manager_v2._eviction_controller
" 2>&1 | tail -2
echo "=== BOGUS_REJECTED (expect a ValueError above) ==="

echo "=== SUMMARY rawref=$rawref_rc default=$default_rc suite_lru=$lru_rc depth=$depth_rc suite_depth=$depth_suite_rc ==="
[ $default_rc -eq 0 ] && [ $lru_rc -eq 0 ] && [ $depth_rc -eq 0 ] && [ $depth_suite_rc -eq 0 ]
exit $?
