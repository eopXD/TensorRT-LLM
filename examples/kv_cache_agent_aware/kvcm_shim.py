# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Import the CUDA-free parts of ``kv_cache_manager_v2`` on a machine with no GPU.

``kv_cache_manager_v2/__init__.py`` eagerly loads the C++ bindings and the CUDA
virtual-memory allocator, so a plain ``import`` fails on a laptop. But
``_common.py`` is stdlib-only and ``_cache_key.py`` imports nothing but
``_common``, so both can be loaded directly under a synthetic package that skips
the real ``__init__``.

This matters for the honesty of the offline numbers: the simulator hashes blocks
with the *production* ``sequence_to_blockchain_keys`` and the production SHA-256
``Hasher``, so block boundaries and prefix-match semantics cannot drift from
what the engine actually does.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import types

_PKG = "_kvcm2_pure"
_MODULES = ("_common", "_cache_key", "_agent_aware")


def _kvcm_dir() -> pathlib.Path:
    here = pathlib.Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "tensorrt_llm" / "runtime" / "kv_cache_manager_v2"
        if candidate.is_dir():
            return candidate
    raise RuntimeError(f"could not locate kv_cache_manager_v2 above {here}")


def _load() -> types.ModuleType:
    if _PKG in sys.modules:
        return sys.modules[_PKG]

    src = _kvcm_dir()
    pkg = types.ModuleType(_PKG)
    pkg.__path__ = [str(src)]
    sys.modules[_PKG] = pkg

    for name in _MODULES:
        path = src / f"{name}.py"
        if not path.exists():
            # _agent_aware.py is added by this PoC; tolerate its absence so the
            # baseline sweep still runs on an unmodified checkout.
            continue
        spec = importlib.util.spec_from_file_location(f"{_PKG}.{name}", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"{_PKG}.{name}"] = module
        spec.loader.exec_module(module)
        setattr(pkg, name, module)

    return pkg


_pkg = _load()

_common = _pkg._common
_cache_key = _pkg._cache_key
_agent_aware = getattr(_pkg, "_agent_aware", None)

Hasher = _cache_key.Hasher
sequence_to_blockchain_keys = _cache_key.sequence_to_blockchain_keys
reuse_scope_to_bytes = _cache_key.reuse_scope_to_bytes

PRIORITY_DEFAULT = _common.PRIORITY_DEFAULT
PRIORITY_MIN = _common.PRIORITY_MIN
PRIORITY_MAX = _common.PRIORITY_MAX

KVCM_SOURCE_DIR = str(_kvcm_dir())
