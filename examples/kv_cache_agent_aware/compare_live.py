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
"""Compare the offline simulator's predictions against the live engine.

Reads the per-cell JSON files written by run_live.py and reports, per cache
budget, the predicted and measured block hit rate for each policy and the
predicted vs measured *delta* between policies.

The delta is the number that matters. Absolute hit rate can drift for reasons
the simulator deliberately does not model (warmup requests, the engine's own
bookkeeping blocks, partial-block edge cases). The claim under test is that
switching eviction policy moves the hit rate by roughly the predicted amount.

    python3 compare_live.py .cachesage-exp
"""

from __future__ import annotations

import json
import pathlib
import sys

# Tolerance on the delta, in percentage points. Wider than measurement noise but
# far narrower than the effect being claimed (+10 to +14 pp), so a real
# disagreement still fails.
DELTA_TOLERANCE_PP = 4.0


def main() -> None:
    root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".cachesage-exp")
    cells: dict[int, dict[str, dict]] = {}
    for path in sorted(root.glob("*.json")):
        data = json.loads(path.read_text())
        cells.setdefault(int(data["max_tokens"]), {})[data["policy"]] = data

    if not cells:
        sys.exit(f"no result JSON found under {root}")

    header = (
        f"{'max_tokens':>11} {'blocks':>7} "
        f"{'lru pred':>9} {'lru meas':>9} "
        f"{'dep pred':>9} {'dep meas':>9} "
        f"{'Δ pred':>8} {'Δ meas':>8} {'verdict':>10}"
    )
    print(header)
    print("-" * len(header))

    verdicts = []
    for max_tokens in sorted(cells):
        cell = cells[max_tokens]
        if "lru" not in cell or "depth" not in cell:
            print(f"{max_tokens:>11} INCOMPLETE (have {sorted(cell)})")
            continue
        lru, dep = cell["lru"], cell["depth"]
        lp, lm = lru["predicted_block_hit"], lru["measured_block_hit"]
        dp, dm = dep["predicted_block_hit"], dep["measured_block_hit"]
        d_pred = (dp - lp) * 100
        d_meas = (dm - lm) * 100
        ok = abs(d_pred - d_meas) <= DELTA_TOLERANCE_PP
        verdicts.append(ok)
        print(
            f"{max_tokens:>11} {lru['budget_blocks']:>7} "
            f"{lp:>8.2%} {lm:>8.2%} {dp:>8.2%} {dm:>8.2%} "
            f"{d_pred:>+7.1f} {d_meas:>+7.1f} {'AGREE' if ok else 'DISAGREE':>10}"
        )

    print()
    if verdicts and all(verdicts):
        print(
            f"VERDICT: simulator VALIDATED -- every predicted policy delta is "
            f"within {DELTA_TOLERANCE_PP} pp of measured."
        )
    else:
        print(
            f"VERDICT: simulator NOT validated -- at least one predicted delta "
            f"missed by more than {DELTA_TOLERANCE_PP} pp. The offline "
            f"conclusions rest on this model and must be re-derived."
        )


if __name__ == "__main__":
    main()
