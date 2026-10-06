#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
"""Warm-first-iteration baseline: identical to `bench_ab.py cold` (fresh process per query, -i 1)
but WITHOUT evicting the page cache. cold - warm_first isolates the cost of physical IO from the
cost of a fresh process (lazy init, footer parse, thread start).

usage: bench_first.py <engine> <suite> <sf|-> <reps> [queries]
Tables go to results-warmfirst.md; the table title still says COLD, the file name is the truth.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench_ab  # noqa: E402

bench_ab.RESULTS = os.path.join(HERE, "results-warmfirst.md")
bench_ab.RAW = os.path.join(HERE, "raw-warmfirst")
os.makedirs(bench_ab.RAW, exist_ok=True)
bench_ab.evict_all = lambda suite, sf: (0, 0, 0)

if __name__ == "__main__":
    engine, suite, sf, reps = sys.argv[1:5]
    bench_ab.emit("\n(NOT EVICTED: warm page cache, fresh process per query, -i 1)")
    bench_ab.cold(engine, suite, sf, int(reps), sys.argv[5] if len(sys.argv) > 5 else None)
