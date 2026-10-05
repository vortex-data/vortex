# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Report paired ratios; a regression is evidence against an unconditional win."""

import csv
import math
import statistics
import sys
from collections import defaultdict

groups = defaultdict(dict)
with open(sys.argv[1]) as source:
    for row in csv.DictReader(source):
        key = row["type"], int(row["length"]), row["pattern"]
        groups[key][row["variant"], int(row["round"])] = float(row["ns"])

print("Ratios are candidate / PR (below 1 is faster). Each cell has 9 alternating rounds.\n")
print("| Type | Rows | Pattern | Baseline ns | PR ns | Candidate ns | PR/baseline | Candidate/PR | Paired min–max | Slower rounds |")
print("|---|---:|---|---:|---:|---:|---:|---:|---|---:|")
results = []
for key, values in groups.items():
    old = [values["pr", i] for i in range(9)]
    new = [values["candidate", i] for i in range(9)]
    baseline = [values["baseline", i] for i in range(9)]
    pr_ratio = statistics.median(o / b for o, b in zip(old, baseline, strict=True))
    ratios = [n / o for n, o in zip(new, old, strict=True)]
    result = key, statistics.median(ratios), sum(r > 1.05 for r in ratios)
    results.append(result)
    print(f"| {key[0]} | {key[1]} | {key[2]} | {statistics.median(baseline):.1f} | {statistics.median(old):.1f} | {statistics.median(new):.1f} | {pr_ratio:.3f} | {result[1]:.3f} | {min(ratios):.3f}–{max(ratios):.3f} | {result[2]}/9 |")

wide = [r for r in results if r[0][0] in {"i128", "BinaryView", "wide32"}]
regressed = [r for r in wide if r[1] > 1.05 and r[2] >= 8]
print(f"\nWide-value geomean: {math.exp(statistics.mean(math.log(r[1]) for r in wide)):.3f}x")
print(f"Consistent wide-value regressions (>5% in at least 8/9 rounds): {len(regressed)}")
print("This finite workload matrix cannot prove an unconditional performance win.")
