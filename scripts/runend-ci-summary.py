# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Compare adaptive dispatch against forced kernels on exactly the same inputs."""

import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

path = Path(sys.argv[1])
groups = defaultdict(dict)
with path.open() as source:
    for row in csv.DictReader(source):
        key = row["type"], int(row["length"]), row["pattern"]
        groups[key][row["variant"], int(row["round"])] = float(row["ns"])

decisions = defaultdict(set)
with path.with_name("decisions.csv").open() as source:
    for row in csv.DictReader(source):
        key = row["type"], int(row["length"]), row["pattern"]
        decisions[key].add(row["selected"])

print("Each ratio compares identical inputs in 12 alternating rounds.\n")
print("Forced/pr below 1 means forcing that kernel is faster than the PR's adaptive selection.")
print("Wide values bypass segment selection; their forced head/exact variants are unchanged controls.\n")
print("| Type | Rows | Pattern | PR choice | PR ns | PR/baseline | Head/pr | Exact/pr | Staged direct/pr | Rollback/pr |")
print("|---|---:|---|---|---:|---:|---:|---:|---:|---:|")
misses = []
for key, values in groups.items():
    rounds = sorted({r for _, r in values})

    def ratio(a, b):
        return statistics.median(values[a, r] / values[b, r] for r in rounds)

    choices = "+".join(sorted(decisions[key]))
    ns = statistics.median(values["pr", r] for r in rounds)
    comparisons = [ratio("pr", "baseline")] + [
        ratio(v, "pr") for v in ("forced_head", "forced_exact", "staged_direct", "candidate")
    ]
    ratios = " | ".join(f"{r:.3f}" for r in comparisons)
    print(f"| {key[0]} | {key[1]} | {key[2]} | {choices} | {ns:.1f} | {ratios} |")
    if "wide_exact" not in decisions[key]:
        for variant in ("forced_head", "forced_exact"):
            faster = sum(values[variant, r] < 0.95 * values["pr", r] for r in rounds)
            if faster >= len(rounds) - 1:
                misses.append((key, variant, ratio(variant, "pr"), faster))

print(f"\nCases where a forced kernel beats the PR by >5% in at least 11/12 rounds: {len(misses)}\n")
for key, variant, ratio_value, faster in misses:
    print(f"- {key}: {variant}/pr={ratio_value:.3f}; {faster}/12 rounds")
