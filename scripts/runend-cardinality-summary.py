# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Summarize local small-dictionary NEON experiments, including full decode costs."""

import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

groups = defaultdict(dict)
with Path(sys.argv[1]).open() as source:
    for row in csv.DictReader(source):
        key = int(row["cardinality"]), int(row["length"]), row["pattern"]
        groups[key][row["variant"], int(row["round"])] = float(row["ns"])

print("u32 values; pre-existing dictionary and byte codes; allocation and table setup included.")
print("Dictionary discovery, code validation, null handling and packed-code unpacking are excluded.")
print("Ratios below 1 mean less time. Each ratio is the median of 24 paired rounds.")
print("Best fused compares decoders starting from the same run codes and dictionary.\n")
print("| Dictionary | Rows | Pattern | NEON ns | NEON/scalar or PR | Best fused | NEON/best | Wins vs best |")
print("|---:|---:|---|---:|---:|---|---:|---:|")
for (cardinality, length, pattern), values in groups.items():
    rounds = range(24)
    variants = {v for v, _ in values}
    assert len(values) == len(variants) * len(rounds)
    assert all((v, r) in values for v in variants for r in rounds)

    def median_ns(variant):
        return statistics.median(values[variant, r] for r in rounds)

    def ratio(a, b):
        return statistics.median(values[a, r] / values[b, r] for r in rounds)

    if pattern == "lookup":
        candidate = "neon"
        reference = best = "scalar"
    else:
        candidate = "dict_neon_pipeline"
        reference = "pr_materialized"
        best = min(
            ("dict_fused_pr", "dict_fused_baseline", "dict_fused_exact"),
            key=median_ns,
        )
    wins = sum(values[candidate, r] < values[best, r] for r in rounds)
    print(
        f"| {cardinality} | {length} | {pattern} | {median_ns(candidate):.1f} | "
        f"{ratio(candidate, reference):.3f} | {best} | {ratio(candidate, best):.3f} | {wins}/{len(rounds)} |"
    )
