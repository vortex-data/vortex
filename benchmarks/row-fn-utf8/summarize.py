# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
"""Summarize raw observations. Arguments have the form label=path-glob."""

import csv
import glob
import statistics
import sys

writer = csv.writer(sys.stdout)
writer.writerow(["label", "case", "rows", "samples", "median_ns", "p25_ns", "p75_ns"])
for argument in sys.argv[1:]:
    label, pattern = argument.split("=", 1)
    observations = {}
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise ValueError(f"No files matched {pattern}")
    for path in paths:
        with open(path) as source:
            for row in csv.DictReader(source):
                key = row["case"], int(row["rows"])
                observations.setdefault(key, []).append(float(row["ns"]))
    for (case, rows), values in sorted(observations.items()):
        p25, _, p75 = statistics.quantiles(values, n=4, method="inclusive")
        writer.writerow([
            label, case, rows, len(values),
            f"{statistics.median(values):.3f}", f"{p25:.3f}", f"{p75:.3f}",
        ])
