# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Flatten LO2v2 Prometheus range-query dumps into one JSON line per series.

Download `light-oauth2-metrics.zip` from https://zenodo.org/records/18937117, unzip it, then:

    python3 lo2_to_jsonl.py light-oauth2-locust-data-LO2v2/LO2_run_1739743201 lo2_series.jsonl
"""

import json, glob, sys
root = sys.argv[1]; out = open(sys.argv[2], "w")
n = s = 0
for f in sorted(glob.glob(f"{root}/*/metrics/metric_*.json")):
    test = f.split("/")[-3]
    try:
        d = json.load(open(f))
    except Exception:
        continue
    for series in d:
        vals = series.get("values") or []
        if not vals: continue
        out.write(json.dumps({"labels": series["metric"], "test": test,
                              "ts": [v[0] for v in vals], "vals": [float(v[1]) for v in vals]}, separators=(",", ":")) + "\n")
        n += 1; s += len(vals)
print(n, "series", s, "samples")
