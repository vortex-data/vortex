# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Temporary paired validation; this harness is excluded from the pruning PR."""

import argparse
import csv
import hashlib
import json
import math
import os
import random
import resource
import statistics
import subprocess
import time
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--baseline", required=True, type=Path)
parser.add_argument("--candidate", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--local", action="store_true")
args = parser.parse_args()
out = args.output.resolve()
out.mkdir(parents=True, exist_ok=True)
binaries = {"baseline": args.baseline.resolve(), "same": args.baseline.resolve(), "pruning": args.candidate.resolve()}
hashes = {k: hashlib.sha256(v.read_bytes()).hexdigest() for k, v in binaries.items()}
assert hashes["baseline"] != hashes["pruning"]
cases = [
    ("tpch-sf1", "tpch", 1, None, 10),
    ("tpcds-sf1", "tpcds", 1, None, 5),
    ("clickbench", "clickbench", None, None, 3),
]
cases += [("tpcds-focus", "tpcds", 1, "4,16,59,65,66,71,72", 30)]
if not args.local:
    cases = [("polarsignals", "polarsignals", 1, None, 10), ("tpch-sf10-focus", "tpch", 10, "6,7", 20), *cases]
rng = random.Random(11348)
orders = {}
for name, *_ in cases:
    first = []
    for _ in range(3):
        row = list(binaries)
        rng.shuffle(row)
        first.append(row)
    orders[name] = first + [list(reversed(row)) for row in first]
plan = {
    "binaries": {k: str(v) for k, v in binaries.items()},
    "sha256": hashes,
    "cases": cases,
    "orders": orders,
    "affinity": sorted(os.sched_getaffinity(0)),
    "inline_array_node": os.environ.get("FLAT_LAYOUT_INLINE_ARRAY_NODE"),
    "timing": (
        "Hot median excludes first execution; CPU is whole-process user+system. No PMU instrumentation. "
        "Six process blocks, balanced ordering, identical-binary control."
    ),
}


def save(name: str, value: object):
    (out / name).write_text(json.dumps(value, indent=2) + "\n")


if (out / "plan.json").exists():
    assert json.loads((out / "plan.json").read_text()) == json.loads(json.dumps(plan))
else:
    save("plan.json", plan)
records = json.loads((out / "records.json").read_text()) if (out / "records.json").exists() else {}


def run(case: tuple[str, str, int | None, str | None, int], label: str, key: str, iterations: int) -> dict:
    if key in records:
        return records[key]
    name, suite, scale, queries, _ = case
    output = out / (key + ".jsonl")
    command = [
        str(binaries[label]),
        suite,
        "--scan-v1",
        "false",
        "--formats",
        "vortex",
        "--iterations",
        str(iterations),
        "--hide-progress-bar",
        "--display-format",
        "gh-json",
        "--runner",
        "paired-pruning-validation",
        "-o",
        str(output),
    ]
    if scale is not None:
        scale_option = str(scale) if suite == "polarsignals" else str(float(scale))
        command += ["--opt", f"scale-factor={scale_option}"]
    if queries:
        command += ["--queries", queries]
    if args.local and suite == "clickbench":
        command += [
            "--opt",
            "remote-data-dir=file:///home/ec2-user/vortex-2/.codex-worktrees/duckdb-scan-v2-fresh-20261029/vortex-bench/data/clickbench_partitioned/",
        ]
    save("active.json", {"key": key, "command": command, "started": time.time(), "completed": len(records)})
    print("START", key, flush=True)
    output.unlink(missing_ok=True)
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    start = time.monotonic()
    with (out / (key + ".log")).open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=600)
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    if result.returncode:
        detail = (out / (key + ".log")).read_text()[-8000:]
        raise RuntimeError(f"{key} failed ({result.returncode}):\n{detail}")
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    expected = {"tpch": 22, "tpcds": 99, "clickbench": 43, "polarsignals": 11}[suite]
    if queries:
        expected = len(queries.split(","))
    assert len(rows) == expected, (key, len(rows), expected)
    assert len({r["name"] for r in rows}) == expected
    assert all(len(r["all_runtimes"]) == iterations and min(r["all_runtimes"]) > 0 for r in rows)
    record = {
        "label": label,
        "case": name,
        "command": command,
        "sha256": hashes[label],
        "wall_s": time.monotonic() - start,
        "cpu_s": after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime,
        "queries": {r["name"].split("/")[0]: r["all_runtimes"] for r in rows},
    }
    records[key] = record
    save("records.json", records)
    print("DONE", key, round(record["wall_s"], 2), "s", flush=True)
    return record


bootstrap = random.Random(20261109)
samples = [[bootstrap.randrange(6) for _ in range(6)] for _ in range(10000)]


def estimate(ratios: list[float]) -> list[float]:
    logs = [math.log(x) for x in ratios]
    boot = sorted(100 * math.expm1(statistics.mean(logs[i] for i in s)) for s in samples)
    return [100 * math.expm1(statistics.mean(logs)), boot[250], boot[9749]]


results = {}
query_rows = []
summary_rows = []
for case in cases:
    name, _, _, _, iterations = case
    for label in ["baseline", "pruning"]:
        run(case, label, name + "-warmup-" + label, 1)
    blocks = []
    for block, order in enumerate(orders[name]):
        blocks.append({label: run(case, label, f"{name}-{block}-{label}", iterations) for label in order})
    queries = sorted(blocks[0]["baseline"]["queries"])
    assert all(set(r["queries"]) == set(queries) for b in blocks for r in b.values())

    def hot(record: dict, query: str) -> float:
        return statistics.median(record["queries"][query][1:])

    result = {}
    for label in ["same", "pruning"]:
        result[label] = {
            "hot": estimate(
                [
                    math.exp(statistics.mean(math.log(hot(b[label], q) / hot(b["baseline"], q)) for q in queries))
                    for b in blocks
                ]
            ),
            "cpu": estimate([b[label]["cpu_s"] / b["baseline"]["cpu_s"] for b in blocks]),
            "queries": {},
        }
        summary_rows.append(
            {
                "suite": name,
                "variant": label,
                **{
                    f"{metric}_{key}": value
                    for metric in ["hot", "cpu"]
                    for key, value in zip(
                        ["change_pct", "ci95_low_pct", "ci95_high_pct"], result[label][metric], strict=True
                    )
                },
            }
        )
        for q in queries:
            row = {
                "suite": name,
                "query": q,
                "variant": label,
                "baseline_ms": statistics.median(hot(b["baseline"], q) for b in blocks) / 1e6,
                "candidate_ms": statistics.median(hot(b[label], q) for b in blocks) / 1e6,
            }
            for control in ["baseline", "same"]:
                for key, value in zip(
                    ["change_pct", "ci95_low_pct", "ci95_high_pct"],
                    estimate([hot(b[label], q) / hot(b[control], q) for b in blocks]),
                    strict=True,
                ):
                    row[f"vs_{control}_{key}"] = value
            query_rows.append(row)
            result[label]["queries"][q] = row
        print("RESULT", name, label, json.dumps({k: v for k, v in result[label].items() if k != "queries"}), flush=True)
    results[name] = result
    save("results.json", results)
    for filename, rows in [("summary.csv", summary_rows), ("queries.csv", query_rows)]:
        with (out / filename).open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
save("active.json", {"status": "complete", "processes": len(records), "ended": time.time()})
