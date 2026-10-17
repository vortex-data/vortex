#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Memory monitoring for large-scale benchmark runs.

Three subcommands, all stdlib-only so they run before any Python environment is set up:

- ``monitor`` samples system memory and the largest process into a CSV until it is terminated.
  Each sample is tagged with the contents of a phase file, which the other steps rewrite to
  attribute memory to data generation, conversion, or an individual query.
- ``run-queries`` runs every query of a benchmark binary in its own process, so one query
  running out of memory or time is recorded and the rest still run. Each query gets its own peak
  RSS (from ``wait4``) and a status of ``ok``, ``oom-killed``, ``alloc-failed``, ``timeout``,
  ``error`` or ``skipped`` (the time budget ran out first).
- ``summarize`` renders the monitor CSV and query records as GitHub Markdown.
"""

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

KIB = 1024
GIB = 1024**3


def read_meminfo() -> dict[str, int]:
    """Return /proc/meminfo in bytes."""
    info = {}
    with open("/proc/meminfo") as f:
        for line in f:
            key, rest = line.split(":", 1)
            info[key] = int(rest.split()[0]) * KIB
    return info


def memory_pressure() -> str:
    """Return the 10s average of the "some" memory pressure stall, if the kernel exposes PSI."""
    try:
        with open("/proc/pressure/memory") as f:
            for line in f:
                if line.startswith("some"):
                    return line.split()[1].split("=", 1)[1]
    except OSError:
        pass
    return ""


def largest_process() -> tuple[int, str, int]:
    """Return (pid, command name, RSS bytes) of the process with the largest resident set."""
    best = (0, "", 0)
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit():
            continue
        try:
            with open(f"/proc/{entry.name}/status") as f:
                name, rss = "", 0
                for line in f:
                    if line.startswith("Name:"):
                        name = line.split(None, 1)[1].strip()
                    elif line.startswith("VmRSS:"):
                        rss = int(line.split()[1]) * KIB
                        break
        except OSError:
            continue
        if rss > best[2]:
            best = (int(entry.name), name, rss)
    return best


def read_phase(path: Path) -> str:
    try:
        return path.read_text().strip() or "idle"
    except OSError:
        return "idle"


def cmd_monitor(args: argparse.Namespace) -> None:
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())

    fields = [
        "timestamp",
        "phase",
        "mem_total_gib",
        "mem_used_gib",
        "mem_available_gib",
        "page_cache_gib",
        "swap_used_gib",
        "psi_some_avg10",
        "top_pid",
        "top_comm",
        "top_rss_gib",
    ]
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        while not stop.is_set():
            mem = read_meminfo()
            pid, comm, rss = largest_process()
            writer.writerow(
                {
                    "timestamp": f"{time.time():.1f}",
                    "phase": read_phase(args.phase_file),
                    "mem_total_gib": f"{mem['MemTotal'] / GIB:.2f}",
                    "mem_used_gib": f"{(mem['MemTotal'] - mem['MemAvailable']) / GIB:.2f}",
                    "mem_available_gib": f"{mem['MemAvailable'] / GIB:.2f}",
                    "page_cache_gib": f"{mem.get('Cached', 0) / GIB:.2f}",
                    "swap_used_gib": f"{(mem.get('SwapTotal', 0) - mem.get('SwapFree', 0)) / GIB:.2f}",
                    "psi_some_avg10": memory_pressure(),
                    "top_pid": pid,
                    "top_comm": comm,
                    "top_rss_gib": f"{rss / GIB:.2f}",
                }
            )
            f.flush()
            stop.wait(args.interval)


def list_queries(binary: str, benchmark: str, options: list[str]) -> list[int]:
    cmd = [binary, benchmark, "--print-queries", *options]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    return [int(line) for line in out.split() if line.strip()]


def kernel_oom_killed(pid: int) -> bool:
    """Whether the kernel log records the OOM killer taking this pid."""
    try:
        log = subprocess.run(["sudo", "-n", "dmesg"], capture_output=True, text=True, check=False).stdout
    except OSError:
        return False
    return any(f"Killed process {pid} " in line or f"Killed process {pid}," in line for line in log.splitlines())


def classify(returncode: int, reason: str | None, pid: int, stderr_tail: str) -> str:
    if reason == "timeout":
        return "timeout"
    if reason == "memory":
        return "oom-killed"
    if returncode == 0:
        return "ok"
    if "Resources exhausted" in stderr_tail:
        return "resources-exhausted"
    if "memory allocation of" in stderr_tail or "Out of Memory Error" in stderr_tail:
        return "alloc-failed"
    if returncode == -signal.SIGKILL and kernel_oom_killed(pid):
        return "oom-killed"
    if returncode == -signal.SIGKILL:
        # Without dmesg access a SIGKILL we did not send is almost always the OOM killer.
        return "oom-killed"
    return "error"


def raise_oom_score() -> None:
    # Make the benchmark the OOM killer's first choice so a runaway query cannot take the
    # runner agent down with it. Raising the score needs no privileges.
    try:
        with open("/proc/self/oom_score_adj", "w") as f:
            f.write("1000")
    except OSError:
        pass


def run_query(
    cmd: list[str], stdout_path: Path, stderr_path: Path, timeout_s: int, min_available: int
) -> tuple[int, str | None, int, int]:
    """Run one query process; return (returncode, kill reason, pid, peak RSS bytes).

    The query is killed when it times out, or when system available memory drops below
    ``min_available``: under heavy pressure systemd-oomd or the runner agent can be killed before
    the kernel OOM killer reaches the query, which loses the whole job, so the query goes first.
    """
    with open(stdout_path, "w") as out, open(stderr_path, "w") as err:
        proc = subprocess.Popen(cmd, stdout=out, stderr=err, start_new_session=True, preexec_fn=raise_oom_score)
        deadline = time.monotonic() + timeout_s
        reason = None
        while True:
            # wait4 reports this child's own peak RSS, unlike the cumulative RUSAGE_CHILDREN.
            pid, status, usage = os.wait4(proc.pid, os.WNOHANG)
            if pid:
                break
            if reason is None:
                if time.monotonic() > deadline:
                    reason = "timeout"
                elif read_meminfo()["MemAvailable"] < min_available:
                    reason = "memory"
                if reason is not None:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            time.sleep(0.1)
        proc.returncode = os.waitstatus_to_exitcode(status)
    return proc.returncode, reason, proc.pid, usage.ru_maxrss * KIB


def tail(path: Path, lines: int = 20) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


def cmd_run_queries(args: argparse.Namespace) -> int:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    options = [arg for opt in args.opt for arg in ("--opt", opt)]
    queries = args.queries or list_queries(args.binary, args.benchmark, options)
    records_path = args.out_dir / "queries.jsonl"
    deadline = time.monotonic() + args.budget if args.budget else None
    # The variant keeps records and memory phases of differently configured runs of one engine apart.
    engine = f"{args.engine}-{args.variant}" if args.variant else args.engine
    failures = 0

    for fmt in args.formats:
        for query in queries:
            label = f"{engine}:{fmt}:q{query:02d}"
            args.phase_file.write_text(label)
            stem = args.out_dir / f"{engine}-{fmt}-q{query:02d}"
            cmd = [
                args.binary,
                args.benchmark,
                "--formats",
                fmt,
                "--queries",
                str(query),
                "--iterations",
                str(args.iterations),
                "--display-format",
                "gh-json",
                "--hide-progress-bar",
                "--track-memory",
                "--runner",
                args.runner,
                *options,
            ]
            if args.engine == "duckdb":
                cmd.append("--delete-duckdb-database")

            if deadline is not None and time.monotonic() > deadline:
                failures += 1
                print(f"::warning title={label} skipped::time budget of {args.budget}s exhausted", flush=True)
                with open(records_path, "a") as f:
                    f.write(json.dumps({"engine": engine, "format": fmt, "query": query, "status": "skipped"}) + "\n")
                continue

            subprocess.run(["sync"], check=False)
            subprocess.run(
                ["sudo", "-n", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"], check=False, capture_output=True
            )

            print(f"::group::{label}", flush=True)
            start = time.monotonic()
            returncode, reason, pid, peak_rss = run_query(
                cmd,
                stem.with_suffix(".jsonl"),
                stem.with_suffix(".log"),
                args.timeout,
                int(args.min_available_gib * GIB),
            )
            elapsed = time.monotonic() - start
            # Classify on a long tail: a backtrace can push the error message far from the end.
            status = classify(returncode, reason, pid, tail(stem.with_suffix(".log"), 400))
            print(tail(stem.with_suffix(".log")))
            print("::endgroup::", flush=True)

            record = {
                "engine": engine,
                "format": fmt,
                "query": query,
                "status": status,
                "returncode": returncode,
                "wall_s": round(elapsed, 1),
                "peak_rss_gib": round(peak_rss / GIB, 2),
            }
            with open(records_path, "a") as f:
                f.write(json.dumps(record) + "\n")

            if status == "ok":
                print(f"{label}: ok in {elapsed:.1f}s, peak RSS {peak_rss / GIB:.1f} GiB", flush=True)
            else:
                failures += 1
                print(
                    f"::error title={label} {status}::exit {returncode} after {elapsed:.0f}s, "
                    f"peak RSS {peak_rss / GIB:.1f} GiB",
                    flush=True,
                )
            args.phase_file.write_text("idle")

    return 1 if failures and args.fail_on_error else 0


def peak_by_phase(samples: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    peaks: dict[str, dict[str, str]] = {}
    for row in samples:
        phase = row["phase"]
        if phase not in peaks or float(row["mem_used_gib"]) > float(peaks[phase]["mem_used_gib"]):
            peaks[phase] = row
    return peaks


def cmd_summarize(args: argparse.Namespace) -> None:
    samples: list[dict[str, str]] = []
    if args.monitor_csv.exists():
        with open(args.monitor_csv) as f:
            samples = list(csv.DictReader(f))

    records: list[dict[str, Any]] = []
    for path in args.queries_jsonl:
        if path.exists():
            records.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())

    out = [f"## {args.title}", ""]
    if samples:
        total = samples[0]["mem_total_gib"]
        overall = max(samples, key=lambda row: float(row["mem_used_gib"]))
        swap = max(float(row["swap_used_gib"]) for row in samples)
        out += [
            f"System memory **{total} GiB**; peak used **{overall['mem_used_gib']} GiB** during "
            f"`{overall['phase']}` (largest process `{overall['top_comm']}` at {overall['top_rss_gib']} GiB); "
            f"peak swap {swap:.2f} GiB.",
            "",
        ]

    phase_peaks = peak_by_phase(samples)
    non_query = {phase: row for phase, row in phase_peaks.items() if phase.count(":") != 2}
    if non_query:
        out += ["| Phase | Peak used (GiB) | Largest process | Its RSS (GiB) |", "|---|---:|---|---:|"]
        for phase, row in non_query.items():
            out.append(f"| `{phase}` | {row['mem_used_gib']} | `{row['top_comm']}` | {row['top_rss_gib']} |")
        out.append("")

    if records:
        by_status: dict[str, int] = defaultdict(int)
        for record in records:
            by_status[record["status"]] += 1
        out += [
            "Queries: " + ", ".join(f"{count} {status}" for status, count in sorted(by_status.items())),
            "",
            "| Engine | Format | Query | Status | Wall (s) | Peak RSS (GiB) | Peak system used (GiB) |",
            "|---|---|---:|---|---:|---:|---:|",
        ]
        for record in records:
            phase = f"{record['engine']}:{record['format']}:q{record['query']:02d}"
            system_peak = phase_peaks.get(phase, {}).get("mem_used_gib", "")
            status = record["status"] if record["status"] == "ok" else f"**{record['status']}**"
            out.append(
                f"| {record['engine']} | {record['format']} | {record['query']} | {status} | "
                f"{record.get('wall_s', '')} | {record.get('peak_rss_gib', '')} | {system_peak} |"
            )
        out.append("")

    print("\n".join(out))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    monitor = sub.add_parser("monitor", help="Sample memory into a CSV until terminated")
    monitor.add_argument("--output", type=Path, required=True)
    monitor.add_argument("--phase-file", type=Path, required=True)
    monitor.add_argument("--interval", type=float, default=2.0)

    run = sub.add_parser("run-queries", help="Run each query in its own process and record its memory")
    run.add_argument("--binary", required=True)
    run.add_argument("--engine", required=True, choices=["datafusion", "duckdb"])
    run.add_argument("--variant", help="Label for this configuration, e.g. spill")
    run.add_argument("--benchmark", default="tpch")
    run.add_argument("--formats", nargs="+", required=True)
    run.add_argument("--queries", type=int, nargs="*")
    run.add_argument("--iterations", type=int, default=1)
    run.add_argument("--timeout", type=int, default=3600, help="Per-query timeout in seconds")
    run.add_argument(
        "--min-available-gib",
        type=float,
        default=4.0,
        help="Kill a query when system available memory falls below this",
    )
    run.add_argument("--budget", type=int, help="Skip the remaining queries after this many seconds")
    run.add_argument("--opt", action="append", default=[], help="Benchmark option, e.g. scale-factor=1000.0")
    run.add_argument("--out-dir", type=Path, required=True)
    run.add_argument("--phase-file", type=Path, required=True)
    run.add_argument(
        "--runner",
        # Matches vx-bench: label records with the EC2 instance runs-on actually provisioned.
        default=f"ec2_{os.environ['RUNS_ON_INSTANCE_TYPE']}" if "RUNS_ON_INSTANCE_TYPE" in os.environ else "local",
        help="Benchmark runner ID; the bench binaries refuse the default in CI",
    )
    run.add_argument("--fail-on-error", action="store_true")

    summarize = sub.add_parser("summarize", help="Render the monitor CSV and query records as Markdown")
    summarize.add_argument("--monitor-csv", type=Path, required=True)
    summarize.add_argument("--queries-jsonl", type=Path, nargs="*", default=[])
    summarize.add_argument("--title", default="Memory usage")

    args = parser.parse_args()
    if args.command == "monitor":
        cmd_monitor(args)
        return 0
    if args.command == "run-queries":
        return cmd_run_queries(args)
    cmd_summarize(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
