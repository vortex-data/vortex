# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Run sequential, paired Vortex/Lance measurements against the same simulated link."""

import argparse
import hashlib
import json
import os
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path


def measure(args: argparse.Namespace, variant: str, pattern: str, repeat: int) -> dict[str, object]:
    name = f"{repeat}-{pattern}-{variant}"
    output = args.output / name
    trace = output.with_suffix(".requests.jsonl")
    simulator = args.output / "s3_simulator.py"
    server_command = [
        sys.executable,
        str(simulator),
        "--root",
        str(args.data),
        "--port",
        str(args.port),
        "--latency-ms",
        str(args.latency_ms),
        "--mib-per-second",
        str(args.mib_per_second),
        "--concurrency",
        str(args.concurrency),
        "--trace",
        str(trace),
    ]
    env = os.environ.copy()
    settings = dict(
        AWS_ACCESS_KEY_ID="simulated",
        AWS_SECRET_ACCESS_KEY="simulated",
        AWS_DEFAULT_REGION="us-east-1",
        AWS_REGION="us-east-1",
        AWS_ENDPOINT=f"http://127.0.0.1:{args.port}",
        AWS_ALLOW_HTTP="true",
        AWS_ENDPOINT_URL=f"http://127.0.0.1:{args.port}",
        AWS_CONFIG_FILE="/dev/null",
        AWS_SHARED_CREDENTIALS_FILE="/dev/null",
        AWS_EC2_METADATA_DISABLED="true",
        VORTEX_BENCH_MATERIALIZE="arrow",
        VORTEX_BENCH_SIZE_THEN_READ="1" if variant == "baseline" else "0",
        VORTEX_BENCH_SCAN="v2",
        VORTEX_BENCH_SUBSEGMENTS="0" if variant == "baseline" else "1",
        VORTEX_BENCH_VERIFY="0",
    )
    env.pop("AWS_SESSION_TOKEN", None)
    env.pop("AWS_PROFILE", None)
    env.update(settings)
    command = [
        str(args.binary.resolve()),
        "--remote-data-dir",
        "s3://bench/",
        "--datasets",
        args.dataset,
        "--formats",
        "lance" if variant == "lance" else "vortex",
        "--patterns",
        pattern,
        "--open-mode",
        args.open_mode,
        "--time-limit",
        str(args.time_limit),
        "--display-format",
        "gh-json",
        "-o",
        str(output.with_suffix(".jsonl")),
        "--ingest-jsonl",
        str(output.with_suffix(".ingest.jsonl")),
    ]
    print(f"Starting {name}: {args.time_limit}s, {args.open_mode}, Arrow materialization", flush=True)
    with output.with_suffix(".server.log").open("w") as server_log:
        server = subprocess.Popen(server_command, stdout=server_log, stderr=subprocess.STDOUT)
        try:
            for _ in range(100):
                if server.poll() is not None:
                    raise RuntimeError(f"Simulator exited; see {output}.server.log")
                try:
                    with socket.create_connection(("127.0.0.1", args.port), timeout=0.1):
                        break
                except OSError:
                    time.sleep(0.05)
            else:
                raise RuntimeError("Simulator did not start")
            with output.with_suffix(".log").open("w") as log:
                subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        finally:
            server.terminate()
            server.wait(timeout=10)
    records = [json.loads(line) for line in output.with_suffix(".ingest.jsonl").read_text().splitlines()]
    record = next(record for record in records if "all_runtimes_ns" in record)
    runtimes = record["all_runtimes_ns"]
    result = dict(
        name=name,
        variant=variant,
        pattern=pattern,
        repeat=repeat,
        median_ms=statistics.median(runtimes) / 1e6,
        iterations=len(runtimes),
        command=command,
        server_command=server_command,
        environment=settings,
    )
    output.with_suffix(".run.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"{name}: {result['median_ms']:.3f} ms ({len(runtimes)} iterations)", flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path("target/release_debug/random-access-bench"))
    parser.add_argument("--data", type=Path, default=Path("vortex-bench/data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", default="nested-structs")
    parser.add_argument("--open-mode", choices=["cached", "reopen"], default="reopen")
    parser.add_argument("--time-limit", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--latency-ms", type=float, default=20)
    parser.add_argument("--mib-per-second", type=float, default=100)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--port", type=int, default=9010)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "s3_simulator.py").write_bytes(Path(__file__).with_name("s3_simulator.py").read_bytes())
    with args.binary.open("rb") as binary:
        digest = hashlib.sha256()
        for chunk in iter(lambda: binary.read(1024 * 1024), b""):
            digest.update(chunk)
    (args.output / "binary.sha256").write_text(digest.hexdigest() + "\n")
    results = []
    for repeat in range(args.repeats):
        variants = ["baseline", "candidate", "lance"]
        if repeat % 2:
            variants.reverse()
        for pattern in ["correlated", "uniform"]:
            for variant in variants:
                results.append(measure(args, variant, pattern, repeat))
                (args.output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
