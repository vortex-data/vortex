# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Run initial and request-calibrated pairs sequentially on the same CI runner."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import aws_credentials
import compare
import learn


def append_chunk(chunk: Path, destination: Path):
    keys = (
        "ci_run_id",
        "sha",
        "region",
        "instance_id",
        "cpus",
        "total_memory",
        "source",
        "suite",
        "scale_factor",
        "format",
        "model",
        "iterations",
        "iteration_overrides",
        "adaptive_io",
        "tail_cache_bytes",
        "objects",
    )
    with chunk.open() as incoming:
        header_line = next(incoming)
        header = json.loads(header_line)
        if header.get("kind") != "environment":
            raise ValueError("Query chunk has no environment record")
        if destination.exists():
            with destination.open() as previous:
                original = json.loads(next(previous))
            if any(original.get(key) != header.get(key) for key in keys):
                raise ValueError("Query chunks differ in hardware, data, profile or configuration")
        temporary = destination.with_suffix(".merge.tmp")
        with temporary.open("w") as output:
            if destination.exists():
                with destination.open() as previous:
                    shutil.copyfileobj(previous, output)
            else:
                output.write(header_line)
            shutil.copyfileobj(incoming, output)
        temporary.replace(destination)
    chunk.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument(
        "--iteration-overrides", type=json.loads, default=os.environ.get("S3_ITERATION_OVERRIDES", "{}")
    )
    parser.add_argument("--scale-factor", default="10")
    parser.add_argument("--format", default="vortex", choices=("vortex", "parquet"))
    parser.add_argument("--output", type=Path, default=Path("s3-validation-results"))
    parser.add_argument("--binary", default="target/release_debug/s3-validate")
    parser.add_argument("--phase", choices=("initial", "tuned", "all", "validate"), default="all")
    parser.add_argument("--profile", type=Path, default=os.environ.get("S3_MODEL_PROFILE"))
    parser.add_argument("--suites", choices=("tpch", "tpcds"), nargs="+", default=["tpch", "tpcds"])
    parser.add_argument("--queries", type=lambda value: [int(part) for part in value.split(",")])
    args = parser.parse_args()
    if args.iterations < 3:
        parser.error("CI validation needs at least three iterations")
    if any(n < 3 for queries in args.iteration_overrides.values() for n in queries.values()):
        parser.error("CI iteration overrides must retain at least three samples")
    if args.phase == "all" and set(args.suites) != {"tpch", "tpcds"}:
        parser.error("A single suite requires --phase initial or --phase tuned")
    if args.phase == "validate" and args.profile is None:
        parser.error("--phase validate needs a frozen --profile")
    if args.queries and (len(args.suites) != 1 or any(q < 1 for q in args.queries)):
        parser.error("Query selection needs one suite and positive query IDs")
    with aws_credentials.environment(os.environ.get("S3_VALIDATION_ROLE_ARN")) as env:
        run(args, env)


def run(args: argparse.Namespace, env: dict[str, str]):
    args.output.mkdir(parents=True, exist_ok=True)
    model = {"ttfb_ms": 25.0, "request_mib_per_sec": 100.0, "max_inflight": 0}
    frozen = None
    if args.phase == "validate":
        frozen = json.loads(args.profile.read_text())
        learn.validate_profile(frozen, env.get("GITHUB_RUN_ID"), env.get("AWS_REGION", ""), args.format)
        if args.profile.resolve() != (args.output / "parameters.json").resolve():
            shutil.copyfile(args.profile, args.output / "parameters.json")
        model = frozen
    phases = ("initial", "tuned") if args.phase == "all" else ("tuned" if frozen else args.phase,)
    if args.phase == "tuned":
        model = json.loads((args.output / "parameters.json").read_text())
    for phase in phases:
        for suite in args.suites:
            canonical = args.output / f"{phase}-{suite}.jsonl"
            output = (
                canonical
                if not args.queries
                else args.output / f"{phase}-{suite}-chunk-{min(args.queries):03}-{max(args.queries):03}.jsonl"
            )
            command = [
                args.binary,
                suite,
                "--source",
                f"{args.source_root.rstrip('/')}/{suite}/",
                "--format",
                args.format,
                "--scale-factor",
                args.scale_factor,
                "--iterations",
                str(args.iterations),
                "--output",
                str(output),
                "--ttfb-ms",
                str(model["ttfb_ms"]),
                "--request-mib-per-sec",
                str(model["request_mib_per_sec"]),
                "--max-inflight",
                str(model["max_inflight"]),
            ]
            if phase == "tuned":
                command.extend(("--model-parameters", str(args.output / "parameters.json")))
            if args.queries:
                command.extend(("--queries", ",".join(str(q) for q in args.queries)))
            if overrides := args.iteration_overrides.get(suite):
                command.extend(("--iteration-overrides", ",".join(f"{q}={n}" for q, n in overrides.items())))
            print("Running", command, flush=True)
            with output.with_suffix(".log").open("w") as log:
                log.write("command: " + json.dumps(command) + "\n")
                with subprocess.Popen(command, stderr=subprocess.PIPE, text=True, env=env) as process:
                    if process.stderr is None:
                        raise RuntimeError("Missing validator log stream")
                    for line in process.stderr:
                        log.write(line)
                        print(line, end="", flush=True)
                    if process.wait():
                        raise subprocess.CalledProcessError(process.returncode, command)
            if args.queries:
                append_chunk(output, canonical)
        paths = [args.output / f"{phase}-{suite}.jsonl" for suite in ("tpch", "tpcds")]
        present = [path for path in paths if path.exists()]
        records = compare.read(present)
        result = compare.compare(records, args.iterations, ["tpch", "tpcds"])
        if frozen:
            digest = hashlib.sha256(args.profile.read_bytes()).hexdigest()
            learn.attach_profile(result, frozen, digest)
        compare.write_reports(result, args.output / f"{phase}.md")
        print(phase, json.dumps({"adequate": result["adequate"], "correlations": result["correlations"]}), flush=True)
        if phase == "initial":
            # A request mismatch invalidates latency tuning; preserve the evidence and stop.
            measured = {(r["suite"], r["query"]) for r in records if r["kind"] == "sample"}
            if any(not row["requests_match"] for row in result["rows"] if (row["suite"], row["query"]) in measured):
                raise RuntimeError("Initial request/result mismatch: investigate before tuning latency")
            if len(present) == len(paths) and all(row["requests_match"] for row in result["rows"]):
                model = compare.calibrate(records)
                (args.output / "parameters.json").write_text(json.dumps(model, indent=2) + "\n")


if __name__ == "__main__":
    main()
