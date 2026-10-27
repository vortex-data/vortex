# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Run initial and request-calibrated pairs sequentially on the same CI runner."""

import argparse
import json
import os
import subprocess
from pathlib import Path

import aws_credentials
import compare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--scale-factor", default="10")
    parser.add_argument("--format", default="vortex", choices=("vortex", "parquet"))
    parser.add_argument("--output", type=Path, default=Path("s3-validation-results"))
    parser.add_argument("--binary", default="target/release_debug/s3-validate")
    parser.add_argument("--phase", choices=("initial", "tuned", "all"), default="all")
    parser.add_argument("--suites", choices=("tpch", "tpcds"), nargs="+", default=["tpch", "tpcds"])
    args = parser.parse_args()
    if args.iterations < 3:
        parser.error("CI validation needs at least three iterations")
    if args.phase == "all" and set(args.suites) != {"tpch", "tpcds"}:
        parser.error("A single suite requires --phase initial or --phase tuned")
    with aws_credentials.environment(os.environ.get("S3_VALIDATION_ROLE_ARN")) as env:
        run(args, env)


def run(args: argparse.Namespace, env: dict[str, str]):
    args.output.mkdir(parents=True, exist_ok=True)
    model = {"ttfb_ms": 25.0, "request_mib_per_sec": 100.0, "max_inflight": 0}
    phases = ("initial", "tuned") if args.phase == "all" else (args.phase,)
    if args.phase == "tuned":
        model = json.loads((args.output / "parameters.json").read_text())
    for phase in phases:
        for suite in args.suites:
            output = args.output / f"{phase}-{suite}.jsonl"
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
            print("Running", command, flush=True)
            with (args.output / f"{phase}-{suite}.log").open("w") as log:
                with subprocess.Popen(command, stderr=subprocess.PIPE, text=True, env=env) as process:
                    if process.stderr is None:
                        raise RuntimeError("Missing validator log stream")
                    for line in process.stderr:
                        log.write(line)
                        print(line, end="", flush=True)
                    if process.wait():
                        raise subprocess.CalledProcessError(process.returncode, command)
        paths = [args.output / f"{phase}-{suite}.jsonl" for suite in ("tpch", "tpcds")]
        present = [path for path in paths if path.exists()]
        records = compare.read(present)
        result = compare.compare(records, args.iterations, ["tpch", "tpcds"])
        compare.write_reports(result, args.output / f"{phase}.md")
        print(phase, json.dumps({"adequate": result["adequate"], "correlations": result["correlations"]}), flush=True)
        if phase == "initial":
            # A request mismatch invalidates latency tuning; preserve the evidence and stop.
            measured = {r["suite"] for r in records if r["kind"] == "environment"}
            if any(not row["requests_match"] for row in result["rows"] if row["suite"] in measured):
                raise RuntimeError("Initial request/result mismatch: investigate before tuning latency")
            if len(present) == len(paths):
                model = compare.calibrate(records)
                (args.output / "parameters.json").write_text(json.dumps(model, indent=2) + "\n")


if __name__ == "__main__":
    main()
