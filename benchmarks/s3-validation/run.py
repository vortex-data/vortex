# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Run initial and request-calibrated pairs sequentially on the same CI runner."""

import argparse
import json
import subprocess
from pathlib import Path

import compare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--scale-factor", default="10")
    parser.add_argument("--format", default="vortex", choices=("vortex", "parquet"))
    parser.add_argument("--output", type=Path, default=Path("s3-validation-results"))
    parser.add_argument("--binary", default="target/release_debug/s3-validate")
    parser.add_argument("--phase", choices=("initial", "tuned", "all"), default="all")
    args = parser.parse_args()
    if args.iterations < 3:
        parser.error("CI validation needs at least three iterations")
    args.output.mkdir(parents=True, exist_ok=True)
    model = {"ttfb_ms": 25.0, "request_mib_per_sec": 100.0, "max_inflight": 0}
    phases = ("initial", "tuned") if args.phase == "all" else (args.phase,)
    if args.phase == "tuned":
        model = json.loads((args.output / "parameters.json").read_text())
    for phase in phases:
        paths = []
        for suite in ("tpch", "tpcds"):
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
            print("Running", command, flush=True)
            with (args.output / f"{phase}-{suite}.log").open("w") as log:
                subprocess.run(command, stderr=log, check=True)
            paths.append(output)
        records = compare.read(paths)
        result = compare.compare(records, args.iterations, ["tpch", "tpcds"])
        (args.output / f"{phase}.json").write_text(json.dumps(result, indent=2) + "\n")
        (args.output / f"{phase}.md").write_text(compare.markdown(result))
        print(phase, json.dumps({"adequate": result["adequate"], "correlations": result["correlations"]}), flush=True)
        if phase == "initial":
            # A request mismatch invalidates latency tuning; preserve the evidence and stop.
            if any(not row["requests_match"] for row in result["rows"]):
                raise RuntimeError("Initial request/result mismatch: investigate before tuning latency")
            model = compare.calibrate(records)
            (args.output / "parameters.json").write_text(json.dumps(model, indent=2) + "\n")


if __name__ == "__main__":
    main()
