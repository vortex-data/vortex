# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Learn a reusable RAM-store profile from real S3 traces across CI runs."""

import argparse
import hashlib
import json
import math
import statistics
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import compare


def learn(
    paths: Sequence[Path], minimum_runs: int = 2, preserve_tail: bool = False, exclude_startup: bool = False
) -> compare.Record:
    records = []
    sources = []
    hashes = set()
    hosts = set()
    memories = []
    seen_samples = set()
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in hashes:
            raise ValueError(f"Duplicate training input: {path}")
        hashes.add(digest)
        entries = compare.read([path])
        environments = [r for r in entries if r.get("kind") == "environment"]
        if len(environments) != 1:
            raise ValueError(f"Expected one environment record in {path}")
        environment = environments[0]
        if exclude_startup and not environment.get("source_warmup_seconds", 0):
            seen_processes = set()
            for record in entries:
                if record.get("kind") != "sample" or record["backend"] not in ("real", "real_repeat"):
                    continue
                process = record.get("process_started_unix_ms", environment.get("process_started_unix_ms"))
                if process not in seen_processes:
                    record["_process_startup"] = True
                    seen_processes.add(process)
        for record in entries:
            if record.get("kind") == "sample":
                record["_training_run_id"] = str(environment.get("ci_run_id"))
                key = (*compare.sample_key(record), record["backend"])
                if key in seen_samples:
                    raise ValueError(f"Duplicate training sample in {path}: {key}")
                seen_samples.add(key)
        required = ("ci_run_id", "sha", "region", "instance_id", "cpus", "total_memory", "format")
        if (
            any(not environment.get(key) for key in required)
            or environment.get("local_smoke")
            or not environment.get("source", "").startswith("s3://")
        ):
            raise ValueError(f"Training input must come from real S3 in CI: {path}")
        hosts.add(tuple(environment[key] for key in ("region", "cpus", "format")))
        memories.append(environment["total_memory"])
        samples = [r for r in entries if r.get("kind") == "sample" and r["backend"] in ("real", "real_repeat")]
        if not samples or any(r["error"] or r["counts"]["errors"] for r in samples):
            raise ValueError(f"Missing or failed real S3 samples in {path}")
        requests = [q for r in samples for q in r["requests"] if q["operation"] == "GET"]
        for request in requests:
            values = (request["duration_ms"], request["first_byte_ms"])
            if request["bytes"] <= 0 or any(v is None or not math.isfinite(v) or v < 0 for v in values):
                raise ValueError(f"Incomplete GET measurement in {path}")
        sources.append(
            {
                "ci_run_id": str(environment["ci_run_id"]),
                "url": f"https://github.com/vortex-data/vortex/actions/runs/{environment['ci_run_id']}",
                "sha": environment["sha"],
                "instance_id": environment["instance_id"],
                "source": environment["source"],
                "input": path.name,
                "sha256": digest,
                "suite": environment["suite"],
                "scale_factor": environment["scale_factor"],
                "adaptive_io": environment.get("adaptive_io"),
                "tail_cache_bytes": environment.get("tail_cache_bytes"),
                "queries": sorted({r["query"] for r in samples}),
                "samples": len(samples),
                "real_backends": sorted({r["backend"] for r in samples}),
                "excluded_startup_samples": sum(r.get("_process_startup", False) for r in samples),
                "get_requests": len(requests),
            }
        )
        records.extend(entries)
    if len(hosts) != 1 or max(memories, default=0) > min(memories, default=0) * 1.01:
        raise ValueError("Train separate profiles for different regions, formats or runner hardware")
    run_ids = sorted({s["ci_run_id"] for s in sources})
    if len(run_ids) < minimum_runs:
        raise ValueError(f"Need {minimum_runs} distinct CI runs; found {len(run_ids)}")
    parameters = compare.calibrate(records, preserve_tail=preserve_tail)
    region, cpus, format_name = next(iter(hosts))
    parameters["profile_version"] = 3 if preserve_tail else 2
    parameters["training"] = {
        "created_at": datetime.now(UTC).isoformat(),
        "run_ids": run_ids,
        "region": region,
        "cpus": cpus,
        "total_memory": statistics.median(memories),
        "format": format_name,
        "weighting": "one observation per completed real GET; query runtimes are not training targets",
        "method": (
            "request-size-specific GET duration residuals; "
            + ("seeded weighted-CDF sampling; " if preserve_tail else "seeded 1000-quantile sampling; ")
            + "exclude samples with median model timer overshoot above 5 ms"
        )
        + ("; preserve each sparse-tail observation with its measured probability" if preserve_tail else ""),
        "scope": "request latency training; partial query coverage does not establish benchmark accuracy",
        "startup_policy": (
            "exclude first timed real sample per identifiable unprimed process; "
            "legacy inputs identify only their initial process"
        )
        if exclude_startup
        else "include startup",
        "sources": sources,
    }
    return parameters


def validate_profile(profile: compare.Record, run_id: str | None, region: str, format_name: str):
    training = profile.get("training", {})
    if profile.get("profile_version") not in (1, 2, 3) or len(set(training.get("run_ids", []))) < 2:
        raise ValueError("A frozen profile needs provenance from at least two distinct CI runs")
    if run_id is not None and str(run_id) in training["run_ids"]:
        raise ValueError("Validation must use a CI run excluded from training")
    format_name = {"vortex": "vortex-file-compressed"}.get(format_name, format_name)
    if training.get("region") != region or training.get("format") != format_name:
        raise ValueError("Profile region/format differs from the validation environment")


def attach_profile(result: compare.Record, profile: compare.Record, digest: str):
    for environment in result["environments"]:
        validate_profile(profile, environment.get("ci_run_id"), environment["region"], environment["format"])
        if any(
            environment["model"].get(key) != profile.get(key, [] if key == "latency_bins" else None)
            for key in environment["model"]
        ):
            raise ValueError("Measured model differs from the frozen learned profile")
        if (
            environment["cpus"] != profile["training"]["cpus"]
            or abs(environment["total_memory"] / profile["training"]["total_memory"] - 1) > 0.01
        ):
            raise ValueError("Validation runner differs from the training hardware")
    result["learned_profile"] = {"sha256": digest, "training": profile["training"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--minimum-runs", type=int, default=2)
    parser.add_argument("--preserve-tail", action="store_true")
    parser.add_argument("--exclude-startup", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    parameters = learn(
        args.inputs, args.minimum_runs, preserve_tail=args.preserve_tail, exclude_startup=args.exclude_startup
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(parameters, indent=2) + "\n")
    print(
        json.dumps(
            {
                "profile": str(args.output),
                "training_runs": parameters["training"]["run_ids"],
                "get_requests": parameters["calibration_get_requests"],
                "ttfb_ms": parameters["ttfb_ms"],
                "request_mib_per_sec": parameters["request_mib_per_sec"],
                "latency_quantiles_ms": parameters["latency_quantiles_ms"],
            }
        )
    )


if __name__ == "__main__":
    main()
