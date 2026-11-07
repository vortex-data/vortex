# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Learn size/concurrency-specific joint S3 response phases from real request observations."""

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import measure

SIZE_LIMITS = (4096, 65536, 262144, 1048576, 8388608, 2**64 - 1)
INFLIGHT_LIMITS = (1, 4, 16, 64, 2**64 - 1)


def limit(value: int, limits: tuple[int, ...]) -> int:
    return next(upper for upper in limits if value <= upper)


def compress(samples: list[tuple[int, int, int, int]], buckets: int) -> list[list[int]]:
    """Keep actual phase tuples, exact integer mass, and every observation in the sparse upper tail."""
    values = sorted(samples, key=lambda sample: (sum(sample[:3]), sample))
    tail = max(1, math.ceil(len(values) * 0.001))
    bulk = len(values) - tail
    count = min(buckets, bulk)
    result = []
    for index in range(count):
        start, end = bulk * index // count, bulk * (index + 1) // count
        result.append([*values[(start + end - 1) // 2], end - start])
    result.extend([*sample, 1] for sample in values[bulk:])
    return result


def normalise(request: dict) -> tuple[int, int, int, int]:
    timings = [request.get(key) for key in ("response_ms", "first_byte_ms", "duration_ms")]
    if any(t is None or not math.isfinite(t) or t < 0 for t in timings) or timings != sorted(timings):
        raise ValueError("A real GET has incomplete or inconsistent response phases")
    size = request["bytes"]
    if (
        type(size) is not int
        or size <= 0
        or type(request["inflight_at_start"]) is not int
        or not 1 <= request["inflight_at_start"] < 2**64
    ):
        raise ValueError("A real GET needs positive consumed bytes and in-flight count")
    header, first, end = timings
    return round(header * 1000), round((first - header) * 1000), round((end - first) * 1000), size


def load(path: Path) -> tuple[dict, list[tuple[tuple, dict]], int]:
    with path.open() as stream:
        environment = json.loads(next(stream))
    if (
        environment.get("kind") != "environment"
        or environment.get("local_smoke")
        or environment.get("endpoint_override")
        or not environment.get("source", "").startswith("s3://")
        or any(
            not environment.get(key) for key in ("ci_run_id", "sha", "region", "cpus", "total_memory", "instance_id")
        )
    ):
        raise ValueError("Training requires real S3 CI traces with hardware provenance")
    observations, excluded = [], 0
    if environment.get("schema_version") == 1:
        measure.read_window(path)
        with path.open() as stream:
            for line in stream:
                request = json.loads(line)
                if request.get("kind") != "request" or request["phase"] != "measure" or request["operation"] != "GET":
                    continue
                key = (request["round"], request["requested_concurrency"], request["bytes"], request["ordinal"])
                observations.append((key, request))
    else:
        seen_processes = set()
        # The earlier SQL investigation emits samples containing requests, with the same measured phases.
        with path.open() as stream:
            for line in stream:
                record = json.loads(line)
                if record.get("kind") != "sample" or record["backend"] not in ("real", "real_repeat"):
                    continue
                if record.get("error") or record["counts"]["errors"]:
                    raise ValueError("Failed S3 workload sample is ineligible for training")
                process = record.get("process_started_unix_ms", environment.get("process_started_unix_ms"))
                if not environment.get("source_warmup_seconds", 0) and process not in seen_processes:
                    seen_processes.add(process)
                    excluded += 1
                    continue
                key = (record["suite"], record["query"], record["iteration"], record["cache"], record["backend"])
                observations.extend(
                    ((*key, index), request)
                    for index, request in enumerate(record["requests"])
                    if request["operation"] == "GET"
                )
    if not observations:
        raise ValueError("No real GET observations in training input")
    return environment, observations, excluded


def learn(paths: list[Path], buckets: int = 256, minimum_samples: int = 32) -> dict:
    if buckets < 1 or minimum_samples < 1:
        raise ValueError("Bucket and minimum sample counts must be positive")
    sizes, loads = defaultdict(list), defaultdict(list)
    sources, hashes, seen, windows, memories, hardware = [], set(), set(), set(), [], set()
    headers = []
    for path in paths:
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest in hashes:
            raise ValueError("Duplicate training file")
        hashes.add(digest)
        environment, observations, excluded = load(path)
        window = (
            str(environment["ci_run_id"]),
            str(environment.get("ci_run_attempt", "unknown")),
            str(environment.get("measurement_id", environment.get("measurement_window_id", "run"))),
        )
        windows.add(window)
        hardware.add((environment["region"], environment["cpus"]))
        memories.append(environment["total_memory"])
        source = {
            key: environment.get(key)
            for key in (
                "ci_run_id",
                "ci_run_attempt",
                "sha",
                "measurement_id",
                "measurement_window_id",
                "instance_id",
                "instance_type",
                "availability_zone",
                "region",
                "cpus",
                "total_memory",
                "source",
                "started_unix_ms",
                "process_started_unix_ms",
            )
        }
        source.update(input=path.name, sha256=digest, real_gets=len(observations), excluded_startup_samples=excluded)
        sources.append(source)
        for identity, request in observations:
            key = (*window, *identity)
            if key in seen:
                raise ValueError("Duplicate real request identity")
            seen.add(key)
            sample = normalise(request)
            size = limit(sample[3], SIZE_LIMITS)
            concurrency = limit(request["inflight_at_start"], INFLIGHT_LIMITS)
            sizes[size].append(sample)
            loads[(size, concurrency)].append(sample)
            if sample[3] <= 65536 and request["inflight_at_start"] <= 4:
                headers.append(sample[0])
    if len(windows) < 2:
        raise ValueError("Use at least two distinct real S3 measurement windows")
    if len(hardware) != 1 or max(memories) > min(memories) * 1.01:
        raise ValueError("Train separate profiles for different region or runner hardware")
    bands = []
    for size, samples in sorted(sizes.items()):
        bands.append(
            {
                "max_bytes": size,
                "observations": len(samples),
                "samples": compress(samples, buckets),
                "concurrency": [
                    {
                        "max_inflight": upper,
                        "observations": len(loads[(size, upper)]),
                        "samples": compress(loads[(size, upper)], buckets),
                    }
                    for upper in INFLIGHT_LIMITS
                    if len(loads[(size, upper)]) >= minimum_samples
                ],
            }
        )
    region, cpus = next(iter(hardware))
    return {
        "profile_version": 1,
        "seed": 20261027,
        "metadata_header_us": round(statistics.median(headers or [s[0] for values in sizes.values() for s in values])),
        "bands": bands,
        "training": {
            "created_utc": datetime.now(UTC).isoformat(),
            "region": region,
            "cpus": cpus,
            "total_memory": round(statistics.median(memories)),
            "measurement_windows": [list(w) for w in sorted(windows)],
            "real_gets": sum(len(values) for values in sizes.values()),
            "sample_fields": ["headers_us", "first_byte_gap_us", "remaining_transfer_us", "source_bytes", "weight"],
            "method": (
                "joint observed response phases by request size and actual in-flight count; integer empirical mass; "
                "bulk buckets plus each sparse upper-tail observation; transfer cost scales with requested bytes"
            ),
            "weighting": "one observation per completed real GET; no query runtime targets",
            "scope": "request-derived opt-in model; matching marginal phases does not establish benchmark accuracy",
            "limitations": [
                "SDK timing includes client work and internal retries",
                "independent draws omit temporal correlation",
                "conditional transfer costs approximate shared bandwidth; no hard concurrency cap established",
                "loopback HTTP and timers add overhead; held-out benchmark validation required",
            ],
            "sources": sources,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--buckets", type=int, default=256)
    args = parser.parse_args()
    profile = learn(args.inputs, buckets=args.buckets)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Compact sample tuples keep the measured data reviewable without a line per scalar.
    args.output.write_text(json.dumps(profile, separators=(",", ":")) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "real_gets": profile["training"]["real_gets"],
                "windows": profile["training"]["measurement_windows"],
                "bands": len(profile["bands"]),
            }
        )
    )


if __name__ == "__main__":
    main()
