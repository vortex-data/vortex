#!/usr/bin/env python3
"""Run paired whole-file vx conversions strictly one process at a time."""

import argparse
import glob
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--dataset", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--dataset-glob", action="append", default=[], metavar="PREFIX=GLOB")
    parser.add_argument(
        "--uncompressed", action="append", default=[], metavar="NAME=ALL:STRING"
    )
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--cpu", type=int, default=2)
    return parser.parse_args()


def parse_datasets(values):
    datasets = []
    for value in values:
        if "=" not in value:
            raise SystemExit(f"dataset must be NAME=PATH: {value}")
        name, raw_path = value.split("=", 1)
        path = Path(raw_path).resolve()
        if not name or not path.is_file():
            raise SystemExit(f"invalid dataset: {value}")
        datasets.append((name, path))
    return datasets


def expand_dataset_globs(values):
    datasets = []
    for value in values:
        if "=" not in value:
            raise SystemExit(f"dataset glob must be PREFIX=GLOB: {value}")
        prefix, pattern = value.split("=", 1)
        paths = sorted((Path(path).resolve() for path in glob.glob(pattern)), key=lambda p: p.name)
        if not prefix or not paths:
            raise SystemExit(f"dataset glob matched nothing: {value}")
        datasets.extend((f"{prefix}-{path.stem}", path) for path in paths)
    return datasets


def parse_uncompressed(values):
    sizes = {}
    for value in values:
        try:
            name, raw_sizes = value.split("=", 1)
            all_bytes, string_bytes = (int(size) for size in raw_sizes.split(":", 1))
        except ValueError as error:
            raise SystemExit(f"uncompressed must be NAME=ALL:STRING: {value}") from error
        sizes[name] = (all_bytes, string_bytes)
    return sizes


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def run(binary, parquet, cpu, hash_output=False):
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = time.perf_counter_ns()
    completed = subprocess.run(
        ["taskset", "-c", str(cpu), str(binary), "convert", "-q", str(parquet)],
        text=True,
        capture_output=True,
        env={**os.environ, "RAYON_NUM_THREADS": "1"},
    )
    elapsed_ns = time.perf_counter_ns() - started
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    if completed.returncode:
        raise RuntimeError(
            f"{binary} failed ({completed.returncode}): {completed.stderr.strip()}"
        )
    vortex = parquet.with_suffix(".vortex")
    metrics = {
        "wall_s": elapsed_ns / 1e9,
        "user_s": after.ru_utime - before.ru_utime,
        "system_s": after.ru_stime - before.ru_stime,
        "major_faults": after.ru_majflt - before.ru_majflt,
        "minor_faults": after.ru_minflt - before.ru_minflt,
        "voluntary_switches": after.ru_nvcsw - before.ru_nvcsw,
        "involuntary_switches": after.ru_nivcsw - before.ru_nivcsw,
        "vortex_bytes": vortex.stat().st_size,
    }
    if hash_output:
        metrics["vortex_sha256"] = sha256(vortex)
    return metrics


def write_result(
    output, phase, repetition, dataset, variant, source_bytes, uncompressed, metrics
):
    record = {
        "type": "result",
        "phase": phase,
        "repetition": repetition,
        "dataset": dataset,
        "variant": variant,
        "parquet_bytes": source_bytes,
        "parquet_gbps": source_bytes / metrics["wall_s"] / 1e9,
        "cpu_gbps": source_bytes / metrics["user_s"] / 1e9,
        **metrics,
    }
    if uncompressed is not None:
        all_bytes, string_bytes = uncompressed
        record["parquet_uncompressed_bytes"] = all_bytes
        record["parquet_uncompressed_gbps"] = all_bytes / metrics["wall_s"] / 1e9
        record["string_uncompressed_bytes"] = string_bytes
        record["string_uncompressed_gbps"] = string_bytes / metrics["wall_s"] / 1e9
    output.write(json.dumps(record, sort_keys=True) + "\n")
    output.flush()


def main():
    args = parse_args()
    datasets = parse_datasets(args.dataset) + expand_dataset_globs(args.dataset_glob)
    if not datasets:
        raise SystemExit("at least one --dataset or --dataset-glob is required")
    uncompressed = parse_uncompressed(args.uncompressed)
    unknown_sizes = set(uncompressed).difference(name for name, _ in datasets)
    if unknown_sizes:
        raise SystemExit(f"uncompressed sizes for unknown datasets: {sorted(unknown_sizes)}")
    binaries = {
        "develop": args.baseline.resolve(),
        "snapshot-greedy": args.candidate.resolve(),
    }
    for name, binary in binaries.items():
        if not binary.is_file():
            raise SystemExit(f"missing {name} binary: {binary}")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")

    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    local_inputs = {}
    for name, source in datasets:
        local = args.work_dir / f"{name}.parquet"
        if local.is_symlink() and local.resolve() == source:
            pass
        elif local.exists() or local.is_symlink():
            raise SystemExit(f"refusing to replace {local}")
        else:
            local.symlink_to(source)
        local_inputs[name] = local

    metadata = {
        "type": "run_metadata",
        "schema_version": 1,
        "started_unix": time.time(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "cpu_affinity": args.cpu,
        "rayon_threads": 1,
        "warmups": args.warmups,
        "iterations": args.iterations,
        "order": "serial paired AB/BA sweeps with rotating dataset order",
        "binaries": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in binaries.items()
        },
        "datasets": {
            name: {
                "path": str(source),
                "parquet_bytes": source.stat().st_size,
                "sha256": sha256(source),
                "parquet_uncompressed_bytes": uncompressed.get(name, (None, None))[0],
                "string_uncompressed_bytes": uncompressed.get(name, (None, None))[1],
            }
            for name, source in datasets
        },
        "command": sys.argv,
    }

    with args.output.open("x", encoding="utf-8") as output:
        output.write(json.dumps(metadata, sort_keys=True) + "\n")
        output.flush()

        for warmup in range(args.warmups):
            for dataset, source in datasets:
                for variant in binaries:
                    print(f"warmup {dataset} rep={warmup} variant={variant}", flush=True)
                    metrics = run(
                        binaries[variant], local_inputs[dataset], args.cpu, hash_output=True
                    )
                    write_result(
                        output,
                        "warmup",
                        warmup,
                        dataset,
                        variant,
                        source.stat().st_size,
                        uncompressed.get(dataset),
                        metrics,
                    )

        for repetition in range(args.iterations):
            offset = repetition % len(datasets)
            sweep = datasets[offset:] + datasets[:offset]
            if repetition % 2:
                sweep.reverse()
            variants = list(binaries)
            if repetition % 2:
                variants.reverse()
            for cell_index, (dataset, source) in enumerate(sweep):
                cell_variants = variants if cell_index % 2 == 0 else list(reversed(variants))
                for variant in cell_variants:
                    print(
                        f"measurement {dataset} rep={repetition} variant={variant}",
                        flush=True,
                    )
                    metrics = run(binaries[variant], local_inputs[dataset], args.cpu)
                    write_result(
                        output,
                        "measurement",
                        repetition,
                        dataset,
                        variant,
                        source.stat().st_size,
                        uncompressed.get(dataset),
                        metrics,
                    )


if __name__ == "__main__":
    main()
