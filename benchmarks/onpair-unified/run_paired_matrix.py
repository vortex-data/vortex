#!/usr/bin/env python3
"""Run a balanced Rust/C++ OnPair matrix strictly one process at a time."""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


ALGORITHMS = ("snapshot-greedy", "cpp-boost")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--blocks", nargs="+", type=int, default=[2, 4, 8, 16, 32])
    parser.add_argument("--bits", type=int, default=12)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--cpu", type=int, default=2)
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def version(command):
    return subprocess.run(command, text=True, capture_output=True, check=True).stdout.strip()


def invoke(binary, algorithm, corpus, block, bits, cpu, warmups):
    command = [
        "taskset",
        "-c",
        str(cpu),
        str(binary),
        algorithm,
        str(corpus),
        str(block),
        str(bits),
        str(warmups),
        "1",
    ]
    completed = subprocess.run(
        command,
        text=True,
        capture_output=True,
        env={**os.environ, "RAYON_NUM_THREADS": "1"},
    )
    if completed.returncode:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"{completed.stderr.strip()}"
        )
    return json.loads(completed.stdout)


def main():
    args = parse_args()
    binary = args.binary.resolve()
    if not binary.is_file():
        raise SystemExit(f"missing binary: {binary}")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")

    cells = []
    corpora = {}
    for dataset in args.datasets:
        corpus = (args.corpus_dir / f"{dataset}.onpair").resolve()
        if not corpus.is_file():
            raise SystemExit(f"missing corpus: {corpus}")
        corpora[dataset] = corpus
        cells.extend((dataset, corpus, block) for block in args.blocks)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "type": "run_metadata",
        "schema_version": 1,
        "started_unix": time.time(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "rustc": version(["rustc", "-Vv"]),
        "gcc": version(["gcc", "--version"]).splitlines()[0],
        "g++": version(["g++", "--version"]).splitlines()[0],
        "binary": str(binary),
        "binary_sha256": sha256(binary),
        "cpu_affinity": args.cpu,
        "rayon_threads": 1,
        "warmups": args.warmups,
        "iterations": args.iterations,
        "algorithms": ALGORITHMS,
        "blocks_mib": args.blocks,
        "bits": args.bits,
        "order": "serial paired AB/BA sweeps with rotating cell order",
        "corpora": {
            dataset: {
                "path": str(corpus),
                "container_bytes": corpus.stat().st_size,
                "sha256": sha256(corpus),
            }
            for dataset, corpus in corpora.items()
        },
        "command": sys.argv,
    }

    with args.output.open("x", encoding="utf-8") as output:
        output.write(json.dumps(metadata, sort_keys=True) + "\n")
        output.flush()

        for dataset, corpus, block in cells:
            for algorithm in ALGORITHMS:
                print(f"warmup {dataset} block={block} algorithm={algorithm}", flush=True)
                record = invoke(
                    binary, algorithm, corpus, block, args.bits, args.cpu, args.warmups
                )
                record.update(type="result", phase="warmup", repetition=0)
                output.write(json.dumps(record, sort_keys=True) + "\n")
                output.flush()

        for repetition in range(args.iterations):
            offset = repetition % len(cells)
            sweep = cells[offset:] + cells[:offset]
            if repetition % 2:
                sweep.reverse()
            algorithms = list(ALGORITHMS)
            if repetition % 2:
                algorithms.reverse()
            for dataset, corpus, block in sweep:
                for algorithm in algorithms:
                    print(
                        f"measurement {dataset} block={block} rep={repetition} "
                        f"algorithm={algorithm}",
                        flush=True,
                    )
                    record = invoke(binary, algorithm, corpus, block, args.bits, args.cpu, 0)
                    record.update(type="result", phase="measurement", repetition=repetition)
                    output.write(json.dumps(record, sort_keys=True) + "\n")
                    output.flush()


if __name__ == "__main__":
    main()
