#!/usr/bin/env python3
"""Run the unified OnPair matrix strictly one process at a time."""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


ALGORITHMS = (
    "snapshot-greedy",
    "cpp-boost",
    "paper-rust16",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--blocks", nargs="+", type=int, default=[2, 4, 8, 16])
    parser.add_argument("--bits", nargs="+", type=int, default=[12, 16])
    parser.add_argument("--algorithms", nargs="+", choices=ALGORITHMS, default=ALGORITHMS)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")

    cells = []
    pair_index = 0
    for dataset in args.datasets:
        corpus = args.corpus_dir / f"{dataset}.onpair"
        if not corpus.is_file():
            raise SystemExit(f"missing corpus: {corpus}")
        for block in args.blocks:
            algorithms = list(args.algorithms)
            if pair_index % 2:
                algorithms.reverse()
            pair_index += 1
            for algorithm in algorithms:
                valid_bits = [16] if algorithm == "paper-rust16" else args.bits
                for bits in valid_bits:
                    cells.append((dataset, corpus, block, algorithm, bits))

    binary_sha256 = hashlib.sha256(args.binary.read_bytes()).hexdigest()
    rustc = subprocess.run(
        ["rustc", "-Vv"], text=True, capture_output=True, check=True
    ).stdout.strip()
    metadata = {
        "type": "run_metadata",
        "schema_version": 1,
        "started_unix": time.time(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "rustc": rustc,
        "binary": str(args.binary.resolve()),
        "binary_sha256": binary_sha256,
        "cpu_affinity": args.cpu,
        "warmups": args.warmups,
        "iterations": args.iterations,
        "cells": len(cells),
        "command": sys.argv,
    }
    prefix = ["taskset", "-c", str(args.cpu)] if args.cpu is not None else []
    env = os.environ.copy()
    env["RAYON_NUM_THREADS"] = "1"

    with args.output.open("x", encoding="utf-8") as output:
        output.write(json.dumps(metadata, sort_keys=True) + "\n")
        output.flush()
        for index, (dataset, corpus, block, algorithm, bits) in enumerate(cells, 1):
            command = prefix + [
                str(args.binary),
                algorithm,
                str(corpus),
                str(block),
                str(bits),
                str(args.warmups),
                str(args.iterations),
            ]
            print(
                f"[{index}/{len(cells)}] {dataset} block={block}MiB "
                f"algorithm={algorithm} bits={bits}",
                flush=True,
            )
            started = time.time()
            completed = subprocess.run(command, text=True, capture_output=True, env=env)
            if completed.returncode == 0:
                record = json.loads(completed.stdout)
                record["type"] = "result"
                record["cell_index"] = index
            else:
                record = {
                    "type": "error",
                    "cell_index": index,
                    "dataset": dataset,
                    "block_mib": block,
                    "algorithm": algorithm,
                    "bits": bits,
                    "returncode": completed.returncode,
                    "stderr": completed.stderr.strip(),
                    "stdout": completed.stdout.strip(),
                }
            record["wall_seconds"] = time.time() - started
            output.write(json.dumps(record, sort_keys=True) + "\n")
            output.flush()
            if completed.returncode != 0 and not args.continue_on_error:
                raise SystemExit(f"cell {index} failed: {completed.stderr.strip()}")


if __name__ == "__main__":
    main()
