# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Run native S3 benchmark tasks using credentials supplied by the workflow."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from functools import partial
from pathlib import Path

import learn

# Keep authenticated tasks within the standard two-hour workflow credential session.
TASK_TIMEOUT_SECONDS = 6900


def execute(cmd: list[str], log: Path):
    print(json.dumps({"command": cmd, "log": str(log)}), flush=True)
    with log.open("w") as output:
        subprocess.run(cmd, stdout=output, stderr=subprocess.STDOUT, check=True, timeout=TASK_TIMEOUT_SECONDS)


def simulate(args: argparse.Namespace, kind: str, name: str, cmd: list[str], local: Path | None = None) -> list[str]:
    command = [
        sys.executable,
        "benchmarks/s3/run.py",
        "--server",
        str(args.binaries / "s3-simulate"),
        "--profile",
        str(args.output / "profile.json"),
        "--kind",
        kind,
        "--timeout",
        str(TASK_TIMEOUT_SECONDS),
        "--output",
        str(args.output / f"{name}-manifest.json"),
    ]
    if local is not None:
        command.extend(["--data-root", str(local)])
    return [*command, "--", *cmd]


def freeze_profile(args: argparse.Namespace, learned: dict) -> Path:
    candidate = args.output / "learned-profile.json"
    candidate.write_text(json.dumps(learned, separators=(",", ":")) + "\n")
    selected = args.profile if args.model == "pinned" else candidate
    profile = args.output / "profile.json"
    profile.write_bytes(selected.read_bytes())
    (args.output / "model-selection.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "source": str(selected),
                "profile_sha256": hashlib.sha256(profile.read_bytes()).hexdigest(),
                "candidate_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
                "promoted": False,
            },
            indent=2,
        )
        + "\n"
    )
    return profile


def calibrate(args: argparse.Namespace):
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    run_id = os.environ["GITHUB_RUN_ID"]
    probe_args = [
        "--sizes",
        "4096,65536,262144,1048576,8388608",
        "--concurrency",
        "1,4,16,64",
        "--requests",
        "64",
        "--rounds",
        "3",
        "--seed",
        "20261028",
        "--warmup-seconds",
        "20",
        "--timeout-seconds",
        "60",
    ]
    for window in (1, 2):
        execute(
            [
                str(args.binaries / "s3-measure"),
                "--source",
                args.remote_root + "/probe/",
                "--output",
                str(root / f"training-{window}.jsonl"),
                "--measurement-id",
                f"{run_id}-calibration-{window}",
                *probe_args,
            ],
            root / f"training-{window}.log",
        )
    model = learn.learn([root / "training-1.jsonl", root / "training-2.jsonl"])
    profile = freeze_profile(args, model)
    fixtures = root / "probe-fixtures"
    fixtures.mkdir()
    for index in range(1, 5):
        shutil.copyfile("s3-probe.bin", fixtures / f"probe-{index}.bin")
    execute(
        [
            str(args.binaries / "s3-measure"),
            "--source",
            args.remote_root + "/probe/",
            "--output",
            str(root / "heldout-real.jsonl"),
            "--measurement-id",
            f"{run_id}-heldout-real",
            *probe_args,
        ],
        root / "heldout-real.log",
    )
    for label, config in (("simulated", ["--profile", str(profile)]), ("zero", ["--zero-delay"])):
        execute(
            [
                sys.executable,
                "benchmarks/s3/run.py",
                "--server",
                str(args.binaries / "s3-simulate"),
                *config,
                "--kind",
                "probe",
                "--data-root",
                str(fixtures),
                "--output",
                str(root / f"probe-{label}-manifest.json"),
                "--",
                str(args.binaries / "s3-measure"),
                "--output",
                str(root / f"probe-{label}.jsonl"),
                "--measurement-id",
                f"{run_id}-{label}",
                *probe_args,
            ],
            root / f"probe-{label}.log",
        )


def prepare_sql(args: argparse.Namespace, suite: str):
    execute(
        [str(args.binaries / "data-gen"), suite, "--formats", "parquet,vortex", "--opt", f"scale-factor={args.scale}"],
        args.output / f"prepare-{suite}.log",
    )
    if "lance" in args.engines.split(","):
        # A local first query prepares Lance through its ordinary conversion path.
        execute(
            [
                str(args.binaries / "lance-bench"),
                suite,
                "--queries",
                "1",
                "--iterations",
                "1",
                "--opt",
                f"scale-factor={args.scale}",
                "--runner",
                "ec2_c8gd.16xlarge",
            ],
            args.output / f"prepare-{suite}-lance.log",
        )


def upload_sql(args: argparse.Namespace, suite: str):
    local = Path("vortex-bench/data") / suite / args.scale
    execute(
        ["aws", "s3", "cp", "--recursive", str(local), f"{args.remote_root}/{suite}/"],
        args.output / f"upload-{suite}.log",
    )


def sql(args: argparse.Namespace, suite: str, engine: str, backend: str):
    name = f"{suite}-{engine}-{backend}"
    cmd = [
        str(args.binaries / f"{engine}-bench"),
        suite,
        "--iterations",
        str(args.iterations),
        "--display-format",
        "gh-json",
        "--hide-progress-bar",
        "--runner",
        "ec2_c8gd.16xlarge",
        "--opt",
        f"scale-factor={args.scale}",
        "--ingest-jsonl",
        str(args.output / f"{name}.jsonl"),
    ]
    if engine != "lance":
        cmd.extend(["--formats", "parquet,vortex"])
    if engine == "duckdb":
        cmd.append("--delete-duckdb-database")
    if backend == "simulated":
        cmd = simulate(args, "sql", name, cmd, Path("vortex-bench/data") / suite / args.scale)
    else:
        cmd.extend(["--opt", f"remote-data-dir={args.remote_root}/{suite}/"])
    execute(cmd, args.output / f"{name}.log")


def prepare_random(args: argparse.Namespace):
    execute(
        [
            str(args.binaries / "random-access-bench"),
            "--prepare-data",
            "--datasets",
            "nested-lists",
            "--formats",
            "parquet,vortex,lance",
        ],
        args.output / "prepare-random.log",
    )


def upload_random(args: argparse.Namespace):
    execute(
        [
            "aws",
            "s3",
            "cp",
            "--recursive",
            "vortex-bench/data/random_access",
            args.remote_root + "/random/random_access/",
        ],
        args.output / "upload-random.log",
    )


def random_or_compress(args: argparse.Namespace, kind: str, backend: str):
    name = f"{kind}-{backend}"
    cmd = [
        str(args.binaries / f"{kind}-bench"),
        "--formats",
        "parquet,vortex,lance",
        "--display-format",
        "gh-json",
        "--ingest-jsonl",
        str(args.output / f"{name}.jsonl"),
    ]
    if kind == "random-access":
        cmd.extend(["--datasets", "nested-lists", "--time-limit", "2", "--open-mode", "both"])
    else:
        cmd.extend(
            [
                "--datasets",
                "^wide table cols=100 chunks=1 rows=1000$",
                "--ops",
                "decompress",
                "--iterations",
                str(args.iterations),
                "--fixture-cache",
                str(args.output / "decode-fixtures"),
            ]
        )
    if backend == "simulated":
        local = Path("vortex-bench/data") if kind == "random-access" else None
        cmd = simulate(args, kind, name, cmd, local)
    else:
        suffix = "random/" if kind == "random-access" else f"decode/{backend}/"
        cmd.extend(["--remote-data-dir", args.remote_root + "/" + suffix])
    execute(cmd, args.output / f"{name}.log")


def tasks(args: argparse.Namespace) -> dict:
    result = {"calibrate": partial(calibrate, args)}
    for suite in ("tpch", "tpcds"):
        result[f"prepare-{suite}"] = partial(prepare_sql, args, suite)
        result[f"upload-{suite}"] = partial(upload_sql, args, suite)
        for engine in args.engines.split(","):
            order = ("real", "simulated", "real-repeat") if suite == "tpch" else ("simulated", "real-repeat", "real")
            for backend in order:
                result[f"sql-{suite}-{engine}-{backend}"] = partial(sql, args, suite, engine, backend)
    result["prepare-random"] = partial(prepare_random, args)
    result["upload-random"] = partial(upload_random, args)
    for kind in ("random-access", "compress"):
        for backend in ("real", "simulated", "real-repeat"):
            result[f"{kind}-{backend}"] = partial(random_or_compress, args, kind, backend)
    result["compare"] = partial(
        execute,
        [sys.executable, "benchmarks/s3/compare.py", "--directory", str(args.output)],
        args.output / "compare.log",
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--scale", default="10.0")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--binaries", type=Path, default=Path("target/release_debug"))
    parser.add_argument("--output", type=Path, default=Path("s3-benchmarks"))
    parser.add_argument("--engines", default="datafusion,duckdb,lance")
    parser.add_argument("--model", choices=("pinned", "learned"), default="pinned")
    parser.add_argument("--profile", type=Path, default=Path("benchmarks/s3/profiles/ci-us-east-1-c8gd16.json"))
    parser.add_argument(
        "--task", help="One task; CI configures standard AWS credentials before each authenticated task"
    )
    args = parser.parse_args()
    if args.iterations < 3:
        parser.error("Use at least three iterations")
    if not args.engines.split(",") or any(
        engine not in ("datafusion", "duckdb", "lance") for engine in args.engines.split(",")
    ):
        parser.error("Unknown SQL engine")
    args.binaries = args.binaries.resolve()
    selected = tasks(args)
    if args.task:
        if args.task not in selected:
            parser.error("Unknown benchmark task")
        selected[args.task]()
    else:
        # Standalone callers supply credentials valid for the whole run.
        for task in selected.values():
            task()


if __name__ == "__main__":
    main()
