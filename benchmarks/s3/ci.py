# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Calibrate on real S3, then exercise native benchmark clients on one CI host."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import aws_credentials
import learn


def execute(cmd: list[str], log: Path, env: dict[str, str] | None = None):
    print(json.dumps({"command": cmd, "log": str(log)}), flush=True)
    with log.open("w") as output:
        subprocess.run(cmd, env=env, stdout=output, stderr=subprocess.STDOUT, check=True, timeout=21600)


def duckdb_environment(env: dict[str, str]) -> dict[str, str]:
    """Native httpfs reads static environment credentials; refresh them before each suite."""
    if "AWS_WEB_IDENTITY_TOKEN_FILE" not in env:
        return env
    response = subprocess.run(
        [
            "aws",
            "sts",
            "assume-role-with-web-identity",
            "--role-arn",
            env["AWS_ROLE_ARN"],
            "--role-session-name",
            "GitHubS3ModelBenchmarks",
            "--duration-seconds",
            "7200",
            "--web-identity-token",
            "file://" + env["AWS_WEB_IDENTITY_TOKEN_FILE"],
            "--output",
            "json",
        ],
        env=env,
        capture_output=True,
        check=True,
        text=True,
        timeout=60,
    )
    credentials = json.loads(response.stdout)["Credentials"]
    return {
        **env,
        "AWS_ACCESS_KEY_ID": credentials["AccessKeyId"],
        "AWS_SECRET_ACCESS_KEY": credentials["SecretAccessKey"],
        "AWS_SESSION_TOKEN": credentials["SessionToken"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--scale", default="10.0")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--binaries", type=Path, default=Path("target/release_debug"))
    parser.add_argument("--output", type=Path, default=Path("s3-benchmarks"))
    parser.add_argument("--engines", default="datafusion,duckdb,lance")
    args = parser.parse_args()
    if args.iterations < 3:
        raise ValueError("Use at least three iterations")
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    binary = args.binaries.resolve()
    run_id = os.environ["GITHUB_RUN_ID"]
    profile = root / "profile.json"
    with aws_credentials.environment(os.environ.get("S3_MEASUREMENT_ROLE_ARN")) as real_env:
        for window in (1, 2):
            execute(
                [
                    str(binary / "s3-measure"),
                    "--source",
                    args.remote_root + "/probe/",
                    "--output",
                    str(root / f"training-{window}.jsonl"),
                    "--measurement-id",
                    f"{run_id}-calibration-{window}",
                ],
                root / f"training-{window}.log",
                real_env,
            )
        model = learn.learn([root / "training-1.jsonl", root / "training-2.jsonl"])
        profile.write_text(json.dumps(model, separators=(",", ":")) + "\n")
        fixtures = root / "probe-fixtures"
        fixtures.mkdir()
        for index in range(1, 5):
            shutil.copyfile("s3-probe.bin", fixtures / f"probe-{index}.bin")
        execute(
            [
                str(binary / "s3-measure"),
                "--source",
                args.remote_root + "/probe/",
                "--output",
                str(root / "heldout-real.jsonl"),
                "--measurement-id",
                f"{run_id}-heldout-real",
            ],
            root / "heldout-real.log",
            real_env,
        )
        for label, config in (("simulated", ["--profile", str(profile)]), ("zero", ["--zero-delay"])):
            execute(
                [
                    sys.executable,
                    "benchmarks/s3/run.py",
                    "--server",
                    str(binary / "s3-simulate"),
                    *config,
                    "--kind",
                    "probe",
                    "--data-root",
                    str(fixtures),
                    "--output",
                    str(root / f"probe-{label}-manifest.json"),
                    "--",
                    str(binary / "s3-measure"),
                    "--output",
                    str(root / f"probe-{label}.jsonl"),
                    "--measurement-id",
                    f"{run_id}-{label}",
                ],
                root / f"probe-{label}.log",
            )
        for suite in ("tpch", "tpcds"):
            execute(
                [str(binary / "data-gen"), suite, "--formats", "parquet,vortex", "--opt", f"scale-factor={args.scale}"],
                root / f"prepare-{suite}.log",
            )
            if "lance" in args.engines.split(","):
                # A local first query prepares Lance through its ordinary conversion path.
                execute(
                    [
                        str(binary / "lance-bench"),
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
                    root / f"prepare-{suite}-lance.log",
                )
            local = Path("vortex-bench/data") / suite / args.scale
            remote = f"{args.remote_root}/{suite}/"
            execute(["aws", "s3", "cp", "--recursive", str(local), remote], root / f"upload-{suite}.log", real_env)
            for engine in args.engines.split(","):
                # Alternate orders across suites, retaining a second real control.
                order = (
                    ("real", "simulated", "real-repeat") if suite == "tpch" else ("simulated", "real-repeat", "real")
                )
                for backend in order:
                    name = f"{suite}-{engine}-{backend}"
                    cmd = [
                        str(binary / f"{engine}-bench"),
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
                        str(root / f"{name}.jsonl"),
                    ]
                    if engine != "lance":
                        cmd.extend(["--formats", "parquet,vortex"])
                    if engine == "duckdb":
                        cmd.append("--delete-duckdb-database")
                    if backend == "simulated":
                        cmd = [
                            sys.executable,
                            "benchmarks/s3/run.py",
                            "--server",
                            str(binary / "s3-simulate"),
                            "--profile",
                            str(profile),
                            "--kind",
                            "sql",
                            "--data-root",
                            str(local),
                            "--timeout",
                            "21600",
                            "--output",
                            str(root / f"{name}-manifest.json"),
                            "--",
                            *cmd,
                        ]
                        execute(cmd, root / f"{name}.log")
                    else:
                        execute(
                            [*cmd, "--opt", f"remote-data-dir={remote}"],
                            root / f"{name}.log",
                            duckdb_environment(real_env) if engine == "duckdb" else real_env,
                        )
        # The same range/read clients are used with real and simulated S3. Preparation is untimed.
        random_local = Path("vortex-bench/data")
        execute(
            [
                str(binary / "random-access-bench"),
                "--prepare-data",
                "--datasets",
                "nested-lists",
                "--formats",
                "parquet,vortex,lance",
            ],
            root / "prepare-random.log",
        )
        execute(
            [
                "aws",
                "s3",
                "cp",
                "--recursive",
                str(random_local / "random_access"),
                args.remote_root + "/random/random_access/",
            ],
            root / "upload-random.log",
            real_env,
        )
        for kind in ("random-access", "compress"):
            for backend in ("real", "simulated", "real-repeat"):
                name = f"{kind}-{backend}"
                cmd = [
                    str(binary / f"{kind}-bench"),
                    "--formats",
                    "parquet,vortex,lance",
                    "--display-format",
                    "gh-json",
                    "--ingest-jsonl",
                    str(root / f"{name}.jsonl"),
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
                            str(root / "decode-fixtures"),
                        ]
                    )
                if backend == "simulated":
                    cmd = [
                        sys.executable,
                        "benchmarks/s3/run.py",
                        "--server",
                        str(binary / "s3-simulate"),
                        "--profile",
                        str(profile),
                        "--kind",
                        kind,
                        "--output",
                        str(root / f"{name}-manifest.json"),
                        *(["--data-root", str(random_local)] if kind == "random-access" else []),
                        "--",
                        *cmd,
                    ]
                    execute(cmd, root / f"{name}.log")
                else:
                    suffix = "random/" if kind == "random-access" else f"decode/{backend}/"
                    execute(
                        [*cmd, "--remote-data-dir", args.remote_root + "/" + suffix], root / f"{name}.log", real_env
                    )
    execute([sys.executable, "benchmarks/s3/compare.py", "--directory", str(root)], root / "compare.log")


if __name__ == "__main__":
    main()
