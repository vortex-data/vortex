#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Hand benchmark binaries between jobs of one workflow run through the RunsOn S3 cache bucket.

`actions/upload-artifact` zips every file on one thread and uploads a single stream, which took
~100s for the SQL benchmark binaries. This uploads every file concurrently, each as a parallel
multipart upload, and downloads them with concurrent ranged GETs. Requires a runner with the
`extras=s3-cache` label, which provides the bucket and credentials.
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path


def s3_prefix(name: str) -> str:
    bucket = os.environ["RUNS_ON_S3_BUCKET_CACHE"]
    repo_prefix = os.environ["RUNS_ON_S3_CACHE_REPO_PREFIX"]
    run_id = os.environ["GITHUB_RUN_ID"]
    return f"s3://{bucket}/{repo_prefix}/bench-binaries/{run_id}/{name}"


@contextmanager
def transfer_env(max_concurrent_requests: int) -> Iterator[dict[str, str]]:
    """Yield an environment whose `aws` processes use tuned S3 transfer settings.

    The settings go into a private copy of the AWS config file, so later steps in the job that use
    `~/.aws/config` are unaffected.
    """
    settings = {
        "max_concurrent_requests": str(max_concurrent_requests),
        "multipart_threshold": "64MB",
        "multipart_chunksize": "64MB",
    }
    with tempfile.TemporaryDirectory() as tmp:
        config = Path(tmp) / "config"
        existing = Path(os.environ.get("AWS_CONFIG_FILE", "~/.aws/config")).expanduser()
        if existing.is_file():
            shutil.copyfile(existing, config)
        env = {**os.environ, "AWS_CONFIG_FILE": str(config)}
        # Without a profile prefix, `configure set` targets the profile in effect (`AWS_PROFILE`).
        for key, value in settings.items():
            subprocess.run(["aws", "configure", "set", f"s3.{key}", value], env=env, check=True)
        yield env


def upload_file(path: Path, dest: str, env: dict[str, str]) -> bool:
    print(f"{path.stat().st_size / 2**20:.0f} MiB {path} -> {dest}", flush=True)
    result = subprocess.run(["aws", "s3", "cp", "--no-progress", str(path), dest], env=env)
    if result.returncode != 0:
        print(f"::error::Upload failed for {path} (exit {result.returncode})", flush=True)
    return result.returncode == 0


def upload(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    files: list[Path] = []
    for pattern in args.paths:
        # Matches `upload-artifact`'s `if-no-files-found: warn`: callers list optional binaries.
        matches = [Path(match) for match in sorted(glob.glob(pattern)) if Path(match).is_file()]
        if not matches:
            print(f"::warning::No files match {pattern}")
        files.extend(matches)
    if not files:
        print("::error::No files to upload")
        return 1

    prefix = s3_prefix(args.name)
    destinations = []
    for path in files:
        try:
            rel = path.resolve().relative_to(root)
        except ValueError:
            print(f"::error::{path} is not under {args.root}")
            return 1
        destinations.append(f"{prefix}/{rel.as_posix()}")

    with transfer_env(32) as env, ThreadPoolExecutor(max_workers=len(files)) as pool:
        results = list(pool.map(lambda path, dest: upload_file(path, dest, env), files, destinations))
    return 0 if all(results) else 1


def download(args: argparse.Namespace) -> int:
    prefix = s3_prefix(args.name)
    # A recursive copy of an empty prefix succeeds, so check that the upload happened.
    listing = subprocess.run(["aws", "s3", "ls", "--recursive", f"{prefix}/"], capture_output=True, text=True)
    if listing.returncode != 0 or not listing.stdout.strip():
        print(f"::error::Nothing uploaded under {prefix}/\n{listing.stderr}")
        return 1

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    with transfer_env(64) as env:
        result = subprocess.run(["aws", "s3", "cp", "--no-progress", "--recursive", f"{prefix}/", f"{dest}/"], env=env)
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subcommands = parser.add_subparsers(required=True)

    upload_parser = subcommands.add_parser("upload", help="Upload files, keeping their paths relative to --root")
    upload_parser.add_argument("--name", required=True, help="Name shared with the download")
    upload_parser.add_argument("--root", default="target/release_debug", help="Directory relative paths are kept from")
    upload_parser.add_argument("paths", nargs="+", help="Files or glob patterns under --root")
    upload_parser.set_defaults(func=upload)

    download_parser = subcommands.add_parser("download", help="Download every file uploaded under --name")
    download_parser.add_argument("--name", required=True, help="Name passed to the upload")
    download_parser.add_argument("--dest", default="target/release_debug", help="Directory to download into")
    download_parser.set_defaults(func=download)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
