#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Merge a local corpus with the remote corpus in S3 and upload using etag-based CAS.

In fuzz mode (default), the result is the union of local and remote files.
In minimize mode (--original-snapshot), the result is the local (minimized) files
plus any remote files that were NOT in the original snapshot (i.e. new discoveries
from concurrent fuzz runs).
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time


def head_etag(bucket: str, key: str) -> str | None:
    """Fetch the current ETag for an object, or None if it doesn't exist."""
    result = subprocess.run(
        [
            "aws", "s3api", "head-object",
            "--bucket", bucket,
            "--key", key,
            "--query", "ETag",
            "--output", "text",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    etag = result.stdout.strip()
    if not etag or etag == "null":
        return None
    return etag


def get_object(bucket: str, key: str, output: str) -> bool:
    """Download an object from S3."""
    result = subprocess.run(
        ["aws", "s3api", "get-object", "--bucket", bucket, "--key", key, output],
        capture_output=True,
    )
    return result.returncode == 0


def put_object(
    bucket: str,
    key: str,
    body: str,
    checksum_algorithm: str | None,
    if_match: str | None,
) -> bool:
    """Upload an object, returning True on success."""
    cmd = [
        "aws", "s3api", "put-object",
        "--bucket", bucket,
        "--key", key,
        "--body", body,
    ]
    if checksum_algorithm:
        cmd.extend(["--checksum-algorithm", checksum_algorithm])
    if if_match:
        cmd.extend(["--if-match", if_match])
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0


def list_files(directory: str) -> set[str]:
    """List regular file names (not paths) in a directory."""
    if not os.path.isdir(directory):
        return set()
    return {f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))}


def main():
    parser = argparse.ArgumentParser(
        description="Merge local corpus with remote and upload with etag CAS",
    )
    parser.add_argument("--bucket", required=True, help="S3 bucket name")
    parser.add_argument("--key", required=True, help="S3 object key for the corpus tar")
    parser.add_argument(
        "--corpus-dir", required=True,
        help="Local corpus directory (also used as the path inside the tar)",
    )
    parser.add_argument(
        "--original-snapshot",
        help="File listing original corpus filenames, one per line (minimize mode)",
    )
    parser.add_argument("--checksum-algorithm", help="Checksum algorithm (e.g. CRC32)")
    parser.add_argument("--timeout", type=int, default=300, help="Maximum total retry time in seconds")
    args = parser.parse_args()

    local_files = list_files(args.corpus_dir)
    print(f"Local corpus: {len(local_files)} files")

    original_files: set[str] | None = None
    if args.original_snapshot:
        with open(args.original_snapshot) as f:
            original_files = {line.strip() for line in f if line.strip()}
        print(f"Original snapshot: {len(original_files)} files")

    deadline = time.monotonic() + args.timeout
    attempt = 0
    while True:
        attempt += 1
        ok = _try_merge_upload(args, local_files, original_files, attempt)
        if ok:
            return

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break

        # Exponential backoff for first 3 attempts (2s, 4s, 8s), then 10s polling
        delay = min(2**attempt, 10) if attempt <= 3 else 10
        delay = min(delay, remaining)
        print(
            f"ETag conflict (attempt {attempt}), retrying in {delay:.0f}s... "
            f"({remaining:.0f}s remaining)",
            file=sys.stderr,
        )
        time.sleep(delay)

    print(f"Corpus merge-upload failed after {attempt} attempts ({args.timeout}s timeout)", file=sys.stderr)
    sys.exit(1)


def _try_merge_upload(
    args: argparse.Namespace,
    local_files: set[str],
    original_files: set[str] | None,
    attempt: int,
) -> bool:
    """Single attempt: download remote, merge, upload with etag CAS. Returns True on success."""
    with tempfile.TemporaryDirectory() as merge_dir:
        merge_corpus = os.path.join(merge_dir, args.corpus_dir)
        os.makedirs(merge_corpus, exist_ok=True)

        # Start with local files
        for f in local_files:
            shutil.copy2(os.path.join(args.corpus_dir, f), os.path.join(merge_corpus, f))

        # Download remote corpus and get its etag
        etag = head_etag(args.bucket, args.key)
        if etag:
            remote_tar = tempfile.mktemp(suffix=".tar.zst")
            try:
                if get_object(args.bucket, args.key, remote_tar):
                    remote_extract = tempfile.mkdtemp()
                    subprocess.run(["tar", "-xf", remote_tar, "-C", remote_extract], check=False)
                    remote_corpus = os.path.join(remote_extract, args.corpus_dir)
                    remote_files = list_files(remote_corpus)

                    if original_files is not None:
                        # Minimize mode: only add files that are genuinely new
                        new_files = remote_files - original_files
                        print(f"Preserving {len(new_files)} new entries from concurrent runs")
                        files_to_add = new_files
                    else:
                        # Fuzz mode: union merge
                        files_to_add = remote_files

                    for f in files_to_add:
                        dest = os.path.join(merge_corpus, f)
                        if not os.path.exists(dest):
                            shutil.copy2(os.path.join(remote_corpus, f), dest)

                    shutil.rmtree(remote_extract, ignore_errors=True)
            finally:
                if os.path.exists(remote_tar):
                    os.unlink(remote_tar)

        merged_count = len(list_files(merge_corpus))
        print(f"Merged corpus: {merged_count} files (attempt {attempt})")

        # Tar and upload with etag CAS
        merged_tar = tempfile.mktemp(suffix=".tar.zst")
        try:
            subprocess.run(
                ["tar", "-acf", merged_tar, "-C", merge_dir, args.corpus_dir],
                check=True,
            )
            if put_object(args.bucket, args.key, merged_tar, args.checksum_algorithm, etag):
                print("Corpus merged and uploaded successfully.")
                return True
        finally:
            if os.path.exists(merged_tar):
                os.unlink(merged_tar)

    return False


if __name__ == "__main__":
    main()
