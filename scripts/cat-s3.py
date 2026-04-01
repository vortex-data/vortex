#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Append JSONL benchmark results to an S3 object with duplicate-commit detection and optimistic locking."""

import argparse
import gzip
import json
import subprocess
import sys
import tempfile
import time


def head_etag(bucket: str, key: str) -> str | None:
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


def get_object(bucket: str, key: str, dest: str, if_match: str) -> bool:
    result = subprocess.run(
        [
            "aws", "s3api", "get-object",
            "--bucket", bucket,
            "--key", key,
            "--if-match", if_match,
            dest,
        ],
    )
    return result.returncode == 0


def put_object(bucket: str, key: str, body: str, if_match: str) -> bool:
    result = subprocess.run(
        [
            "aws", "s3api", "put-object",
            "--bucket", bucket,
            "--key", key,
            "--body", body,
            "--if-match", if_match,
        ],
    )
    return result.returncode == 0


def read_jsonl(path: str) -> list[str]:
    """Read a JSONL file, returning raw lines."""
    with open(path) as f:
        return [line for line in f if line.strip()]


def extract_commit_ids(lines: list[str]) -> set[str]:
    """Extract unique commit_id values from JSONL lines."""
    ids = set()
    for line in lines:
        obj = json.loads(line)
        if "commit_id" in obj:
            ids.add(obj["commit_id"])
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Append JSONL to an S3 object with dedup")
    parser.add_argument("bucket", help="S3 bucket name")
    parser.add_argument("key", help="S3 object key")
    parser.add_argument("local_file", help="Local JSONL file to append")
    parser.add_argument("--max-retries", type=int, default=100)
    args = parser.parse_args()

    is_gz = args.key.endswith(".gz")
    new_lines = read_jsonl(args.local_file)
    new_commit_ids = extract_commit_ids(new_lines)

    for attempt in range(1, args.max_retries + 1):
        etag = head_etag(args.bucket, args.key)
        if etag is None:
            print("Failed to retrieve ETag.", file=sys.stderr)
            sys.exit(1)

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            local_copy = tmp.name

        if not get_object(args.bucket, args.key, local_copy, etag):
            print(f"ETag mismatch during download (attempt {attempt}), retrying...", file=sys.stderr)
            continue

        # Read existing data.
        if is_gz:
            with gzip.open(local_copy, "rt") as f:
                existing_lines = [line for line in f if line.strip()]
        else:
            with open(local_copy) as f:
                existing_lines = [line for line in f if line.strip()]

        # Check for duplicate commits.
        existing_commit_ids = extract_commit_ids(existing_lines)
        duplicates = new_commit_ids & existing_commit_ids
        if duplicates:
            print(
                f"ERROR: commit(s) {', '.join(sorted(duplicates))} already exist in "
                f"s3://{args.bucket}/{args.key}. Refusing to append duplicate data.",
                file=sys.stderr,
            )
            sys.exit(1)

        # Concatenate.
        combined = "".join(existing_lines) + "".join(new_lines)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".gz" if is_gz else "") as tmp:
            output_path = tmp.name
            if is_gz:
                with gzip.open(output_path, "wt") as f:
                    f.write(combined)
            else:
                with open(output_path, "w") as f:
                    f.write(combined)

        if put_object(args.bucket, args.key, output_path, etag):
            print("File updated and uploaded successfully.")
            return

        print(f"ETag mismatch during upload (attempt {attempt}), retrying...", file=sys.stderr)
        time.sleep(0.1)

    print(f"Too many failures: {args.max_retries}.", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
