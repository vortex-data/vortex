# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Append JSONL benchmark results to an S3 object with duplicate-commit detection and optimistic locking."""

import gzip
import os
import subprocess
import sys
import tempfile
import time

import pandas as pd


def head_etag(bucket: str, key: str) -> str | None:
    result = subprocess.run(
        [
            "aws",
            "s3api",
            "head-object",
            "--bucket",
            bucket,
            "--key",
            key,
            "--query",
            "ETag",
            "--output",
            "text",
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
            "aws",
            "s3api",
            "get-object",
            "--bucket",
            bucket,
            "--key",
            key,
            "--if-match",
            if_match,
            dest,
        ],
    )
    return result.returncode == 0


def put_object(bucket: str, key: str, body: str, if_match: str) -> bool:
    result = subprocess.run(
        [
            "aws",
            "s3api",
            "put-object",
            "--bucket",
            bucket,
            "--key",
            key,
            "--body",
            body,
            "--if-match",
            if_match,
        ],
    )
    return result.returncode == 0


def extract_commit_ids(path: str, is_gz: bool) -> set[str]:
    """Extract unique commit identifiers from a JSONL file using pandas.

    Supports both benchmark data ("commit_id" column) and commit metadata ("id" column).
    """
    df = pd.read_json(path, lines=True, compression="gzip" if is_gz else None)
    ids: set[str] = set()
    if "commit_id" in df.columns:
        ids.update(df["commit_id"].dropna().unique())
    if "id" in df.columns:
        ids.update(df["id"].dropna().unique())
    return ids


def main() -> None:
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <bucket> <key> <local_file>", file=sys.stderr)
        sys.exit(1)

    bucket = sys.argv[1]
    key = sys.argv[2]
    local_file = sys.argv[3]
    max_retries = 100

    is_gz = key.endswith(".gz")

    with open(local_file) as f:
        new_data = f.read()
    new_commit_ids = extract_commit_ids(local_file, is_gz=False)

    for attempt in range(1, max_retries + 1):
        etag = head_etag(bucket, key)
        if etag is None:
            print("Failed to retrieve ETag.", file=sys.stderr)
            sys.exit(1)

        local_copy = tempfile.mktemp()
        try:
            if not get_object(bucket, key, local_copy, etag):
                print(
                    f"ETag mismatch during download (attempt {attempt}), retrying...",
                    file=sys.stderr,
                )
                continue

            # Check for duplicate commits.
            existing_commit_ids = extract_commit_ids(local_copy, is_gz)
            duplicates = new_commit_ids & existing_commit_ids
            if duplicates:
                print(
                    f"ERROR: commit(s) {', '.join(sorted(duplicates))} already exist in "
                    f"s3://{bucket}/{key}. Refusing to append duplicate data.",
                    file=sys.stderr,
                )
                sys.exit(1)

            # Decompress existing data, concatenate, recompress.
            if is_gz:
                with gzip.open(local_copy, "rt") as f:
                    existing_data = f.read()
            else:
                with open(local_copy) as f:
                    existing_data = f.read()

            combined = existing_data + new_data
            output_path = tempfile.mktemp(suffix=".gz" if is_gz else "")
            try:
                if is_gz:
                    with gzip.open(output_path, "wt") as f:
                        f.write(combined)
                else:
                    with open(output_path, "w") as f:
                        f.write(combined)

                if put_object(bucket, key, output_path, etag):
                    print("File updated and uploaded successfully.")
                    return

                print(
                    f"ETag mismatch during upload (attempt {attempt}), retrying...",
                    file=sys.stderr,
                )
                time.sleep(0.1)
            finally:
                if os.path.exists(output_path):
                    os.unlink(output_path)
        finally:
            if os.path.exists(local_copy):
                os.unlink(local_copy)

    print(f"Too many failures: {max_retries}.", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
