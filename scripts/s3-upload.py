#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Upload a file to S3 with exponential backoff retry and optional optimistic locking."""

import argparse
import subprocess
import sys
import time


def head_etag(bucket: str, key: str) -> str | None:
    """Fetch the current ETag for an object, or None if it doesn't exist."""
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


def put_object(
    bucket: str,
    key: str,
    body: str,
    checksum_algorithm: str | None,
    if_match: str | None,
) -> bool:
    """Upload an object, returning True on success."""
    cmd = [
        "aws",
        "s3api",
        "put-object",
        "--bucket",
        bucket,
        "--key",
        key,
        "--body",
        body,
    ]
    if checksum_algorithm:
        cmd.extend(["--checksum-algorithm", checksum_algorithm])
    if if_match:
        cmd.extend(["--if-match", if_match])

    result = subprocess.run(cmd)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Upload a file to S3 with retry and optional optimistic locking")
    parser.add_argument("--bucket", required=True, help="S3 bucket name")
    parser.add_argument("--key", required=True, help="S3 object key")
    parser.add_argument("--body", required=True, help="Local file to upload")
    parser.add_argument("--checksum-algorithm", help="Checksum algorithm (e.g. CRC32)")
    parser.add_argument(
        "--optimistic-lock",
        action="store_true",
        help="Use ETag-based optimistic locking (re-fetches ETag on each retry)",
    )
    parser.add_argument("--max-retries", type=int, default=5, help="Maximum number of retries")
    args = parser.parse_args()

    for attempt in range(1, args.max_retries + 1):
        if_match = None
        if args.optimistic_lock:
            if_match = head_etag(args.bucket, args.key)
            # New object, no ETag to match — just upload without locking
            # (this handles the first-ever upload case)

        if put_object(args.bucket, args.key, args.body, args.checksum_algorithm, if_match):
            print("Upload successful.")
            return

        if attempt == args.max_retries:
            break

        delay = min(2**attempt, 30)
        print(
            f"S3 upload failed (attempt {attempt}/{args.max_retries}), retrying in {delay}s...",
            file=sys.stderr,
        )
        time.sleep(delay)

    print(
        f"S3 upload failed after {args.max_retries} attempts",
        file=sys.stderr,
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
