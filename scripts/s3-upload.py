#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Upload a file to S3 with exponential backoff retry."""

import argparse
import subprocess
import sys
import time
from pathlib import Path

PRECONDITION_FAILED = 3


def put_object(
    bucket: str,
    key: str,
    body: str,
    checksum_algorithm: str | None,
    if_match: str | None,
    if_none_match: str | None,
) -> tuple[str, str | None]:
    """Upload an object, returning its status and new ETag."""
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
    if if_none_match:
        cmd.extend(["--if-none-match", if_none_match])
    cmd.extend(["--query", "ETag", "--output", "text"])

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        return "success", result.stdout.strip()

    error = result.stderr or ""
    if error:
        print(error.rstrip(), file=sys.stderr)
    if "PreconditionFailed" in error or "ConditionalRequestConflict" in error:
        return "precondition-failed", None
    return "failure", None


def main():
    parser = argparse.ArgumentParser(description="Upload a file to S3 with retry")
    parser.add_argument("--bucket", required=True, help="S3 bucket name")
    parser.add_argument("--key", required=True, help="S3 object key")
    parser.add_argument("--body", required=True, help="Local file to upload")
    parser.add_argument("--checksum-algorithm", help="Checksum algorithm (e.g. CRC32)")
    condition = parser.add_mutually_exclusive_group()
    condition.add_argument("--if-match", help="Only replace an object with this ETag")
    condition.add_argument(
        "--if-none-match",
        choices=["*"],
        help="Only create the object if it does not exist",
    )
    parser.add_argument(
        "--etag-output",
        help="Write the successfully uploaded object's new ETag to this file",
    )
    parser.add_argument("--max-retries", type=int, default=5, help="Maximum number of retries")
    args = parser.parse_args()
    if args.if_match == "":
        parser.error("--if-match cannot be empty")

    for attempt in range(1, args.max_retries + 1):
        status, etag = put_object(
            args.bucket,
            args.key,
            args.body,
            args.checksum_algorithm,
            args.if_match,
            args.if_none_match,
        )
        if status == "success":
            if args.etag_output:
                if not etag or etag == "None":
                    print("S3 upload succeeded without returning an ETag", file=sys.stderr)
                    sys.exit(1)
                Path(args.etag_output).write_text(f"{etag}\n")
            print("Upload successful.")
            return
        if status == "precondition-failed":
            print("S3 upload precondition failed", file=sys.stderr)
            sys.exit(PRECONDITION_FAILED)

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
