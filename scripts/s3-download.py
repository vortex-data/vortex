#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Download a file from S3 with exponential backoff retry."""

import argparse
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description="Download a file from S3 with retry")
    parser.add_argument("s3_url", help="S3 URL to download (e.g. s3://bucket/key)")
    parser.add_argument("output", help="Local output file path")
    parser.add_argument(
        "--no-sign-request",
        action="store_true",
        help="Do not sign the request (for public buckets)",
    )
    parser.add_argument("--max-retries", type=int, default=5, help="Maximum number of retries")
    args = parser.parse_args()

    cmd = ["aws", "s3", "cp", args.s3_url, args.output]
    if args.no_sign_request:
        cmd.append("--no-sign-request")

    for attempt in range(1, args.max_retries + 1):
        result = subprocess.run(cmd)
        if result.returncode == 0:
            return

        if attempt == args.max_retries:
            break

        delay = min(2**attempt, 30)
        print(
            f"S3 download failed (attempt {attempt}/{args.max_retries}), retrying in {delay}s...",
            file=sys.stderr,
        )
        time.sleep(delay)

    print(
        f"S3 download failed after {args.max_retries} attempts",
        file=sys.stderr,
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
