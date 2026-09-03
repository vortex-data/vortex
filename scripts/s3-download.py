#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Download a file from S3 with exponential backoff retry."""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse


def parse_s3_url(s3_url: str) -> tuple[str, str]:
    parsed = urlparse(s3_url)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.lstrip("/"):
        raise ValueError(f"Invalid S3 URL: {s3_url}")
    return parsed.netloc, parsed.path.lstrip("/")


def main():
    parser = argparse.ArgumentParser(description="Download a file from S3 with retry")
    parser.add_argument("s3_url", help="S3 URL to download (e.g. s3://bucket/key)")
    parser.add_argument("output", help="Local output file path")
    parser.add_argument(
        "--etag-output",
        help="Write the downloaded object's ETag to this file",
    )
    parser.add_argument(
        "--no-sign-request",
        action="store_true",
        help="Do not sign the request (for public buckets)",
    )
    parser.add_argument("--max-retries", type=int, default=5, help="Maximum number of retries")
    args = parser.parse_args()

    if args.etag_output:
        try:
            bucket, key = parse_s3_url(args.s3_url)
        except ValueError as error:
            parser.error(str(error))
        cmd = [
            "aws",
            "s3api",
            "get-object",
            "--bucket",
            bucket,
            "--key",
            key,
            args.output,
            "--query",
            "ETag",
            "--output",
            "text",
        ]
    else:
        cmd = ["aws", "s3", "cp", args.s3_url, args.output]
    if args.no_sign_request:
        cmd.append("--no-sign-request")

    for attempt in range(1, args.max_retries + 1):
        result = subprocess.run(
            cmd,
            capture_output=bool(args.etag_output),
            text=True,
        )
        if result.returncode == 0:
            if args.etag_output:
                etag = result.stdout.strip()
                if not etag or etag == "None":
                    print("S3 download succeeded without returning an ETag", file=sys.stderr)
                    sys.exit(1)
                Path(args.etag_output).write_text(f"{etag}\n")
            return

        if result.stderr:
            print(result.stderr.rstrip(), file=sys.stderr)

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
