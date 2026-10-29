#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Upload benchmark binaries' debuginfo to Polar Signals Cloud with `parca-debuginfo`.

Each binary is a separate upload, so they run concurrently. The bearer token is read from the
`POLARSIGNALS_CLOUD_TOKEN` environment variable and handed to the CLI through a private temporary
file, so it never appears on a command line.
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Another job already uploaded, or is uploading, debuginfo for the same build ID.
ALREADY_UPLOADED = re.compile(
    r"upload id mismatch|already exists|AlreadyExists|previous upload is still in-progress"
    r"|not stale yet|only stale uploads can be retried",
    re.IGNORECASE,
)


def upload(binary: Path, args: argparse.Namespace, token_file: str) -> tuple[bool, str]:
    cmd = [
        "parca-debuginfo",
        "upload",
        f"--store-address={args.store_address}",
        f"--bearer-token-file={token_file}",
        f"--grpc-headers=projectID={args.project_id}",
        str(binary),
    ]
    if args.force:
        cmd.append("--force")

    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout + result.stderr
    if result.returncode == 0:
        return True, output
    if ALREADY_UPLOADED.search(output):
        return True, output + f"::notice::Debuginfo upload already exists or is in progress for {binary}; continuing\n"
    return False, output + f"::error::Debuginfo upload failed for {binary} (exit {result.returncode})\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project-id", required=True, help="Polar Signals Cloud project ID")
    parser.add_argument("--store-address", default="grpc.polarsignals.com:443", help="Debuginfo gRPC endpoint")
    parser.add_argument("--force", action="store_true", help="Replace existing debuginfo for the same build ID")
    parser.add_argument("binaries", nargs="*", type=Path, help="Binaries whose debuginfo to upload")
    args = parser.parse_args()

    if not args.binaries:
        print("::error::No debuginfo upload targets were provided")
        sys.exit(1)
    # os.path.isfile, unlike Path.is_file, also returns False for names the OS rejects.
    missing = [binary for binary in args.binaries if not os.path.isfile(binary)]
    for binary in missing:
        print(f"::error::Debuginfo upload target does not exist: {binary}")
    if missing:
        sys.exit(1)

    token = os.environ.get("POLARSIGNALS_CLOUD_TOKEN")
    if not token:
        print("::error::POLARSIGNALS_CLOUD_TOKEN is not set")
        sys.exit(1)

    # NamedTemporaryFile creates the file with mode 0600.
    with tempfile.NamedTemporaryFile("w", prefix="polarsignals-token-") as token_file:
        token_file.write(token)
        token_file.flush()
        with ThreadPoolExecutor(max_workers=len(args.binaries)) as pool:
            results = list(pool.map(lambda binary: upload(binary, args, token_file.name), args.binaries))

    for binary, (_, output) in zip(args.binaries, results):
        print(f"::group::{binary}")
        print(output, end="")
        print("::endgroup::")
    sys.exit(0 if all(ok for ok, _ in results) else 1)


if __name__ == "__main__":
    main()
