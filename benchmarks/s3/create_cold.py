# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Upload any missing objects of the persistent cold S3 prefix, never rewriting existing ones.

Rewriting an object would reset how long it has been idle, so existing keys are only checked.
"""

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

# Slack beyond the range so each read starts at a seeded offset, like the hot sweep.
SLACK_BYTES = 65536
BATCH = 256


def existing(root: str) -> dict[str, int]:
    listing = subprocess.run(["aws", "s3", "ls", "--recursive", root], capture_output=True, text=True)
    if listing.returncode not in (0, 1):  # 1 means the prefix is still empty.
        raise RuntimeError(listing.stderr)
    bucket_prefix = root.removeprefix("s3://").split("/", 1)[1]
    objects = {}
    for line in listing.stdout.splitlines():
        _, _, size, key = line.split(maxsplit=3)
        objects[key.removeprefix(bucket_prefix)] = int(size)
    return objects


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="s3://bucket/prefix/ that only the cold sweep reads")
    parser.add_argument("--sizes", required=True, type=lambda v: [int(s) for s in v.split(",")])
    parser.add_argument("--objects", required=True, type=int, help="Objects per range size")
    args = parser.parse_args()
    if not args.root.startswith("s3://") or not args.root.endswith("/"):
        parser.error("Root must be an s3:// prefix ending with a slash")
    present = existing(args.root)
    wanted = {f"{size}/{index:05}.bin": size + SLACK_BYTES for size in args.sizes for index in range(args.objects)}
    wrong = [key for key, size in present.items() if key in wanted and size != wanted[key]]
    if wrong:
        raise ValueError(f"Cold objects have unexpected sizes: {wrong[:5]}")
    missing = sorted(key for key in wanted if key not in present)
    print(f"{len(wanted) - len(missing)} cold objects present, uploading {len(missing)}", flush=True)
    for start in range(0, len(missing), BATCH):
        staging = Path(tempfile.mkdtemp(dir=os.environ.get("RUNNER_TEMP")))
        try:
            for key in missing[start : start + BATCH]:
                (staging / key).parent.mkdir(parents=True, exist_ok=True)
                (staging / key).write_bytes(os.urandom(wanted[key]))
            subprocess.run(
                ["aws", "s3", "cp", "--recursive", "--only-show-errors", str(staging), args.root], check=True
            )
        finally:
            shutil.rmtree(staging)


if __name__ == "__main__":
    main()
