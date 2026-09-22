# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
"""Run warm native benchmark processes in alternating order."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("before", type=Path)
parser.add_argument("after", type=Path)
parser.add_argument("output", type=Path, help="Prefix for the output files")
parser.add_argument("--pairs", type=int, default=3)
parser.add_argument("--backtrace", choices=["0", "1"], default="0")
args = parser.parse_args()
args.output.parent.mkdir(parents=True, exist_ok=True)
binaries = {"before": args.before.resolve(), "after": args.after.resolve()}
metadata = {
    "backtrace": args.backtrace,
    "pairs": args.pairs,
    "binaries": {
        label: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for label, path in binaries.items()
    },
}
Path(f"{args.output}.json").write_text(json.dumps(metadata, indent=2) + "\n")
for pair in range(args.pairs):
    order = ["before", "after"] if pair % 2 == 0 else ["after", "before"]
    for label in order:
        with Path(f"{args.output}-{label}-{pair}.csv").open("w") as output:
            subprocess.run(
                [str(binaries[label])],
                stdout=output,
                env={**os.environ, "RUST_BACKTRACE": args.backtrace},
                check=True,
            )
