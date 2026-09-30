#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Compare prebuilt readers in both directions using a separate fixture corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True, help="Old vortex-compat binary")
    parser.add_argument("--candidate", type=Path, required=True, help="New vortex-compat binary")
    parser.add_argument("--output", type=Path, required=True, help="New directory for files and results")
    parser.add_argument("--exclude", default="", help="Fixture name substrings, separated by commas")
    args = parser.parse_args()
    binaries = {
        "baseline": args.baseline.resolve(strict=True),
        "candidate": args.candidate.resolve(strict=True),
    }
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    commands: list[dict[str, object]] = []
    manifest = {
        "binaries": {
            name: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for name, path in binaries.items()
        },
        "exclude": args.exclude,
        "commands": commands,
    }
    manifest_path = output / "comparison.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    excludes = ["--exclude", args.exclude] if args.exclude else []

    def run(label: str, command: list[str]):
        print(label, flush=True)
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        (output / f"{label}.stdout").write_text(result.stdout)
        (output / f"{label}.stderr").write_text(result.stderr)
        commands.append({"label": label, "argv": command, "exit_code": result.returncode})
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        result.check_returncode()

    for writer, binary in binaries.items():
        run(
            f"generate-{writer}",
            [str(binary), "generate", "--output", str(output / writer), *excludes],
        )
    for reader, binary in binaries.items():
        for writer in binaries:
            run(
                f"read-{reader}-written-{writer}",
                [str(binary), "check", "--dir", str(output / writer), "--mode", "exact", *excludes],
            )


if __name__ == "__main__":
    main()
