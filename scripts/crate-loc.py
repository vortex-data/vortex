#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Compute Rust lines of code per workspace crate.

Reads the workspace members from a repository's root ``Cargo.toml``, runs
``tokei`` once over the tree, and attributes each Rust source file to the
crate whose directory is its longest path prefix (so nested crates are not
double counted). Emits a JSON object mapping crate path to code-line count on
stdout.
"""

import argparse
import json
import subprocess
import tomllib
from pathlib import Path


def workspace_crates(repo_root: Path) -> list[str]:
    """Return workspace member directories relative to ``repo_root``."""
    with open(repo_root / "Cargo.toml", "rb") as f:
        manifest = tomllib.load(f)

    members = manifest.get("workspace", {}).get("members", [])
    crates: set[str] = set()
    for member in members:
        # Members may contain globs such as "encodings/*".
        matches = [member] if "*" not in member else [
            str(p.relative_to(repo_root)) for p in sorted(repo_root.glob(member))
        ]
        for candidate in matches:
            if (repo_root / candidate / "Cargo.toml").is_file():
                crates.add(candidate.replace("\\", "/"))
    return sorted(crates)


def run_tokei(repo_root: Path) -> dict:
    """Run ``tokei`` over ``repo_root`` and return its parsed JSON output."""
    result = subprocess.run(
        ["tokei", "--output", "json", "--files", str(repo_root)],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def crate_for(rel_path: str, crate_dirs: list[str]) -> str | None:
    """Find the crate whose directory is the longest prefix of ``rel_path``."""
    parts = rel_path.split("/")
    for crate in crate_dirs:
        crate_parts = crate.split("/")
        if parts[: len(crate_parts)] == crate_parts:
            return crate
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Rust LOC per workspace crate")
    parser.add_argument("repo_root", help="Path to the repository root")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    crates = workspace_crates(repo_root)
    # Match the most deeply nested crate first.
    crate_dirs = sorted(crates, key=lambda c: c.count("/"), reverse=True)

    tokei = run_tokei(repo_root)
    rust = tokei.get("Rust")
    if rust is None:
        print(json.dumps({crate: 0 for crate in crates}))
        return

    loc = {crate: 0 for crate in crates}
    for report in rust.get("reports", []):
        rel = Path(report["name"]).resolve()
        try:
            rel_path = rel.relative_to(repo_root).as_posix()
        except ValueError:
            continue
        crate = crate_for(rel_path, crate_dirs)
        if crate is not None:
            loc[crate] += report["stats"]["code"]

    print(json.dumps(loc, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
