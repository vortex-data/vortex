#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Render per-crate compiled (.text) size as a collapsible markdown comment.

Consumes the JSON produced by ``cargo bloat --crates --message-format json``
for a linked binary, keeps only first-party workspace crates, and prints a
single ``<details>`` block: the ``<summary>`` is a one-line total so the
comment stays compact, and expanding it reveals the full per-crate breakdown
of machine code attributed to each Vortex crate.
"""

import argparse
import json
import subprocess


def fmt_size(size_bytes: int) -> str:
    """Format a byte count using binary units."""
    if size_bytes >= 1024**2:
        return f"{size_bytes / 1024**2:.2f} MiB"
    if size_bytes >= 1024:
        return f"{size_bytes / 1024:.1f} KiB"
    return f"{size_bytes} B"


def workspace_crate_names(manifest_path: str) -> set[str]:
    """Return the set of first-party crate names (as cargo-bloat reports them)."""
    out = subprocess.run(
        ["cargo", "metadata", "--no-deps", "--format-version", "1", "--manifest-path", manifest_path],
        capture_output=True,
        text=True,
        check=True,
    )
    metadata = json.loads(out.stdout)
    names: set[str] = set()
    for pkg in metadata["packages"]:
        names.add(pkg["name"].replace("-", "_"))
        for target in pkg["targets"]:
            names.add(target["name"].replace("-", "_"))
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description="Render per-crate .text size as a markdown comment")
    parser.add_argument("bloat_file", help="cargo-bloat --crates JSON output")
    parser.add_argument("--manifest-path", default="Cargo.toml", help="Workspace Cargo.toml")
    parser.add_argument("--target-name", default="datafusion-bench", help="Binary the sizes are measured from")
    args = parser.parse_args()

    with open(args.bloat_file) as f:
        bloat = json.load(f)

    workspace = workspace_crate_names(args.manifest_path)
    rows = [(c["name"], c["size"]) for c in bloat.get("crates", []) if c["name"] in workspace]
    rows.sort(key=lambda r: r[1], reverse=True)

    vortex_text = sum(size for _, size in rows)
    total_text = bloat.get("text-section-size", 0)
    share = f"{vortex_text / total_text * 100:.0f}%" if total_text else "?"

    summary = (
        f"Binary size ({args.target_name}, release): Vortex crates = {fmt_size(vortex_text)} "
        f"of .text across {len(rows)} crates ({share} of the {fmt_size(total_text)} binary)"
    )

    print("<details>")
    print(f"<summary>{summary}</summary>")
    print("")
    print("<br>")
    print("")
    print("| Crate | .text | % of Vortex |")
    print("|-------|------:|------------:|")
    for name, size in rows:
        pct = f"{size / vortex_text * 100:.1f}%" if vortex_text else "—"
        print(f"| `{name}` | {fmt_size(size)} | {pct} |")
    print("")
    print(f"**Vortex total:** {fmt_size(vortex_text)} of the {fmt_size(total_text)} binary `.text`")
    print("")
    print("</details>")


if __name__ == "__main__":
    main()
