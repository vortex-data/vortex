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
single ``<details>`` block: the ``<summary>`` is a one-line Vortex total so the
comment stays compact, and expanding it reveals the full per-crate breakdown of
machine code attributed to each Vortex crate, with deltas against ``develop``
when a base measurement is provided.
"""

import argparse
import json
import subprocess


def fmt_size(size_bytes: int) -> str:
    """Format a byte count using binary units."""
    if abs(size_bytes) >= 1024**2:
        return f"{size_bytes / 1024**2:.2f} MiB"
    if abs(size_bytes) >= 1024:
        return f"{size_bytes / 1024:.1f} KiB"
    return f"{size_bytes} B"


def fmt_delta(delta: int) -> str:
    """Format a signed size delta, or an em dash when unchanged."""
    if delta == 0:
        return "—"
    return f"{'+' if delta > 0 else '−'}{fmt_size(abs(delta))}"


def fmt_pct(base: int, head: int) -> str:
    """Format a percentage change, handling newly introduced crates."""
    if base == 0:
        return "new" if head > 0 else "—"
    if head == base:
        return "—"
    pct = (head / base - 1) * 100
    return f"{'+' if pct > 0 else '−'}{abs(pct):.1f}%"


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


def crate_sizes(bloat_file: str, workspace: set[str]) -> dict[str, int]:
    """Load a cargo-bloat JSON file and keep only first-party crate sizes."""
    with open(bloat_file) as f:
        bloat = json.load(f)
    sizes = {c["name"]: c["size"] for c in bloat.get("crates", []) if c["name"] in workspace}
    sizes["__text_section_size__"] = bloat.get("text-section-size", 0)
    return sizes


def main() -> None:
    parser = argparse.ArgumentParser(description="Render per-crate .text size as a markdown comment")
    parser.add_argument("head_file", help="cargo-bloat --crates JSON for HEAD")
    parser.add_argument("--base-file", default=None, help="cargo-bloat --crates JSON for develop")
    parser.add_argument("--manifest-path", default="Cargo.toml", help="Workspace Cargo.toml")
    parser.add_argument("--target-name", default="datafusion-bench", help="Binary the sizes are measured from")
    args = parser.parse_args()

    workspace = workspace_crate_names(args.manifest_path)
    head = crate_sizes(args.head_file, workspace)
    base = crate_sizes(args.base_file, workspace) if args.base_file else {}
    have_base = bool(base)

    total_text = head.pop("__text_section_size__", 0)
    base.pop("__text_section_size__", 0)

    crates = sorted(set(head) | set(base))
    rows = [(c, base.get(c, 0), head.get(c, 0), head.get(c, 0) - base.get(c, 0)) for c in crates]

    vortex_head = sum(h for _, _, h, _ in rows)
    vortex_base = sum(b for _, b, _, _ in rows)
    vortex_delta = vortex_head - vortex_base
    n_crates = sum(1 for _, _, h, _ in rows if h > 0)
    share = f"{vortex_head / total_text * 100:.0f}%" if total_text else "?"

    # Largest movers first, then largest crates.
    rows.sort(key=lambda r: (abs(r[3]), r[2]), reverse=True)

    if have_base and vortex_delta != 0:
        summary = (
            f"Binary size ({args.target_name}, release vs develop): Vortex crates = "
            f"{fmt_size(vortex_head)} of .text across {n_crates} crates "
            f"({fmt_delta(vortex_delta)}, {fmt_pct(vortex_base, vortex_head)}, {share} of binary)"
        )
    else:
        suffix = " vs develop: no change" if have_base else ""
        summary = (
            f"Binary size ({args.target_name}, release): Vortex crates = {fmt_size(vortex_head)} "
            f"of .text across {n_crates} crates ({share} of binary){suffix}"
        )

    # Nothing changed against develop: keep the comment to a single line.
    if have_base and vortex_delta == 0:
        print(summary)
        return

    print("<details>")
    print(f"<summary>{summary}</summary>")
    print("")
    print("<br>")
    print("")
    if have_base:
        print("| Crate | .text | Δ vs develop | % |")
        print("|-------|------:|-------------:|--:|")
        for name, b, h, d in rows:
            print(f"| `{name}` | {fmt_size(h)} | {fmt_delta(d)} | {fmt_pct(b, h)} |")
        print("")
        print(f"**Vortex total:** {fmt_size(vortex_base)} → {fmt_size(vortex_head)} ({fmt_delta(vortex_delta)})")
    else:
        print("| Crate | .text | % of Vortex |")
        print("|-------|------:|------------:|")
        for name, _, h, _ in rows:
            pct = f"{h / vortex_head * 100:.1f}%" if vortex_head else "—"
            print(f"| `{name}` | {fmt_size(h)} | {pct} |")
        print("")
        print(f"**Vortex total:** {fmt_size(vortex_head)} of the {fmt_size(total_text)} binary `.text`")
    print("")
    print("</details>")


if __name__ == "__main__":
    main()
