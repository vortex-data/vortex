#!/usr/bin/env python3
"""Check that the frozen edition records under `vortex/editions` are append-only.

A frozen edition carries a read-forever guarantee, so its record may never change: the only
legal edit to the directory is adding a file for a newly frozen edition, and that edition must
be newer than every edition already recorded for its family.

Usage:
    python3 check_frozen_editions.py --base origin/develop
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

RECORD_DIR = "vortex/editions"

# `core2026.08.0.toml`: the file name is the edition id, so the record's identity is visible
# in the diff without reading the file.
RECORD_NAME = re.compile(
    r"^(?P<family>[a-z]+)(?P<year>\d{4})\.(?P<month>\d{2})\.(?P<version>\d+)\.toml$"
)

EDITION_FIELD = re.compile(r'^edition = "(?P<edition>[^"]+)"$', re.MULTILINE)

REMEDY = (
    "A frozen edition is immutable. To add encodings, declare a NEW edition in\n"
    "  vortex/src/editions/<family>/ and regenerate the records with\n"
    "  `UPDATE_FROZEN_EDITIONS=1 cargo test -p vortex --lib editions::frozen`."
)


def git(*args: str) -> str:
    result = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        sys.exit(f"git {' '.join(args)} failed:\n{result.stderr.strip()}")
    return result.stdout


def merge_base(base: str) -> str:
    result = subprocess.run(
        ["git", "merge-base", base, "HEAD"], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        sys.exit(
            f"cannot find a merge base between {base} and HEAD:\n"
            f"{result.stderr.strip()}\n"
            "The checkout is probably too shallow; this check needs `fetch-depth: 0`."
        )
    return result.stdout.strip()


def parse_name(name: str) -> tuple[str, tuple[int, int, int]]:
    """Split a record file name into its family and its chronological sort key."""
    match = RECORD_NAME.match(name)
    if match is None:
        sys.exit(
            f"{RECORD_DIR}/{name} is not a valid record name.\n"
            "Records are named after the edition they record, e.g. `core2026.08.0.toml`."
        )
    return match["family"], (int(match["year"]), int(match["month"]), int(match["version"]))


def changed_records(base: str) -> list[tuple[str, list[str]]]:
    """The status and paths of every change to the record directory since `base`."""
    raw = git("diff", "--name-status", "-z", base, "HEAD", "--", RECORD_DIR)
    fields = [field for field in raw.split("\0") if field]
    changes: list[tuple[str, list[str]]] = []
    index = 0
    while index < len(fields):
        status = fields[index]
        # Renames and copies carry both the old and the new path.
        count = 2 if status[0] in ("R", "C") else 1
        changes.append((status, fields[index + 1 : index + 1 + count]))
        index += 1 + count
    return changes


def recorded_at(base: str) -> dict[str, tuple[int, int, int]]:
    """The newest edition already recorded for each family at `base`."""
    newest: dict[str, tuple[int, int, int]] = {}
    listing = git("ls-tree", "-r", "--name-only", base, "--", RECORD_DIR)
    for path in listing.splitlines():
        family, key = parse_name(Path(path).name)
        newest[family] = max(key, newest.get(family, (0, 0, 0)))
    return newest


def check(base: str) -> list[str]:
    errors: list[str] = []
    added: list[str] = []

    for status, paths in changed_records(base):
        if status == "A":
            added.extend(paths)
            continue
        verb = {"M": "modifies", "D": "deletes", "R": "renames", "C": "copies", "T": "retypes"}
        errors.append(f"{verb.get(status[0], 'changes')} the frozen record {' -> '.join(paths)}")

    newest = recorded_at(base)
    for path in sorted(added):
        name = Path(path).name
        family, key = parse_name(name)

        previous = newest.get(family)
        if previous is not None and key <= previous:
            recorded = f"{previous[0]}.{previous[1]:02}.{previous[2]}"
            errors.append(
                f"adds {name}, which is not newer than the {family} edition already "
                f"recorded ({family}{recorded}). Editions may only be added going forward."
            )

        # The file name is the edition's identity, so it has to agree with the content.
        text = Path(path).read_text()
        match = EDITION_FIELD.search(text)
        if match is None:
            errors.append(f"adds {name}, which has no `edition` field")
        elif match["edition"] != name.removesuffix(".toml"):
            errors.append(
                f"adds {name}, which records edition {match['edition']!r}; the file name "
                "must be the edition id"
            )

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        default="origin/develop",
        help="the revision to compare against (default: origin/develop)",
    )
    args = parser.parse_args()

    base = merge_base(args.base)
    errors = check(base)
    if not errors:
        print(f"{RECORD_DIR} is append-only against {args.base} ({base[:12]}).")
        return 0

    print(f"This change breaks the frozen edition records in {RECORD_DIR}:\n", file=sys.stderr)
    for error in errors:
        print(f"  - it {error}", file=sys.stderr)
    print(f"\n{REMEDY}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
