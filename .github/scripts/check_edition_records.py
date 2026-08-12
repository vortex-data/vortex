#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["pygit2>=1.14"]
# ///
"""Check that frozen edition records under `vortex/editions` never change.

A record's mutability follows its edition. A draft is still being assembled, so its record may
change, be renamed, or be dropped. Freezing -- recording a `min_vortex_version` -- turns the
record into a read-forever contract, and from then on it may never change again. Whether a
record was frozen is read from the base revision, so a change cannot unfreeze an edition and
edit it in the same diff.

A newly added record must also be newer than every edition already recorded for its family:
editions are only ever added going forward. Records are grouped by family, so
`vortex/editions/core/core2025.05.0.toml` must sit under the family its name declares. The
`family.toml` beside them documents the family rather than pinning a contract, so it is exempt.

Both revisions are read straight out of the object database, so the check sees committed
state only and never the working tree.

Usage:
    uv run --script .github/scripts/check_edition_records.py --base origin/develop
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import PurePosixPath
from typing import Any, Iterator

import pygit2
from pygit2.enums import DeltaStatus

RECORD_DIR = "vortex/editions"

# `core/core2026.08.0.toml`: the file name is the edition id and its directory is the family,
# so a record's identity is visible in the diff without reading the file.
RECORD_NAME = re.compile(
    r"^(?P<family>[a-z]+)(?P<year>\d{4})\.(?P<month>\d{2})\.(?P<version>\d+)\.toml$"
)

# A record carries this exactly when the edition it records is frozen.
FROZEN_MARKER = "min_vortex_version"

# Beside each family's editions sits a record of the family itself. That one is documentation
# rather than a contract, so it stays editable and these rules leave it alone.
FAMILY_FILE = "family.toml"

# Every way a record can change other than being added. Renames and copies carry an old path
# and a new one; the rest carry one.
CHANGE_VERBS = {
    DeltaStatus.MODIFIED: "modifies",
    DeltaStatus.DELETED: "deletes",
    DeltaStatus.RENAMED: "renames",
    DeltaStatus.COPIED: "copies",
    DeltaStatus.TYPECHANGE: "retypes",
}

REMEDY = (
    "A frozen edition is immutable. To add encodings, declare a NEW edition in\n"
    "  vortex-edition/src/declarations/<family>/ and regenerate the records with\n"
    "  `cargo run -p xtask -- generate-editions`."
)


def parse_record(text: str, where: str) -> dict[str, Any]:
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        sys.exit(f"{where} is not valid TOML: {error}")


def read_record(commit: pygit2.Commit, path: str) -> dict[str, Any] | None:
    """Parse a record out of a commit's tree, or None when it holds no such file."""
    try:
        blob = commit.tree[path]
    except KeyError:
        return None
    return parse_record(blob.data.decode(), f"{path} at {commit.short_id}")


def record_paths(commit: pygit2.Commit) -> Iterator[str]:
    """Every record path in a commit, relative to the repository root."""

    def walk(tree: pygit2.Tree, prefix: str) -> Iterator[str]:
        for entry in tree:
            path = f"{prefix}/{entry.name}"
            if isinstance(entry, pygit2.Tree):
                yield from walk(entry, path)
            elif entry.name.endswith(".toml") and entry.name != FAMILY_FILE:
                yield path

    try:
        records = commit.tree[RECORD_DIR]
    except KeyError:
        return
    yield from walk(records, RECORD_DIR)


def parse_name(name: str) -> tuple[str, tuple[int, int, int]]:
    """Split a record file name into its family and its chronological sort key."""
    match = RECORD_NAME.match(name)
    if match is None:
        sys.exit(
            f"{RECORD_DIR}/{name} is not a valid record name.\n"
            "Records are named after the edition they record, e.g. "
            "`core/core2026.08.0.toml`."
        )
    return match["family"], (int(match["year"]), int(match["month"]), int(match["version"]))


def changed_records(base: pygit2.Commit, head: pygit2.Commit) -> pygit2.Diff:
    """The record directory's diff between two commits, with renames detected."""
    diff = base.tree.diff_to_tree(head.tree)
    diff.find_similar()
    return diff


def newest_recorded(commit: pygit2.Commit) -> dict[str, tuple[int, int, int]]:
    """The newest edition already recorded for each family at `commit`."""
    newest: dict[str, tuple[int, int, int]] = {}
    for path in record_paths(commit):
        family, key = parse_name(PurePosixPath(path).name)
        newest[family] = max(key, newest.get(family, (0, 0, 0)))
    return newest


def check_modification(before: dict[str, Any], after: dict[str, Any], name: str) -> list[str]:
    """A frozen record may not change at all; name the fields that did."""
    changed = sorted(
        key for key in before.keys() | after.keys() if before.get(key) != after.get(key)
    )
    if not changed:
        return []
    if FROZEN_MARKER in changed and FROZEN_MARKER not in after:
        return [
            f"unfreezes {name}; an edition that recorded a {FROZEN_MARKER} carries a "
            "read-forever guarantee and may never return to draft"
        ]
    return [f"modifies the frozen record {name}: {', '.join(changed)}"]


def check_addition(
    path: str, record: dict[str, Any], newest: dict[str, tuple[int, int, int]]
) -> list[str]:
    """A new record must extend its family's chronology, and be filed under it."""
    errors = []
    name = PurePosixPath(path).name
    family, key = parse_name(name)

    previous = newest.get(family)
    if previous is not None and key <= previous:
        recorded = f"{previous[0]}.{previous[1]:02}.{previous[2]}"
        errors.append(
            f"adds {name}, which is not newer than the {family} edition already "
            f"recorded ({family}{recorded}). Editions may only be added going forward."
        )

    # A record's family decides which chronology it extends, so the directory it sits in has
    # to agree with the family its name declares.
    directory = PurePosixPath(path).parent.name
    if directory != family:
        errors.append(
            f"adds {name} under {directory}/, but it records a {family} edition; "
            "records are grouped by family"
        )

    # The file name is the edition's identity, so it has to agree with the content.
    edition = record.get("edition")
    if edition is None:
        errors.append(f"adds {name}, which has no `edition` field")
    elif edition != name.removesuffix(".toml"):
        errors.append(
            f"adds {name}, which records edition {edition!r}; the file name must be the edition id"
        )
    return errors


def under_record_dir(*paths: str | None) -> bool:
    return any(path is not None and path.startswith(f"{RECORD_DIR}/") for path in paths)


def check(base: pygit2.Commit, head: pygit2.Commit) -> list[str]:
    errors: list[str] = []
    added: list[str] = []

    for patch in changed_records(base, head):
        delta = patch.delta
        old_path, new_path = delta.old_file.path, delta.new_file.path
        if not under_record_dir(old_path, new_path):
            continue
        if PurePosixPath(new_path).name == FAMILY_FILE:
            continue

        if delta.status == DeltaStatus.ADDED:
            added.append(new_path)
            continue

        # Frozen-ness comes from the base revision, so a diff cannot unfreeze an edition and
        # then edit it. A draft's record is free to change, move, or go away with the draft.
        before = read_record(base, old_path)
        if before is None or FROZEN_MARKER not in before:
            continue

        if delta.status == DeltaStatus.MODIFIED:
            after = read_record(head, new_path) or {}
            errors.extend(check_modification(before, after, PurePosixPath(new_path).name))
        else:
            verb = CHANGE_VERBS.get(delta.status, "changes")
            moved = old_path if old_path == new_path else f"{old_path} -> {new_path}"
            errors.append(f"{verb} the frozen record {moved}")

    newest = newest_recorded(base)
    for path in sorted(added):
        record = read_record(head, path)
        if record is None:
            continue
        errors.extend(check_addition(path, record, newest))

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        default="origin/develop",
        help="the revision to compare against (default: origin/develop)",
    )
    args = parser.parse_args()

    repo = pygit2.Repository(pygit2.discover_repository("."))
    try:
        base_tip = repo.revparse_single(args.base).peel(pygit2.Commit)
    except KeyError:
        sys.exit(f"cannot resolve {args.base!r} in this repository")

    head = repo.head.peel(pygit2.Commit)
    merge_base = repo.merge_base(base_tip.id, head.id)
    if merge_base is None:
        sys.exit(
            f"{args.base} and HEAD have no common ancestor.\n"
            "The checkout is probably too shallow; this check needs `fetch-depth: 0`."
        )
    base = repo[merge_base]

    errors = check(base, head)
    if not errors:
        print(f"{RECORD_DIR} preserves every frozen record against {args.base} ({base.short_id}).")
        return 0

    print(f"This change breaks the edition records in {RECORD_DIR}:\n", file=sys.stderr)
    for error in errors:
        print(f"  - it {error}", file=sys.stderr)
    print(f"\n{REMEDY}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
