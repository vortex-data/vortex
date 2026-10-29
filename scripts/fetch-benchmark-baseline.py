#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["psycopg[binary]"]
# ///

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Fetch the develop baseline for a PR benchmark run from the benchmarks Postgres.

Reads the PR's `--ingest-jsonl` records, finds the newest commit reachable from
`HEAD` that recorded any of the same measurements, and writes that commit's
matching rows in the same JSONL record shape. `compare-benchmark-jsons.py` then
compares the two record files.

The connection uses the read-only `bench_read` role. Its password comes from
`$PGPASSWORD`, so it never appears in the process arguments.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from psycopg import Connection

# Per record kind: the fact table and the natural-key columns other than
# `commit_sha`. Rows from different commits match when these columns are equal.
# `None` compares equal to `None`, as the ingest path stores absent dims as NULL.
_KINDS: dict[str, tuple[str, tuple[str, ...], tuple[str, ...]]] = {
    "query_measurement": (
        "query_measurements",
        ("dataset", "dataset_variant", "scale_factor", "query_idx", "storage", "engine", "format"),
        ("value_ns", "all_runtimes_ns"),
    ),
    "compression_time": (
        "compression_times",
        ("dataset", "dataset_variant", "format", "op"),
        ("value_ns", "all_runtimes_ns"),
    ),
    "compression_size": (
        "compression_sizes",
        ("dataset", "dataset_variant", "format"),
        ("value_bytes", "uncompressed_bytes"),
    ),
    "random_access_time": (
        "random_access_times",
        ("dataset", "format", "open_mode"),
        ("value_ns", "all_runtimes_ns"),
    ),
}

_INTEGER_DIMS = frozenset({"query_idx"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pr_records", type=Path, help="The PR run's --ingest-jsonl output.")
    parser.add_argument("output", type=Path, help="Where to write the baseline records.")
    parser.add_argument(
        "--postgres",
        metavar="DSN",
        required=True,
        help="Libpq DSN for the read-only bench_read role, with sslmode=verify-full.",
    )
    return parser.parse_args()


def read_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as lines:
        return [json.loads(line) for line in lines if line.strip()]


def identities_by_kind(records: list[dict]) -> dict[str, list[tuple]]:
    """Return the distinct natural keys of the PR records, grouped by kind."""

    identities: dict[str, set[tuple]] = {}
    for record in records:
        kind = record.get("kind")
        if kind not in _KINDS:
            continue
        _table, dims, _values = _KINDS[kind]
        identities.setdefault(kind, set()).add(tuple(record.get(dim) for dim in dims))
    return {kind: sorted(keys, key=repr) for kind, keys in identities.items()}


def reachable_commits() -> list[str]:
    """Return every commit reachable from `HEAD`; a shallow checkout cannot pick a baseline."""

    is_shallow = subprocess.run(
        ["git", "rev-parse", "--is-shallow-repository"], check=True, capture_output=True, text=True
    ).stdout.strip()
    if is_shallow == "true":
        raise SystemExit(
            "Cannot resolve a baseline from a shallow checkout: the baseline is the newest "
            "recorded commit reachable from HEAD. Check the repository out with `fetch-depth: 0`."
        )
    commits = subprocess.run(["git", "rev-list", "HEAD"], check=True, capture_output=True, text=True).stdout
    return commits.split()


def _identity_join(kind: str, identities: list[tuple]) -> tuple[str, list[list]]:
    """Return a join of `t` against the PR identities, and its array parameters."""

    _table, dims, _values = _KINDS[kind]
    arrays = [[identity[i] for identity in identities] for i in range(len(dims))]
    casts = ", ".join("%s::integer[]" if dim in _INTEGER_DIMS else "%s::text[]" for dim in dims)
    conditions = " AND ".join(f"t.{dim} IS NOT DISTINCT FROM p.{dim}" for dim in dims)
    return f"JOIN unnest({casts}) AS p({', '.join(dims)}) ON {conditions}", arrays


def newest_baseline_commit(conn: Connection, identities: dict[str, list[tuple]], commits: list[str]) -> str | None:
    """Return the newest reachable commit that recorded any PR identity."""

    newest = None
    for kind, keys in identities.items():
        table, _dims, _values = _KINDS[kind]
        join, arrays = _identity_join(kind, keys)
        row = conn.execute(
            f"""
            SELECT t.commit_sha, c.timestamp
              FROM {table} t
              JOIN commits c ON c.commit_sha = t.commit_sha
              {join}
             WHERE t.commit_sha = ANY(%s::text[])
             ORDER BY c.timestamp DESC, t.commit_sha DESC
             LIMIT 1
            """,
            [*arrays, commits],
        ).fetchone()
        if row is not None and (newest is None or (row[1], row[0]) > (newest[1], newest[0])):
            newest = row
    return None if newest is None else newest[0]


def baseline_records(conn: Connection, identities: dict[str, list[tuple]], commit_sha: str) -> list[dict]:
    """Return the rows `commit_sha` recorded for the PR identities, as ingest records."""

    records = []
    for kind, keys in identities.items():
        table, dims, values = _KINDS[kind]
        join, arrays = _identity_join(kind, keys)
        columns = ("commit_sha", *dims, *values)
        rows = conn.execute(
            f"""
            SELECT {", ".join(f"t.{column}" for column in columns)}
              FROM {table} t
              {join}
             WHERE t.commit_sha = %s
            """,
            [*arrays, commit_sha],
        ).fetchall()
        records.extend({"kind": kind, **dict(zip(columns, row))} for row in rows)
    return records


def connect(dsn: str) -> Connection:
    import psycopg
    from psycopg import conninfo

    params = conninfo.conninfo_to_dict(dsn)
    if params.get("sslmode", "verify-full") != "verify-full":
        raise SystemExit("--postgres requires sslmode=verify-full")
    params["sslmode"] = "verify-full"
    params["options"] = f"{params.get('options') or ''} -c search_path=public -c statement_timeout=60000".strip()
    params["connect_timeout"] = 10
    return psycopg.connect(**params)


def main() -> None:
    args = parse_args()
    identities = identities_by_kind(read_records(args.pr_records))
    records: list[dict] = []
    if identities:
        commits = reachable_commits()
        with connect(args.postgres) as conn:
            conn.read_only = True
            commit_sha = newest_baseline_commit(conn, identities, commits)
            if commit_sha is not None:
                records = baseline_records(conn, identities, commit_sha)
                print(f"Baseline commit {commit_sha}: {len(records)} records")
    if not records:
        print("No baseline found; the report shows PR measurements without comparison.")
    with args.output.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
