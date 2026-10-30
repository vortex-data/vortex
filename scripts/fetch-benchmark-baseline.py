#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["psycopg[binary]", "boto3"]
# ///

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Fetch the develop baseline for a PR benchmark run from the benchmarks Postgres.

Reads the PR's `--ingest-jsonl` records, finds the newest commit reachable from
`HEAD` that recorded any of the same measurements, and writes that commit's
matching rows in the same JSONL record shape. `compare-benchmark-jsons.py` then
compares the two record files.

It connects as the read-only `bench_ci_read` Postgres user, with an RDS IAM token minted
(by `post-ingest.py`'s token helper) from the AWS credentials of the assumed
`GH_BENCH_READ_ROLE_ARN` role, over verify-full TLS. The session is read-only.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
from pathlib import Path
from types import ModuleType
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

# Rows read per query while looking for an identity's newest reachable commit.
_PAGE_SIZE = 200


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pr_records", type=Path, help="The PR run's --ingest-jsonl output.")
    parser.add_argument("output", type=Path, help="Where to write the baseline records.")
    parser.add_argument(
        "--postgres",
        metavar="DSN",
        required=True,
        help="Libpq DSN of the benchmarks database. Without a password, an RDS IAM token is minted.",
    )
    parser.add_argument("--region", default=None, help="AWS region for RDS IAM token minting.")
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


def identity_condition(kind: str, identity: tuple) -> tuple[str, list]:
    """Return a `WHERE` condition on `t` matching one identity, and its parameters.

    Absent dims match with `IS NULL` rather than `IS NOT DISTINCT FROM`, which
    no index can serve, so each lookup is a range scan of the table's chart or
    summary index instead of a scan of the whole table.
    """

    _table, dims, _values = _KINDS[kind]
    conditions = []
    params = []
    for dim, value in zip(dims, identity):
        if value is None:
            conditions.append(f"t.{dim} IS NULL")
        else:
            conditions.append(f"t.{dim} = %s")
            params.append(value)
    return " AND ".join(conditions), params


def newest_baseline_commit(
    conn: Connection, identities: dict[str, list[tuple]], reachable: set[str]
) -> tuple[str, object] | None:
    """Return the newest reachable commit that recorded any PR identity, with its timestamp.

    Each identity's rows are read newest first, a page at a time, until one is from a commit
    reachable from `HEAD`. That is usually the first page: the baseline is the PR's merge base
    or close to it. One query per identity, as Postgres plans a `UNION ALL` of them far worse.
    """

    newest = None
    for kind, keys in identities.items():
        table, _dims, _values = _KINDS[kind]
        # `query_measurements` carries its commit's timestamp at the end of its summary index,
        # so its rows come back newest first straight from an index range scan.
        if table == "query_measurements":
            source, timestamp = "query_measurements t", "t.commit_timestamp"
        else:
            source, timestamp = f"{table} t JOIN commits c ON c.commit_sha = t.commit_sha", "c.timestamp"
        for identity in keys:
            where, params = identity_condition(kind, identity)
            before = None
            while True:
                page = conn.execute(
                    f"SELECT t.commit_sha, {timestamp} FROM {source}"
                    f" WHERE {where} AND {timestamp} IS NOT NULL"
                    + (f" AND {timestamp} < %s" if before is not None else "")
                    + f" ORDER BY {timestamp} DESC LIMIT {_PAGE_SIZE}",
                    [*params, before] if before is not None else params,
                ).fetchall()
                found = next((row for row in page if row[0] in reachable), None)
                if found is not None or len(page) < _PAGE_SIZE:
                    break
                before = page[-1][1]
            if found is not None and (newest is None or (found[1], found[0]) > (newest[1], newest[0])):
                newest = (found[0], found[1])
    return newest


def baseline_records(
    conn: Connection, identities: dict[str, list[tuple]], commit_sha: str, commit_timestamp: object
) -> list[dict]:
    """Return the rows `commit_sha` recorded for the PR identities, as ingest records."""

    records = []
    for kind, keys in identities.items():
        table, dims, values = _KINDS[kind]
        columns = ("commit_sha", *dims, *values)
        for identity in keys:
            where, params = identity_condition(kind, identity)
            where += " AND t.commit_sha = %s"
            params.append(commit_sha)
            if table == "query_measurements":
                where += " AND t.commit_timestamp = %s"
                params.append(commit_timestamp)
            rows = conn.execute(f"SELECT {', '.join(f't.{c}' for c in columns)} FROM {table} t WHERE {where}", params)
            records.extend({"kind": kind, **dict(zip(columns, row))} for row in rows.fetchall())
    return records


def post_ingest_module() -> ModuleType:
    """Load `scripts/post-ingest.py`, whose RDS IAM token helper this script reuses."""

    path = Path(__file__).resolve().parent / "post-ingest.py"
    spec = importlib.util.spec_from_file_location("post_ingest", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def connect_postgres(dsn: str, region: str | None) -> Connection:
    """Open a read-only, verify-full TLS connection, minting an RDS IAM token if the DSN has no password."""

    import psycopg
    from psycopg import conninfo

    params = conninfo.conninfo_to_dict(dsn)
    if params.setdefault("sslmode", "verify-full") != "verify-full":
        raise SystemExit(f"--postgres requires sslmode=verify-full; DSN specified {params['sslmode']!r}.")
    if not params.get("password"):
        params["password"] = post_ingest_module()._rds_iam_token(
            host=params["host"], port=int(params.get("port", 5432)), user=params["user"], region=region
        )
    params["connect_timeout"] = 10
    params["options"] = "-c statement_timeout=30000"
    conn = psycopg.connect(**params)
    conn.read_only = True
    return conn


def main() -> None:
    args = parse_args()
    identities = identities_by_kind(read_records(args.pr_records))
    records: list[dict] = []
    if identities:
        commits = reachable_commits()
        with connect_postgres(args.postgres, args.region) as conn:
            newest = newest_baseline_commit(conn, identities, set(commits))
            if newest is not None:
                commit_sha, commit_timestamp = newest
                records = baseline_records(conn, identities, commit_sha, commit_timestamp)
                print(f"Baseline commit {commit_sha}: {len(records)} records")
    if not records:
        print("No baseline found; the report shows PR measurements without comparison.")
    with args.output.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
