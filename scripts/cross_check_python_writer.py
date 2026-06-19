#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
"""Cross-check the Python ``--postgres`` ingest writer against seeded rows.

The v4 historical seed (the PR-3.4 LOCAL rehearsal, then the PR-5.0 prod load)
loads the v3 DuckDB into Postgres via the Rust loader, which copies the
(Rust-computed) ``measurement_id`` values verbatim. This
harness confirms the LIVE property the upsert-not-duplicate invariant depends on:
the Python ``post-ingest.py --postgres`` writer, given a v3 envelope whose dim
tuple already exists in the seeded data, recomputes the SAME ``measurement_id``
(Python port == Rust hash, golden-gated) and UPDATEs -- rather than
duplicate-INSERTs -- the seeded row, with the value columns round-tripping.

Run it right after the seed (the PR-3.4 LOCAL rehearsal, then the PR-5.0 prod
seed) for earliest detection, and re-run it as the PR-5.1 pre-promotion gate:

    uv run scripts/cross_check_python_writer.py --postgres "$DSN" \\
        --envelopes real_envelopes.json

``--envelopes`` is a JSON file ``{"commit": {...}, "records": [...]}`` in the v3
envelope shape ``post-ingest.py --postgres`` consumes; every record's dim tuple
MUST already exist in the seeded data (so the writer UPDATEs, not INSERTs). The
harness exits non-zero on any problem: an INSERT where an UPDATE was expected (a
measurement_id that did not match a seeded row -> a would-be duplicate), or a
value column that did not round-trip.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent


def _load_module(filename: str, modname: str):
    """Load a sibling script by file path (the hyphen in some names blocks ``import``)."""
    path = _SCRIPTS_DIR / filename
    spec = importlib.util.spec_from_file_location(modname, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_TABLE_BY_KIND = {
    "query_measurement": "query_measurements",
    "compression_time": "compression_times",
    "compression_size": "compression_sizes",
    "random_access_time": "random_access_times",
    "vector_search_run": "vector_search_runs",
}

# Value (non-hashed) columns compared per kind, mirroring the verifier (PR-3.2)
# and the writer's ``ON CONFLICT DO UPDATE SET`` lists. Only the columns a record
# actually carries are compared; the rest default and are the writer's concern.
_VALUE_COLUMNS = {
    "query_measurement": (
        "value_ns",
        "all_runtimes_ns",
        "peak_physical",
        "peak_virtual",
        "physical_delta",
        "virtual_delta",
        "env_triple",
    ),
    "compression_time": ("value_ns", "all_runtimes_ns", "env_triple"),
    "compression_size": ("value_bytes",),
    "random_access_time": ("value_ns", "all_runtimes_ns", "env_triple"),
    "vector_search_run": (
        "value_ns",
        "all_runtimes_ns",
        "matches",
        "rows_scanned",
        "bytes_scanned",
        "iterations",
        "env_triple",
    ),
}


def measurement_id_for(mid_mod, record: dict) -> int:
    """Recompute a record's ``measurement_id`` independently of the writer."""
    kind = record["kind"]
    if kind == "query_measurement":
        return mid_mod.measurement_id_query(
            commit_sha=record["commit_sha"],
            dataset=record["dataset"],
            dataset_variant=record.get("dataset_variant"),
            scale_factor=record.get("scale_factor"),
            query_idx=record["query_idx"],
            storage=record["storage"],
            engine=record["engine"],
            format=record["format"],
        )
    if kind == "compression_time":
        return mid_mod.measurement_id_compression_time(
            commit_sha=record["commit_sha"],
            dataset=record["dataset"],
            dataset_variant=record.get("dataset_variant"),
            format=record["format"],
            op=record["op"],
        )
    if kind == "compression_size":
        return mid_mod.measurement_id_compression_size(
            commit_sha=record["commit_sha"],
            dataset=record["dataset"],
            dataset_variant=record.get("dataset_variant"),
            format=record["format"],
        )
    if kind == "random_access_time":
        return mid_mod.measurement_id_random_access(
            commit_sha=record["commit_sha"],
            dataset=record["dataset"],
            format=record["format"],
        )
    if kind == "vector_search_run":
        return mid_mod.measurement_id_vector_search(
            commit_sha=record["commit_sha"],
            dataset=record["dataset"],
            layout=record["layout"],
            flavor=record["flavor"],
            threshold=record["threshold"],
        )
    raise ValueError(f"unknown record kind {kind!r}")


def value_mismatches(db_row: dict, record: dict) -> list[str]:
    """Value columns the record carries that disagree with the DB row.

    Only columns present in ``record`` are compared; the comparison is exact
    (lists for ``all_runtimes_ns`` compare element-wise and order-sensitively).
    """
    kind = record["kind"]
    out: list[str] = []
    for col in _VALUE_COLUMNS[kind]:
        if col not in record:
            continue
        expected = record[col]
        actual = db_row.get(col)
        if actual != expected:
            out.append(f"{kind}.{col}: db={actual!r} != envelope={expected!r}")
    return out


class CrossCheckReport:
    """Outcome of one cross-check run."""

    def __init__(self) -> None:
        self.records = 0
        self.inserted = 0
        self.updated = 0
        self.problems: list[str] = []

    def is_clean(self) -> bool:
        """True iff every record UPDATEd a seeded row and all values round-tripped."""
        return self.inserted == 0 and not self.problems

    def __str__(self) -> str:
        head = f"cross-check: {self.records} records, {self.updated} updated, {self.inserted} inserted"
        if self.is_clean():
            return head + " -- CLEAN (writer UPDATEd seeded rows; values round-trip)"
        return "\n".join([head + " -- FAILED", *(f"  - {p}" for p in self.problems)])


def cross_check(conn, commit: dict, records: list[dict]) -> CrossCheckReport:
    """Run the writer over ``records`` (whose dim tuples must already be seeded) and
    confirm it UPDATEs the seeded rows -- never duplicate-INSERTs -- with the value
    columns round-tripping.

    ``conn`` is a live psycopg connection (the kind ``ingest_postgres`` accepts).
    """
    post_ingest = _load_module("post-ingest.py", "post_ingest")
    mid_mod = _load_module("_measurement_id.py", "_measurement_id")

    report = CrossCheckReport()
    report.records = len(records)
    expected = [(measurement_id_for(mid_mod, record), record) for record in records]

    inserted, updated = post_ingest.ingest_postgres(conn, commit, records)
    report.inserted = inserted
    report.updated = updated
    if inserted:
        report.problems.append(
            f"{inserted} of {len(records)} records INSERTed where an UPDATE was expected: "
            "a computed measurement_id did not match a seeded row, so the writer created a "
            "DUPLICATE instead of upserting the seeded row"
        )

    for mid, record in expected:
        table = _TABLE_BY_KIND[record["kind"]]
        cols = [c for c in _VALUE_COLUMNS[record["kind"]] if c in record]
        row = conn.execute(
            f"SELECT {', '.join(cols)} FROM {table} WHERE measurement_id = %s",
            (mid,),
        ).fetchone()
        if row is None:
            report.problems.append(f"{record['kind']} measurement_id {mid}: no row found after ingest")
            continue
        report.problems.extend(value_mismatches(dict(zip(cols, row)), record))

    return report


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-check the Python --postgres writer.")
    parser.add_argument(
        "--postgres",
        required=True,
        help="RDS DSN (verify-full TLS; connects as the bench_ingest role).",
    )
    parser.add_argument(
        "--region",
        help="AWS region for RDS IAM token minting (when the DSN carries no password).",
    )
    parser.add_argument(
        "--envelopes",
        required=True,
        type=Path,
        help='JSON file: {"commit": {...}, "records": [...]} in the v3 envelope shape.',
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    post_ingest = _load_module("post-ingest.py", "post_ingest")
    envelope = json.loads(args.envelopes.read_text())
    commit, records = envelope["commit"], envelope["records"]
    if not records:
        print("cross-check: envelope has no records", file=sys.stderr)
        return 2
    conn = post_ingest.connect_postgres(args.postgres, args.region)
    try:
        report = cross_check(conn, commit, records)
    finally:
        conn.close()
    print(report)
    return 0 if report.is_clean() else 1


if __name__ == "__main__":
    raise SystemExit(main())
