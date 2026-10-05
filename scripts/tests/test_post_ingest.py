# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Regression tests for the ordering guarantees of `scripts/post-ingest.py`."""

import importlib.util
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def writer() -> ModuleType:
    spec = importlib.util.spec_from_file_location("post_ingest", REPO_ROOT / "scripts/post-ingest.py")
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def commit() -> dict[str, str]:
    return {
        "sha": "a" * 40,
        "timestamp": "2026-09-08T12:00:00Z",
        "message": "benchmark",
        "author_name": "Author",
        "author_email": "author@example.com",
        "committer_name": "Committer",
        "committer_email": "committer@example.com",
        "tree_sha": "b" * 40,
        "url": "https://github.com/vortex-data/vortex/commit/" + "a" * 40,
    }


@pytest.fixture
def records(commit) -> list[dict[str, object]]:
    common = {"commit_sha": commit["sha"], "dataset": "test", "format": "vortex"}
    timing = {"value_ns": 100, "all_runtimes_ns": [99, 101]}
    return [
        {**common, **timing, "kind": "query_measurement", "query_idx": 1, "storage": "nvme", "engine": "datafusion"},
        {**common, **timing, "kind": "compression_time", "op": "encode"},
        {**common, "kind": "compression_size", "value_bytes": 10, "uncompressed_bytes": 100},
        {**common, **timing, "kind": "random_access_time", "open_mode": "cached"},
        {
            "kind": "vector_search_run",
            "commit_sha": commit["sha"],
            "dataset": "test",
            "layout": "flat",
            "flavor": "exact",
            "threshold": 0.5,
            **timing,
            "matches": 1,
            "rows_scanned": 10,
            "bytes_scanned": 100,
            "iterations": 2,
        },
    ]


@pytest.mark.parametrize(
    "invalid", [None, {"extra": 1}, {"iterations": True}, {"threshold": float("inf")}, {"commit_sha": "c" * 40}]
)
def test_late_invalid_record_never_starts_transaction(writer, commit, records, invalid):
    records[-1] = None if invalid is None else {**records[-1], **invalid}
    # A connection with no methods proves validation precedes even BEGIN, including the commit upsert.
    with pytest.raises(SystemExit, match="record 4"):
        writer.ingest_postgres(object(), commit, records)


def test_fact_rows_apply_in_lock_order(writer, commit, records, monkeypatch):
    records += [{**records[0], "query_idx": idx} for idx in (2, 3)]
    applied = []

    def apply(connection, mid, record) -> bool:
        applied.append((record["kind"], mid))
        return False

    for kind in writer._APPLY_RECORD:
        monkeypatch.setitem(writer._APPLY_RECORD, kind, apply)
    connection = SimpleNamespace(transaction=nullcontext, execute=lambda *args: None)
    # Every input order must produce the same row lock order, or overlapping writers could deadlock.
    assert writer.ingest_postgres(connection, commit, records[::-1]) == (len(records), 0)
    assert len(applied) == len(records)
    assert applied == sorted(applied)
