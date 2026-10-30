# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Regression tests for the ordering guarantees of `scripts/post-ingest.py`."""

import importlib.util
from contextlib import nullcontext
from copy import deepcopy
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
        "branch": "develop",
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
    expected_records = deepcopy(records)
    applied = []

    def apply(connection, mid, record) -> bool:
        applied.append((record["kind"], mid, record))
        return False

    for kind in writer._APPLY_RECORD:
        monkeypatch.setitem(writer._APPLY_RECORD, kind, apply)
    connection = SimpleNamespace(transaction=nullcontext, execute=lambda *args: None)
    # Every input order must produce the same row lock order, or overlapping writers could deadlock.
    assert writer.ingest_postgres(connection, commit, records[::-1]) == (len(records), 0)
    assert sorted(
        (record for _, _, record in applied), key=lambda record: (record["kind"], record.get("query_idx", 0))
    ) == sorted(expected_records, key=lambda record: (record["kind"], record.get("query_idx", 0)))
    keys = [(kind, mid) for kind, mid, _ in applied]
    assert keys == sorted(keys)


class _Transient(Exception):
    pass


def test_retries_transient_errors_until_success(writer):
    calls: list[int] = []
    sleeps: list[float] = []

    def attempt() -> str:
        calls.append(1)
        if len(calls) < 3:
            raise _Transient("connection refused")
        return "written"

    result = writer._with_retries(
        attempt, should_retry=lambda e: isinstance(e, _Transient), delays=(5, 10, 20), sleep=sleeps.append
    )
    assert result == "written"
    assert len(calls) == 3
    assert sleeps == [5, 10]


def test_raises_the_last_error_when_retries_run_out(writer):
    sleeps: list[float] = []

    def attempt() -> None:
        raise _Transient("connection refused")

    with pytest.raises(_Transient):
        writer._with_retries(
            attempt, should_retry=lambda e: isinstance(e, _Transient), delays=(5, 10), sleep=sleeps.append
        )
    assert sleeps == [5, 10]


def test_does_not_retry_other_errors(writer):
    sleeps: list[float] = []

    def attempt() -> None:
        raise ValueError("bad record")

    with pytest.raises(ValueError):
        writer._with_retries(attempt, should_retry=lambda e: isinstance(e, _Transient), sleep=sleeps.append)
    assert sleeps == []


def test_retry_schedule_spans_an_rds_restart(writer):
    # An OS update on the instance refused connections for about 3 minutes.
    assert sum(writer._RETRY_DELAYS_S) >= 180
