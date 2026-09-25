# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Writer regressions, including real PostgreSQL transactions against the website migrations.

Set BENCH_TEST_POSTGRES_DSN to a disposable superuser database and BENCH_WEBSITE_DIR to a website
checkout for integration tests. The fixture creates and drops its own database and invokes the
website's migration runner. CI pins that checkout in .github/workflows/ci.yml.
"""

import importlib.util
import os
import subprocess
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from threading import Event
from types import ModuleType, SimpleNamespace

import psycopg
import pytest
from psycopg import errors, sql
from psycopg.conninfo import make_conninfo

REPO_ROOT = Path(__file__).resolve().parents[2]
TABLES = ("query_measurements", "compression_times", "compression_sizes", "random_access_times", "vector_search_runs")


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


def test_connection_timeouts_override_dsn_options(writer, monkeypatch):
    captured = {}
    connection = SimpleNamespace(pgconn=SimpleNamespace(ssl_in_use=True))

    def connect(**kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        return connection

    monkeypatch.setattr(psycopg, "connect", connect)
    assert (
        writer.connect_postgres(
            "host=example.com user=bench_ingest password=secret connect_timeout=0 "
            "options='-c statement_timeout=0 -c search_path=other'",
            None,
        )
        is connection
    )
    assert captured["connect_timeout"] == 10
    assert captured["sslmode"] == "verify-full"
    assert captured["options"].endswith("-c search_path=public -c statement_timeout=30000")


@pytest.fixture(scope="module")
def database():
    dsn = os.environ.get("BENCH_TEST_POSTGRES_DSN")
    website = os.environ.get("BENCH_WEBSITE_DIR")
    if not dsn or not website:
        pytest.skip("set BENCH_TEST_POSTGRES_DSN and BENCH_WEBSITE_DIR to run PostgreSQL writer tests")
    name = "writer_test_" + uuid.uuid4().hex
    with psycopg.connect(dsn, autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
        target = make_conninfo(dsn, dbname=name)
        try:
            subprocess.run(
                [sys.executable, str(Path(website) / "scripts/migrate-schema.py"), "apply", "--target", target],
                check=True,
                capture_output=True,
                text=True,
            )
            yield target
        finally:
            admin.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(name)))


def ingest_connection(database):
    connection = psycopg.connect(database, autocommit=True)
    connection.execute("SET ROLE bench_ingest")
    return connection


@pytest.fixture
def conn(database):
    with psycopg.connect(database, autocommit=True) as admin:
        admin.execute(sql.SQL("TRUNCATE commits, {} CASCADE").format(sql.SQL(", ").join(map(sql.Identifier, TABLES))))
    with ingest_connection(database) as connection:
        yield connection


def stored_rows(conn) -> list[list[tuple[object, ...]]]:
    return [
        conn.execute(sql.SQL("SELECT * FROM {} ORDER BY measurement_id").format(sql.Identifier(table))).fetchall()
        for table in TABLES
    ]


def test_replay_updates_every_family_without_duplicate_ids(writer, conn, commit, records):
    assert writer.ingest_postgres(conn, commit, records) == (5, 0)
    before = stored_rows(conn)
    for record in records:
        if "value_ns" in record:
            record.update(value_ns=200, all_runtimes_ns=[199, 201])
        else:
            record.update(value_bytes=20, uncompressed_bytes=200)
    assert writer.ingest_postgres(conn, commit, records + records) == (0, 10)
    after = stored_rows(conn)
    assert [rows[0][0] for rows in after] == [rows[0][0] for rows in before]
    assert all(len(rows) == 1 for rows in after)
    assert all(old != new for old, new in zip(before, after, strict=True))
    assert conn.execute("SELECT count(*) FROM commits").fetchone() == (1,)
    assert writer.ingest_postgres(conn, commit, records) == (0, 5)
    assert stored_rows(conn) == after


def test_late_database_error_rolls_back_every_fact_row(writer, database, conn, commit, records):
    assert writer.ingest_postgres(conn, commit, records) == (5, 0)
    before = stored_rows(conn)
    commit["message"] = "updated"
    records[0]["value_ns"] = 999
    records.insert(1, {**records[0], "query_idx": 2})
    # vector_search_runs sorts last, so every other fact row is written before this one fails.
    records[-1]["value_ns"] = 1337
    with psycopg.connect(database, autocommit=True) as admin:
        admin.execute("ALTER TABLE vector_search_runs ADD CONSTRAINT reject_test_value CHECK (value_ns <> 1337)")
        try:
            with pytest.raises(errors.CheckViolation):
                writer.ingest_postgres(conn, commit, records)
        finally:
            admin.execute("ALTER TABLE vector_search_runs DROP CONSTRAINT reject_test_value")
    assert stored_rows(conn) == before
    # The commit row commits in its own transaction, before the fact rows.
    assert conn.execute("SELECT message FROM commits").fetchone() == ("updated",)
    assert conn.info.transaction_status == psycopg.pq.TransactionStatus.IDLE


def test_writers_for_the_same_commit_do_not_wait_on_each_other(writer, database, conn, commit, records, monkeypatch):
    held_records, other_records = records[:2], records[2:]
    inside = Event()
    release = Event()
    original = writer._APPLY_RECORD["query_measurement"]

    def hold_transaction_open(connection, mid, record) -> bool:
        result = original(connection, mid, record)
        inside.set()
        assert release.wait(timeout=10)
        return result

    monkeypatch.setitem(writer._APPLY_RECORD, "query_measurement", hold_transaction_open)
    with ThreadPoolExecutor(max_workers=1) as pool:
        held = pool.submit(writer.ingest_postgres, conn, commit, held_records)
        assert inside.wait(timeout=10)
        try:
            with ingest_connection(database) as other:
                # Fail fast instead of hanging if this writer queues behind the held transaction.
                other.execute("SET lock_timeout = '2s'")
                assert writer.ingest_postgres(other, commit, other_records) == (3, 0)
        finally:
            release.set()
        assert held.result(timeout=10) == (2, 0)
    assert all(len(rows) == 1 for rows in stored_rows(conn))


def test_refresh_failure_keeps_successful_ingest(writer, database, conn, commit, records, monkeypatch, capsys):
    monkeypatch.setattr(writer, "read_records", lambda path: records)
    monkeypatch.setattr(writer, "build_commit", lambda *args: commit)
    monkeypatch.setattr(writer, "connect_postgres", lambda *args: psycopg.connect(database))
    monkeypatch.setenv("BENCH_SITE_BASE_URL", "https://bench.example.com")
    monkeypatch.setenv("BENCH_REVALIDATE_TOKEN", "secret")

    def fail_refresh(*args: object) -> None:
        raise TimeoutError("refresh timed out")

    monkeypatch.setattr(writer, "_http", fail_refresh)
    args = SimpleNamespace(
        jsonl_path=None, commit_sha=commit["sha"], repo_url=None, git_dir=None, postgres=None, region=None, timeout=1
    )
    assert writer._main_postgres(args) == 0
    assert all(len(rows) == 1 for rows in stored_rows(conn))
    output = capsys.readouterr()
    assert '"inserted":5,"updated":0' in output.out
    assert "warning: cache revalidate failed" in output.err
