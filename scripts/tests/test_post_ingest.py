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
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
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


@pytest.fixture
def clock(monkeypatch, writer) -> SimpleNamespace:
    state = SimpleNamespace(now=0.0, sleeps=[])

    def sleep(delay):
        state.sleeps.append(delay)
        state.now += delay

    monkeypatch.setattr(writer.time, "monotonic", lambda: state.now)
    monkeypatch.setattr(writer.time, "sleep", sleep)
    monkeypatch.setattr(writer.random, "uniform", lambda lower, upper: upper)
    return state


@pytest.mark.parametrize(
    "invalid", [None, {"extra": 1}, {"iterations": True}, {"threshold": float("inf")}, {"commit_sha": "c" * 40}]
)
def test_late_invalid_record_never_starts_transaction(writer, commit, records, invalid):
    records[-1] = None if invalid is None else {**records[-1], **invalid}
    # A connection with no methods proves validation precedes even BEGIN, including the commit upsert.
    with pytest.raises(SystemExit, match="record 4"):
        writer.ingest_postgres(object(), commit, records)


def test_invalid_commit_sha_never_starts_transaction(writer, commit):
    commit["sha"] = "not-a-sha"
    with pytest.raises(SystemExit, match="commit SHA"):
        writer.ingest_postgres(object(), commit, [])


@pytest.mark.parametrize("error", [errors.DeadlockDetected, errors.SerializationFailure])
def test_conflict_attempts_and_backoff_are_bounded(writer, clock, error, capsys):
    calls = 0

    def fail():
        nonlocal calls
        calls += 1
        raise error("sensitive server detail")

    with pytest.raises(error):
        writer._retry_write_conflicts(fail)
    assert calls == 8
    assert clock.sleeps == [1, 2, 4, 8, 10, 10, 10]
    diagnostics = capsys.readouterr().err
    assert "attempt=8 elapsed=45.000s" in diagnostics
    assert f"sqlstate={error.sqlstate} outcome=exhausted" in diagnostics
    assert "sensitive server detail" not in diagnostics


@pytest.mark.parametrize("elapsed, sleeps", [(119.5, [0.5]), (121, [])])
def test_elapsed_budget_prevents_another_attempt(writer, clock, elapsed, sleeps):
    calls = 0

    def fail():
        nonlocal calls
        calls += 1
        clock.now += elapsed
        raise errors.DeadlockDetected()

    with pytest.raises(errors.DeadlockDetected):
        writer._retry_write_conflicts(fail)
    assert calls == 1
    assert clock.sleeps == sleeps


def test_budget_does_not_cancel_successful_transaction(writer, clock):
    def succeed() -> tuple[int, int]:
        clock.now += 121
        return (5, 0)

    assert writer._retry_write_conflicts(succeed) == (5, 0)
    assert clock.sleeps == []


@pytest.mark.parametrize(
    "error", [errors.CheckViolation, errors.LockNotAvailable, errors.QueryCanceled, psycopg.OperationalError]
)
def test_other_database_errors_are_not_retried(writer, clock, error, capsys):
    def fail():
        raise error("sensitive server detail")

    with pytest.raises(error):
        writer._retry_write_conflicts(fail)
    assert clock.sleeps == []
    assert "outcome=failed" in capsys.readouterr().err


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
            "options='-c statement_timeout=0 -c lock_timeout=0 -c search_path=other'",
            None,
        )
        is connection
    )
    assert captured["connect_timeout"] == 10
    assert captured["sslmode"] == "verify-full"
    assert captured["options"].endswith("-c search_path=public -c statement_timeout=30000 -c lock_timeout=10000")


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


@pytest.fixture
def conn(database):
    with psycopg.connect(database, autocommit=True) as connection:
        connection.execute(
            sql.SQL("TRUNCATE commits, {} CASCADE").format(sql.SQL(", ").join(map(sql.Identifier, TABLES)))
        )
        connection.execute("SET ROLE bench_ingest")
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


def test_late_database_error_rolls_back_entire_file(writer, database, conn, commit, records, capsys):
    assert writer.ingest_postgres(conn, commit, records) == (5, 0)
    before = stored_rows(conn)
    commit["message"] = "must roll back"
    records[0]["value_ns"] = 999
    records.insert(1, {**records[0], "query_idx": 2})
    records[-1]["value_ns"] = 1337
    with psycopg.connect(database, autocommit=True) as admin:
        admin.execute("ALTER TABLE vector_search_runs ADD CONSTRAINT reject_test_value CHECK (value_ns <> 1337)")
        try:
            with pytest.raises(errors.CheckViolation):
                writer.ingest_postgres(conn, commit, records)
        finally:
            admin.execute("ALTER TABLE vector_search_runs DROP CONSTRAINT reject_test_value")
    assert stored_rows(conn) == before
    assert conn.execute("SELECT message FROM commits").fetchone() == ("benchmark",)
    assert "attempt=1" in capsys.readouterr().err
    assert conn.info.transaction_status == psycopg.pq.TransactionStatus.IDLE


@pytest.mark.parametrize("conflict", ["deadlock", "serialization"])
def test_real_transaction_conflict_retries_the_entire_file(
    writer, database, conn, commit, records, monkeypatch, conflict, capsys
):
    other_commit = {**commit, "sha": "c" * 40}
    writer.ingest_postgres(conn, other_commit, [])
    writer.ingest_postgres(conn, commit, [])
    reached = Event()
    original: Callable[..., bool] = writer._APPLY_RECORD["query_measurement"]
    attempts = 0
    with psycopg.connect(database, autocommit=True) as other:
        if conflict == "serialization":
            conn.execute("SET default_transaction_isolation = 'serializable'")
        else:
            # The writer detects the cycle first and becomes the deadlock victim deterministically.
            conn.execute("RESET ROLE")
            conn.execute("SET deadlock_timeout = '100ms'")
            conn.execute("SET ROLE bench_ingest")
            other.execute("SET deadlock_timeout = '5s'")
            other.execute("BEGIN")
            other.execute("SELECT 1 FROM commits WHERE commit_sha = %s FOR UPDATE", (other_commit["sha"],))

        def insert_and_conflict(connection, mid_mod, record) -> bool:
            nonlocal attempts
            attempts += 1
            result = original(connection, mid_mod, record)
            if attempts == 1:
                if conflict == "serialization":
                    other.execute(
                        "UPDATE commits SET message = 'concurrent' WHERE commit_sha = %s", (other_commit["sha"],)
                    )
                reached.set()
                connection.execute("SELECT 1 FROM commits WHERE commit_sha = %s FOR UPDATE", (other_commit["sha"],))
            return result

        def complete_deadlock():
            assert reached.wait(timeout=10)
            other.execute("UPDATE commits SET message = 'concurrent' WHERE commit_sha = %s", (commit["sha"],))
            other.execute("COMMIT")

        monkeypatch.setitem(writer._APPLY_RECORD, "query_measurement", insert_and_conflict)
        monkeypatch.setattr(writer.random, "uniform", lambda lower, upper: lower)
        with ThreadPoolExecutor(max_workers=1) as pool:
            blocker = pool.submit(complete_deadlock) if conflict == "deadlock" else None
            assert writer.ingest_postgres(conn, commit, records) == (5, 0)
            if blocker:
                blocker.result(timeout=10)
    assert attempts == 2
    assert all(len(rows) == 1 for rows in stored_rows(conn))
    assert conn.execute("SELECT message FROM commits WHERE commit_sha = %s", (commit["sha"],)).fetchone() == (
        "benchmark",
    )
    diagnostics = capsys.readouterr().err
    sqlstate = "40P01" if conflict == "deadlock" else "40001"
    assert f"sqlstate={sqlstate} outcome=retry" in diagnostics
    assert "attempt=2" in diagnostics


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
