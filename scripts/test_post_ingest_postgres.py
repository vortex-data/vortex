# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Integration + unit tests for `post-ingest.py`'s `--postgres` (v4) mode.

The testcontainer tests apply the real `migrations/` against a vanilla
`postgres:16-alpine` (via the PR-1.2 migrate runner) and exercise
`post_ingest.ingest_postgres` against the resulting 6-table schema. They pin the
migration's load-bearing behavior-preservation invariants:

- insert-vs-update accounting (re-ingesting an identical envelope upserts: 0
  inserted, N updated; row counts unchanged);
- `ON CONFLICT (measurement_id) DO UPDATE` overwrites value columns while
  leaving the dim tuple (hence the `measurement_id`) untouched;
- every stored `measurement_id` matches the Python port in `_measurement_id.py`;
- a non-finite f64 dim (`vector_search_run.threshold`) raises loudly and writes
  nothing (the transaction rolls back);
- the v3 server's `deny_unknown_fields`, `storage IN ('nvme','s3')`,
  memory-quartet, and envelope-commit-sha validations are reproduced.

The non-Docker unit tests cover `SCHEMA_VERSION` lockstep with the Rust source
of truth, RDS hostname region parsing, and the IAM-token-vs-DSN-password branch
of `connect_postgres` (boto3 + psycopg mocked).

`post-ingest.py` and `migrate-schema.py` are loaded by file path because the
hyphen in their names blocks `import`.
"""

import importlib.util
import inspect
import json
import os
import re
import sys
import types
from collections.abc import Iterator
from pathlib import Path

import psycopg
import pytest
from testcontainers.postgres import PostgresContainer

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
REPO_MIGRATIONS_DIR = REPO_ROOT / "migrations"

COMMIT_SHA = "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0"

_FACT_TABLES = [
    "query_measurements",
    "compression_times",
    "compression_sizes",
    "random_access_times",
    "vector_search_runs",
]

_TABLE_BY_KIND = {
    "query_measurement": "query_measurements",
    "compression_time": "compression_times",
    "compression_size": "compression_sizes",
    "random_access_time": "random_access_times",
    "vector_search_run": "vector_search_runs",
}


def _load_module(filename: str, modname: str):
    """Load a sibling script by file path (the hyphen blocks `import`)."""
    path = SCRIPTS_DIR / filename
    spec = importlib.util.spec_from_file_location(modname, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


post_ingest = _load_module("post-ingest.py", "post_ingest")
migrate_runner = _load_module("migrate-schema.py", "migrate_schema")
cross_check_mod = _load_module("cross_check_python_writer.py", "cross_check_python_writer")


def _sample_commit() -> dict:
    return {
        "sha": COMMIT_SHA,
        "timestamp": "2026-01-02T03:04:05+00:00",
        "message": "feat: a sample commit (#123)",
        "author_name": "Ada Lovelace",
        "author_email": "ada@example.com",
        "committer_name": "Ada Lovelace",
        "committer_email": "ada@example.com",
        "tree_sha": "0123456789abcdef0123456789abcdef01234567",
        "url": "https://github.com/vortex-data/vortex/commit/" + COMMIT_SHA,
    }


def _sample_records() -> list[dict]:
    """One record per fact-table kind, exercising every measurement_id path.

    `query_measurement` carries no memory quartet (all four NULL); the optional
    string dims are a mix of present and absent to exercise `write_opt_str`.
    """
    return [
        {
            "kind": "query_measurement",
            "commit_sha": COMMIT_SHA,
            "dataset": "tpch",
            "scale_factor": "1",
            "query_idx": 7,
            "storage": "nvme",
            "engine": "vortex",
            "format": "parquet",
            "value_ns": 1230,
            "all_runtimes_ns": [1230, 1300, 1180],
        },
        {
            "kind": "compression_time",
            "commit_sha": COMMIT_SHA,
            "dataset": "taxi",
            "format": "vortex",
            "op": "encode",
            "value_ns": 500,
            "all_runtimes_ns": [500, 520],
        },
        {
            "kind": "compression_size",
            "commit_sha": COMMIT_SHA,
            "dataset": "taxi",
            "format": "vortex",
            "value_bytes": 99999,
        },
        {
            "kind": "random_access_time",
            "commit_sha": COMMIT_SHA,
            "dataset": "chimp",
            "format": "parquet",
            "value_ns": 200,
            "all_runtimes_ns": [200, 210],
        },
        {
            "kind": "vector_search_run",
            "commit_sha": COMMIT_SHA,
            "dataset": "cohere-large-10m",
            "layout": "flat",
            "flavor": "raw",
            "threshold": 0.8,
            "value_ns": 700,
            "all_runtimes_ns": [700, 720],
            "matches": 5,
            "rows_scanned": 100,
            "bytes_scanned": 2048,
            "iterations": 3,
        },
    ]


def _expected_mid(mid_mod, rec: dict) -> int:
    """Recompute a record's measurement_id independently of the writer."""
    kind = rec["kind"]
    if kind == "query_measurement":
        return mid_mod.measurement_id_query(
            commit_sha=rec["commit_sha"],
            dataset=rec["dataset"],
            dataset_variant=rec.get("dataset_variant"),
            scale_factor=rec.get("scale_factor"),
            query_idx=rec["query_idx"],
            storage=rec["storage"],
            engine=rec["engine"],
            format=rec["format"],
        )
    if kind == "compression_time":
        return mid_mod.measurement_id_compression_time(
            commit_sha=rec["commit_sha"],
            dataset=rec["dataset"],
            dataset_variant=rec.get("dataset_variant"),
            format=rec["format"],
            op=rec["op"],
        )
    if kind == "compression_size":
        return mid_mod.measurement_id_compression_size(
            commit_sha=rec["commit_sha"],
            dataset=rec["dataset"],
            dataset_variant=rec.get("dataset_variant"),
            format=rec["format"],
        )
    if kind == "random_access_time":
        return mid_mod.measurement_id_random_access(
            commit_sha=rec["commit_sha"],
            dataset=rec["dataset"],
            format=rec["format"],
        )
    if kind == "vector_search_run":
        return mid_mod.measurement_id_vector_search(
            commit_sha=rec["commit_sha"],
            dataset=rec["dataset"],
            layout=rec["layout"],
            flavor=rec["flavor"],
            threshold=rec["threshold"],
        )
    raise AssertionError(f"unhandled kind {kind!r}")


def _count(conn: psycopg.Connection, table: str) -> int:
    return conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]


def _docker_available() -> bool:
    import subprocess

    try:
        result = subprocess.run(["docker", "info"], capture_output=True, timeout=5, check=False)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _require_docker_for_testcontainers() -> None:
    # The testcontainer suite MUST run in CI: a missing Docker daemon is a HARD FAILURE there,
    # not a silent skip. A green CI job with skipped testcontainer tests would let a Postgres-writer
    # regression merge undetected, which is exactly what wiring `scripts/` into CI (PR-2.3) closes.
    # Locally (no CI env) we skip so developers without Docker can still run the pure-unit tests.
    if _docker_available():
        return
    if os.environ.get("CI"):
        pytest.fail("Docker unavailable in CI (`docker info` failed); the testcontainer suite must run, not skip")
    pytest.skip("Docker not running; skipping Postgres testcontainer tests")


@pytest.fixture(scope="module")
def postgres_dsn() -> Iterator[str]:
    """Module-scoped Postgres testcontainer; yields a libpq DSN.

    `BENCH_TEST_PG_DSN`, when set, is used verbatim instead of spinning up a
    container (a superuser DSN against any throwaway Postgres). Otherwise a
    `postgres:16-alpine` testcontainer is used, skipped locally when Docker is
    unavailable but FAILED in CI (where the `CI` env var is set) so the
    testcontainer suite can never silently skip on the CI runner.
    """
    override = os.environ.get("BENCH_TEST_PG_DSN")
    if override:
        yield override
        return
    _require_docker_for_testcontainers()
    with PostgresContainer("postgres:16-alpine") as container:
        host = container.get_container_host_ip()
        port = container.get_exposed_port(5432)
        dsn = f"postgresql://{container.username}:{container.password}@{host}:{port}/{container.dbname}"
        yield dsn


@pytest.fixture
def schema_conn(postgres_dsn: str) -> Iterator[psycopg.Connection]:
    """A connection with the real migrations freshly applied to empty tables.

    The container is module-scoped, so each test scrubs the `public` schema and
    re-applies `migrations/` (role creation is `IF NOT EXISTS`-guarded, so the
    re-apply is idempotent) to start from empty fact tables.
    """
    with psycopg.connect(postgres_dsn) as conn:
        conn.autocommit = True
        # Guard the destructive scrub for the BENCH_TEST_PG_DSN OVERRIDE path only:
        # that DSN is operator-supplied and honored verbatim, so refuse to drop the
        # public schema unless the resolved host is loopback/socket. A fixture-spun
        # testcontainer is throwaway by construction, so it is exempt -- and its
        # get_container_host_ip() can be a non-loopback IP under TCP DOCKER_HOST /
        # DinD / DooD, where this guard would otherwise fail the whole suite.
        if os.environ.get("BENCH_TEST_PG_DSN"):
            host = conn.info.host
            if host and host not in {"localhost", "127.0.0.1", "::1"} and not host.startswith("/"):
                pytest.fail(
                    f"schema_conn refuses to scrub a non-local Postgres host {host!r}; "
                    "point BENCH_TEST_PG_DSN at a loopback/socket throwaway database."
                )
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS public._applied_migrations CASCADE")
            cur.execute(
                """
                DO $$
                DECLARE r RECORD;
                BEGIN
                    FOR r IN (SELECT tablename FROM pg_tables WHERE schemaname = 'public') LOOP
                        EXECUTE 'DROP TABLE IF EXISTS public.' || quote_ident(r.tablename) || ' CASCADE';
                    END LOOP;
                END$$;
                """
            )
        migrate_runner.apply(conn, REPO_MIGRATIONS_DIR)
        yield conn


def test_ingest_inserts_then_updates(schema_conn: psycopg.Connection) -> None:
    commit = _sample_commit()
    records = _sample_records()

    inserted, updated = post_ingest.ingest_postgres(schema_conn, commit, records)
    assert (inserted, updated) == (len(records), 0)

    # Re-ingesting the identical envelope upserts every row: 0 new, N updated.
    inserted2, updated2 = post_ingest.ingest_postgres(schema_conn, commit, records)
    assert (inserted2, updated2) == (0, len(records))

    # One row per fact table, and exactly one commit row (no duplicates).
    assert _count(schema_conn, "commits") == 1
    for kind, table in _TABLE_BY_KIND.items():
        assert _count(schema_conn, table) == 1, f"{kind} -> {table} row count drifted"


def test_query_measurement_insert_populates_commit_timestamp(schema_conn: psycopg.Connection) -> None:
    """The denormalized `commit_timestamp` (migration 006, the read path's latest-per-series sort
    key) is stamped from the envelope's `commits` row on BOTH upsert paths, so rows written by this
    writer never depend on the post-deploy re-backfill."""
    commit = _sample_commit()
    records = [r for r in _sample_records() if r["kind"] == "query_measurement"]
    assert records, "expected at least one query_measurement sample record"

    def _unstamped_or_drifted() -> int:
        # Count rows that are NOT stamped to their commit's timestamp -- 0 means every
        # query_measurements row was correctly stamped (not just an arbitrary fetchone row).
        return schema_conn.execute(
            "SELECT count(*) FROM query_measurements q JOIN commits c USING (commit_sha)"
            " WHERE q.commit_timestamp IS NULL OR q.commit_timestamp <> c.timestamp"
        ).fetchone()[0]

    post_ingest.ingest_postgres(schema_conn, commit, records)
    assert _count(schema_conn, "query_measurements") == len(records)
    assert _unstamped_or_drifted() == 0

    # The update path re-stamps as well: scrub the column, re-ingest the same envelope (an
    # ON CONFLICT DO UPDATE), and every row's timestamp must come back.
    schema_conn.execute("UPDATE query_measurements SET commit_timestamp = NULL")
    post_ingest.ingest_postgres(schema_conn, commit, records)
    assert _unstamped_or_drifted() == 0


# Per-kind mutation of every value/side-counter/env column each table's ON
# CONFLICT DO UPDATE SET list owns (dim columns deliberately excluded). Mirrors
# the SET lists in benchmarks-website/server/src/ingest.rs; a stale/incorrect SET
# list in any table is caught by the parametrized update test below.
# `commit_timestamp` is also in the query_measurements SET list but is DERIVED
# from `commits` (not a record field), so it is pinned by
# `test_query_measurement_insert_populates_commit_timestamp` instead of here.
_UPDATE_VALUE_MUTATIONS = {
    "query_measurement": {
        "value_ns": 4242,
        "all_runtimes_ns": [4242, 4243],
        "peak_physical": 11,
        "peak_virtual": 22,
        "physical_delta": 3,
        "virtual_delta": 4,
        "env_triple": "aarch64-linux-gnu",
    },
    "compression_time": {"value_ns": 4242, "all_runtimes_ns": [4242], "env_triple": "aarch64-linux-gnu"},
    "compression_size": {"value_bytes": 424242},
    "random_access_time": {"value_ns": 4242, "all_runtimes_ns": [4242], "env_triple": "aarch64-linux-gnu"},
    "vector_search_run": {
        "value_ns": 4242,
        "all_runtimes_ns": [4242],
        "matches": 9,
        "rows_scanned": 99,
        "bytes_scanned": 999,
        "iterations": 7,
        "env_triple": "aarch64-linux-gnu",
    },
}


@pytest.mark.parametrize("kind", list(_UPDATE_VALUE_MUTATIONS))
def test_update_overwrites_all_value_columns_per_table(schema_conn: psycopg.Connection, kind: str) -> None:
    """For every fact table, re-ingesting the same dim tuple with changed value
    columns updates in place: row count stays 1, measurement_id is unchanged, and
    every column the ON CONFLICT SET list owns is overwritten. Guards all five SET
    lists (not just query_measurements) against silent regression."""
    mid_mod = post_ingest._measurement_id_module()
    table = _TABLE_BY_KIND[kind]
    mutation = _UPDATE_VALUE_MUTATIONS[kind]
    commit = _sample_commit()
    rec = next(r for r in _sample_records() if r["kind"] == kind)

    post_ingest.ingest_postgres(schema_conn, commit, [rec])
    mid = _expected_mid(mid_mod, rec)

    # Same dim tuple, every value/side-counter/env column changed -> must UPDATE.
    mutated = dict(rec, **mutation)
    inserted, updated = post_ingest.ingest_postgres(schema_conn, commit, [mutated])
    assert (inserted, updated) == (0, 1)
    assert _count(schema_conn, table) == 1

    cols = list(mutation)
    row = schema_conn.execute(f"SELECT measurement_id, {', '.join(cols)} FROM {table}").fetchone()
    assert row[0] == mid, f"{kind}: measurement_id changed on update (dim tuple should be stable)"
    for i, col in enumerate(cols, start=1):
        assert row[i] == mutation[col], f"{kind}.{col} was not overwritten by ON CONFLICT DO UPDATE"


def test_same_dim_tuple_twice_in_one_envelope_counts_second_as_update(schema_conn: psycopg.Connection) -> None:
    """Two records with the same dim tuple in ONE envelope: the second is an
    UPDATE, not a second insert, and last-write-wins.

    Pins the RETURNING (xmax = 0) classifier's same-transaction behavior
    (empirically (1, 1), not (2, 0)): ON CONFLICT DO UPDATE locks the tuple the
    first record created earlier in the same transaction, stamping its xmax with
    the current txid, so the second is correctly classified as updated -- matching
    the v3 server's exists()-preflight semantics. Resolves the cycle-2 reviewer
    disagreement in favor of the empirical result."""
    commit = _sample_commit()
    base = next(r for r in _sample_records() if r["kind"] == "compression_size")
    second = dict(base, value_bytes=base["value_bytes"] + 1)  # same dims, new value

    inserted, updated = post_ingest.ingest_postgres(schema_conn, commit, [base, second])
    assert (inserted, updated) == (1, 1)
    assert _count(schema_conn, "compression_sizes") == 1
    stored = schema_conn.execute("SELECT value_bytes FROM compression_sizes").fetchone()[0]
    assert stored == second["value_bytes"]  # last-write-wins


def test_measurement_ids_match_python_port(schema_conn: psycopg.Connection) -> None:
    mid_mod = post_ingest._measurement_id_module()
    commit = _sample_commit()
    records = _sample_records()
    post_ingest.ingest_postgres(schema_conn, commit, records)

    for rec in records:
        table = _TABLE_BY_KIND[rec["kind"]]
        expected = _expected_mid(mid_mod, rec)
        n = schema_conn.execute(f"SELECT count(*) FROM {table} WHERE measurement_id = %s", (expected,)).fetchone()[0]
        assert n == 1, f"{rec['kind']}: no row stored under Python-computed measurement_id {expected}"


def test_empty_records_upserts_commit_only(schema_conn: psycopg.Connection) -> None:
    commit = _sample_commit()
    inserted, updated = post_ingest.ingest_postgres(schema_conn, commit, [])
    assert (inserted, updated) == (0, 0)
    assert _count(schema_conn, "commits") == 1
    for table in _FACT_TABLES:
        assert _count(schema_conn, table) == 0


# The four fact tables that carry all_runtimes_ns (compression_sizes has none).
_ARRAY_TABLE_KINDS = ["query_measurement", "compression_time", "random_access_time", "vector_search_run"]


@pytest.mark.parametrize("kind", _ARRAY_TABLE_KINDS)
def test_empty_all_runtimes_ns_roundtrips(schema_conn: psycopg.Connection, kind: str) -> None:
    # An empty all_runtimes_ns ([]) is a plausible producer output; psycopg sends
    # it as the untyped literal '{}', which only types correctly because of the
    # explicit ::bigint[] cast. Pin the cast for EVERY array-valued fact table so
    # a dropped ::bigint[] in any of them fails loudly.
    table = _TABLE_BY_KIND[kind]
    commit = _sample_commit()
    rec = next(r for r in _sample_records() if r["kind"] == kind)
    rec = dict(rec, all_runtimes_ns=[])
    inserted, updated = post_ingest.ingest_postgres(schema_conn, commit, [rec])
    assert (inserted, updated) == (1, 0)
    row = schema_conn.execute(f"SELECT all_runtimes_ns FROM {table}").fetchone()
    assert row[0] == []


@pytest.mark.parametrize("bad_runtimes", ["{}", [1, None], "not-a-list", [1, 2.5], [1, True]])
def test_malformed_all_runtimes_ns_rejected(schema_conn: psycopg.Connection, bad_runtimes: object) -> None:
    # all_runtimes_ns is presence-checked but the ::bigint[] cast is permissive:
    # the string "{}" parses to an empty array and [1, null] adapts to {1,NULL},
    # both of which the v3 Vec<i64> serde rejects. Content validation must make
    # these fail loud (record-indexed) and write nothing.
    commit = _sample_commit()
    rec = next(r for r in _sample_records() if r["kind"] == "compression_time")
    rec = dict(rec, all_runtimes_ns=bad_runtimes)
    with pytest.raises(SystemExit, match="array of integers"):
        post_ingest.ingest_postgres(schema_conn, commit, [rec])
    assert _count(schema_conn, "commits") == 0
    assert _count(schema_conn, "compression_times") == 0


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_threshold_raises_and_rolls_back(schema_conn: psycopg.Connection, bad: float) -> None:
    commit = _sample_commit()
    vsr = next(r for r in _sample_records() if r["kind"] == "vector_search_run")
    vsr = dict(vsr, threshold=bad)

    with pytest.raises(SystemExit, match="not a finite number"):
        post_ingest.ingest_postgres(schema_conn, commit, [vsr])

    # All-or-nothing: the commit upsert that ran before the bad record is rolled
    # back too, so nothing is written.
    assert _count(schema_conn, "vector_search_runs") == 0
    assert _count(schema_conn, "commits") == 0


# --- read_records / git_show_field unit tests (pure; no testcontainer needed) ---


def test_read_records_happy_path_mixed_newlines(tmp_path: Path) -> None:
    # Pin read_records for valid input: records split across mixed line endings (LF, CRLF, and bare
    # CR -- all handled by text-mode universal-newline translation), blank lines skipped, a
    # multi-byte UTF-8 string round-tripped, and a final line with no trailing newline.
    jsonl = tmp_path / "envelope.jsonl"
    jsonl.write_bytes(
        b'{"kind": "compression_size", "dataset": "a"}\n'  # LF
        b"\n"  # blank line -> skipped
        b'{"kind": "compression_size", "dataset": "b"}\r\n'  # CRLF
        b'{"kind": "compression_size", "dataset": "\xc3\xa9"}\r'  # bare CR + UTF-8 e-acute
        b'{"kind": "compression_size", "dataset": "d"}'  # no trailing newline
    )
    records = post_ingest.read_records(jsonl)
    assert [r["dataset"] for r in records] == ["a", "b", "é", "d"]


def test_read_records_rejects_malformed_json(tmp_path: Path) -> None:
    # A malformed JSON line fails loud with the path:line convention for debuggability.
    jsonl = tmp_path / "envelope.jsonl"
    jsonl.write_text('{"kind": "compression_size"\n', encoding="utf-8")
    with pytest.raises(SystemExit, match=r"envelope\.jsonl:1: invalid JSON"):
        post_ingest.read_records(jsonl)


def test_git_show_field_decodes_and_strips(monkeypatch) -> None:
    # Well-formed multi-byte UTF-8 metadata round-trips through subprocess text mode and is stripped
    # of the trailing newline git appends.
    def fake_run(*_a, **kwargs):
        assert kwargs.get("capture_output") is True, "git_show_field must capture stdout"
        assert kwargs.get("text") is True, "git_show_field decodes git output as text"
        return types.SimpleNamespace(stdout="café\n")

    monkeypatch.setattr(post_ingest.subprocess, "run", fake_run)
    assert post_ingest.git_show_field("0" * 40, "%an", None) == "café"


def test_require_docker_fails_loud_in_ci(monkeypatch) -> None:
    # In CI a missing Docker daemon must FAIL the testcontainer suite, not skip it: a green job with
    # skipped Postgres tests would let a writer regression merge undetected (the gap PR-2.3 closes).
    # Catch BOTH outcome types (Failed/Skipped both subclass BaseException) so an always-skip
    # regression -- the helper raising Skipped -- is caught HERE and asserted against, rather than
    # escaping a bare pytest.raises(Failed) and being recorded by pytest as a (green) test-skip.
    monkeypatch.setenv("CI", "true")
    monkeypatch.setattr(sys.modules[__name__], "_docker_available", lambda: False)
    with pytest.raises((pytest.fail.Exception, pytest.skip.Exception)) as exc_info:
        _require_docker_for_testcontainers()
    assert isinstance(exc_info.value, pytest.fail.Exception), "must FAIL (not skip) when Docker is absent in CI"


def test_require_docker_skips_without_ci(monkeypatch) -> None:
    # Locally (no CI env) a missing Docker daemon skips so the pure-unit tests still run. Symmetric to
    # the CI test: catch both outcome types and assert it is a skip (not a fail).
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setattr(sys.modules[__name__], "_docker_available", lambda: False)
    with pytest.raises((pytest.fail.Exception, pytest.skip.Exception)) as exc_info:
        _require_docker_for_testcontainers()
    assert isinstance(exc_info.value, pytest.skip.Exception), "must skip (not fail) when not in CI"


def test_retry_write_conflicts_retries_deadlock_then_succeeds() -> None:
    # A deadlock aborts one writer's transaction; _retry_write_conflicts must re-run the whole
    # transaction (mirroring the v3 server's retry_write_conflicts) until it succeeds, rather
    # than surfacing the transient conflict to the caller.
    calls = {"n": 0}

    def op() -> tuple[int, int]:
        calls["n"] += 1
        if calls["n"] < 3:
            raise psycopg.errors.DeadlockDetected("simulated deadlock")
        return (4, 2)

    assert post_ingest._retry_write_conflicts(op) == (4, 2)
    assert calls["n"] == 3


def test_retry_write_conflicts_propagates_validation_error() -> None:
    # A non-conflict error (e.g. a validation SystemExit) is deterministic and must propagate
    # immediately, not be retried.
    def op() -> tuple[int, int]:
        raise SystemExit("record 0 (compression_size): bad")

    with pytest.raises(SystemExit, match="bad"):
        post_ingest._retry_write_conflicts(op)


def test_retry_write_conflicts_gives_up_after_attempt_cap() -> None:
    # A transaction that conflicts on every attempt eventually re-raises the conflict (loud)
    # after exhausting _WRITE_CONFLICT_ATTEMPTS, rather than looping forever.
    calls = {"n": 0}

    def op() -> tuple[int, int]:
        calls["n"] += 1
        raise psycopg.errors.SerializationFailure("always conflicts")

    with pytest.raises(psycopg.errors.SerializationFailure):
        post_ingest._retry_write_conflicts(op)
    assert calls["n"] == post_ingest._WRITE_CONFLICT_ATTEMPTS


# Per fact-table _insert_ function: the dim columns that feed measurement_id and so MUST be
# EXCLUDED from the ON CONFLICT DO UPDATE SET clause (BAN: the upsert overwrites only value/env
# columns, never the dim tuple). commit_sha is intentionally NOT listed here: it is a dim-hash
# input but is included in the SET to match the v3 server exactly (a measurement_id conflict
# means the dims, incl. commit_sha, are identical, so the overwrite is a no-op).
_SET_EXCLUDED_DIM_COLUMNS = {
    "_insert_query_measurement": [
        "dataset",
        "dataset_variant",
        "scale_factor",
        "query_idx",
        "storage",
        "engine",
        "format",
    ],
    "_insert_compression_time": ["dataset", "dataset_variant", "format", "op"],
    "_insert_compression_size": ["dataset", "dataset_variant", "format"],
    "_insert_random_access": ["dataset", "format"],
    "_insert_vector_search": ["dataset", "layout", "flavor", "threshold"],
}


@pytest.mark.parametrize("fn_name,dim_columns", list(_SET_EXCLUDED_DIM_COLUMNS.items()))
def test_on_conflict_set_excludes_dim_columns(fn_name: str, dim_columns: list[str]) -> None:
    # The existing per-table update test pins that value columns ARE in the SET list; this pins
    # the inverse (load-bearing) BAN invariant: dim columns are NEVER in the SET list, so a future
    # edit adding a dim to a DO UPDATE SET (which would overwrite the dim tuple on an upsert) fails
    # loudly. Parse EVERY assigned (left-hand-side) column in each _insert_ function's SQL SET
    # clause -- regardless of the right-hand side -- so a dim assigned from any RHS (e.g.
    # `dataset = 'x'`, not just `dataset = excluded.dataset`) is also caught.
    src = inspect.getsource(getattr(post_ingest, fn_name))
    match = re.search(r"DO UPDATE SET(.*?)RETURNING", src, re.DOTALL)
    assert match, f"{fn_name}: no `DO UPDATE SET ... RETURNING` clause found"
    set_columns = set(re.findall(r"(\w+)\s*=", match.group(1)))
    assert set_columns, f"{fn_name}: parsed an empty SET column list (regex drift?)"
    for dim in dim_columns:
        assert dim not in set_columns, (
            f"{fn_name}: dim column {dim!r} must NOT be in the ON CONFLICT DO UPDATE SET clause "
            f"(it feeds measurement_id); found SET columns: {sorted(set_columns)}"
        )


def test_unknown_field_rejected(schema_conn: psycopg.Connection) -> None:
    commit = _sample_commit()
    qm = next(r for r in _sample_records() if r["kind"] == "query_measurement")
    qm = dict(qm, surprise_field="x")
    with pytest.raises(SystemExit, match="deny_unknown_fields"):
        post_ingest.ingest_postgres(schema_conn, commit, [qm])
    assert _count(schema_conn, "commits") == 0


def test_missing_required_field_rejected(schema_conn: psycopg.Connection) -> None:
    commit = _sample_commit()
    qm = next(r for r in _sample_records() if r["kind"] == "query_measurement")
    qm = {k: v for k, v in qm.items() if k != "engine"}
    with pytest.raises(SystemExit, match="missing required field"):
        post_ingest.ingest_postgres(schema_conn, commit, [qm])


def test_unknown_kind_rejected(schema_conn: psycopg.Connection) -> None:
    commit = _sample_commit()
    with pytest.raises(SystemExit, match="unknown kind"):
        post_ingest.ingest_postgres(schema_conn, commit, [{"kind": "not_a_kind", "commit_sha": COMMIT_SHA}])


@pytest.mark.parametrize("bad_kind", [[], {}, 5])
def test_non_scalar_kind_rejected(schema_conn: psycopg.Connection, bad_kind: object) -> None:
    # A non-string (unhashable) kind must hit the controlled record-indexed
    # unknown-kind error, not a bare TypeError at the `in` membership check.
    commit = _sample_commit()
    with pytest.raises(SystemExit, match="unknown kind"):
        post_ingest.ingest_postgres(schema_conn, commit, [{"kind": bad_kind, "commit_sha": COMMIT_SHA}])
    assert _count(schema_conn, "commits") == 0


def test_validation_errors_carry_the_record_index(schema_conn: psycopg.Connection) -> None:
    # A bad record at index 1 reports "record 1 (...)", matching the v3 server's
    # indexed per-record errors.
    commit = _sample_commit()
    good = next(r for r in _sample_records() if r["kind"] == "compression_time")
    bad = dict(next(r for r in _sample_records() if r["kind"] == "query_measurement"), storage="floppy")
    with pytest.raises(SystemExit, match=r"record 1 \(query_measurement\): storage must be"):
        post_ingest.ingest_postgres(schema_conn, commit, [good, bad])


@pytest.mark.parametrize("bad", [[], "x", 5, None])
def test_non_object_record_rejected(schema_conn: psycopg.Connection, bad: object) -> None:
    # A JSONL line that parses to a non-object (list/str/int/null) must fail with
    # the record-indexed SystemExit convention, not an uncontrolled AttributeError,
    # and must write nothing (commit upsert rolls back).
    commit = _sample_commit()
    with pytest.raises(SystemExit, match="expected a JSON object"):
        post_ingest.ingest_postgres(schema_conn, commit, [bad])
    assert _count(schema_conn, "commits") == 0


def test_wrong_typed_scalar_fails_loud_and_writes_nothing(schema_conn: psycopg.Connection) -> None:
    # A wrong-typed scalar integer field fails loud with a record-indexed error
    # (via _validate_record_values -> _require_int) and rolls back, rather than
    # writing a divergent row.
    commit = _sample_commit()
    qm = next(r for r in _sample_records() if r["kind"] == "query_measurement")
    qm = dict(qm, query_idx="not-an-int")
    with pytest.raises(SystemExit, match="query_idx must be an integer"):
        post_ingest.ingest_postgres(schema_conn, commit, [qm])
    assert _count(schema_conn, "commits") == 0
    assert _count(schema_conn, "query_measurements") == 0


@pytest.mark.parametrize(
    ("kind", "field", "bad"),
    [
        ("query_measurement", "value_ns", 1230.5),
        ("query_measurement", "peak_physical", 10.0),
        ("compression_size", "value_bytes", 1.5),
        ("vector_search_run", "matches", 5.0),
        ("vector_search_run", "iterations", 3.5),
        ("compression_time", "value_ns", True),
    ],
)
def test_float_or_bool_scalar_value_rejected(
    schema_conn: psycopg.Connection, kind: str, field: str, bad: object
) -> None:
    # Integer value columns bind straight to BIGINT/INTEGER; psycopg adapts a
    # Python float to float8 and Postgres assignment-casts (rounds) it -- where
    # v3 serde i64/i32 rejects a float. _require_int must fail loud (record-indexed)
    # and write nothing. (A memory-quartet field is set fully so only the type
    # under test triggers the failure.)
    commit = _sample_commit()
    rec = next(r for r in _sample_records() if r["kind"] == kind)
    if kind == "query_measurement" and field == "peak_physical":
        rec = dict(rec, peak_physical=bad, peak_virtual=20, physical_delta=1, virtual_delta=2)
    else:
        rec = dict(rec, **{field: bad})
    with pytest.raises(SystemExit, match=f"{field} must be an integer"):
        post_ingest.ingest_postgres(schema_conn, commit, [rec])
    assert _count(schema_conn, "commits") == 0
    assert _count(schema_conn, _TABLE_BY_KIND[kind]) == 0


@pytest.mark.parametrize(
    ("kind", "field", "bad"),
    [
        ("compression_time", "format", 123),  # required str
        ("vector_search_run", "layout", 5),  # required str
        ("query_measurement", "engine", ["x"]),  # required str
    ],
)
def test_required_string_field_type_rejected(
    schema_conn: psycopg.Connection, kind: str, field: str, bad: object
) -> None:
    # A non-string required field must fail loud (record-indexed) rather than
    # reach a TEXT bind or crash in the hash encoder (.encode), matching v3 serde.
    commit = _sample_commit()
    rec = dict(next(r for r in _sample_records() if r["kind"] == kind), **{field: bad})
    with pytest.raises(SystemExit, match=f"{field} must be a string"):
        post_ingest.ingest_postgres(schema_conn, commit, [rec])
    assert _count(schema_conn, "commits") == 0


@pytest.mark.parametrize(
    ("kind", "field", "bad"),
    [
        ("compression_time", "env_triple", 123),  # Option<String>
        ("query_measurement", "dataset_variant", 1),  # Option<String>
        ("query_measurement", "scale_factor", 2.0),  # Option<String>
    ],
)
def test_optional_string_field_type_rejected(
    schema_conn: psycopg.Connection, kind: str, field: str, bad: object
) -> None:
    commit = _sample_commit()
    rec = dict(next(r for r in _sample_records() if r["kind"] == kind), **{field: bad})
    with pytest.raises(SystemExit, match=f"{field} must be a string or null"):
        post_ingest.ingest_postgres(schema_conn, commit, [rec])
    assert _count(schema_conn, "commits") == 0


@pytest.mark.parametrize(
    ("kind", "field", "bad", "bits"),
    [
        ("query_measurement", "query_idx", 2**31, 32),  # i32 hash dim
        ("vector_search_run", "iterations", 2**31, 32),  # i32, NOT a hash dim
        ("query_measurement", "value_ns", 2**63, 64),  # i64
        ("compression_size", "value_bytes", -(2**63) - 1, 64),  # i64 low end
    ],
)
def test_int_out_of_range_rejected(schema_conn: psycopg.Connection, kind: str, field: str, bad: int, bits: int) -> None:
    # An out-of-i32/i64 integer must fail loud (record-indexed) rather than later
    # as an uncaught struct.error (i32 hash dims) or a raw Postgres 22003 overflow.
    commit = _sample_commit()
    rec = dict(next(r for r in _sample_records() if r["kind"] == kind), **{field: bad})
    with pytest.raises(SystemExit, match=f"out of int{bits} range"):
        post_ingest.ingest_postgres(schema_conn, commit, [rec])
    assert _count(schema_conn, "commits") == 0
    assert _count(schema_conn, _TABLE_BY_KIND[kind]) == 0


def test_all_runtimes_element_out_of_i64_range_rejected(schema_conn: psycopg.Connection) -> None:
    commit = _sample_commit()
    rec = dict(next(r for r in _sample_records() if r["kind"] == "compression_time"), all_runtimes_ns=[2**63])
    with pytest.raises(SystemExit, match="element out of int64 range"):
        post_ingest.ingest_postgres(schema_conn, commit, [rec])
    assert _count(schema_conn, "commits") == 0


def test_late_validation_failure_rolls_back_earlier_fact_row(schema_conn: psycopg.Connection) -> None:
    # All-or-nothing: a valid record at index 0 writes a fact row inside the
    # transaction, then an invalid record at index 1 fails validation -- the whole
    # transaction (commit + the index-0 fact row) must roll back, not just abort
    # the second record.
    commit = _sample_commit()
    good = next(r for r in _sample_records() if r["kind"] == "compression_time")
    bad = dict(next(r for r in _sample_records() if r["kind"] == "compression_size"), value_bytes="not-an-int")
    with pytest.raises(SystemExit, match=r"record 1 \(compression_size\): value_bytes must be an integer"):
        post_ingest.ingest_postgres(schema_conn, commit, [good, bad])
    assert _count(schema_conn, "commits") == 0
    assert _count(schema_conn, "compression_times") == 0  # index-0 fact row rolled back
    assert _count(schema_conn, "compression_sizes") == 0


def test_storage_must_be_nvme_or_s3(schema_conn: psycopg.Connection) -> None:
    commit = _sample_commit()
    qm = next(r for r in _sample_records() if r["kind"] == "query_measurement")
    qm = dict(qm, storage="floppy")
    with pytest.raises(SystemExit, match="storage must be"):
        post_ingest.ingest_postgres(schema_conn, commit, [qm])


def test_partial_memory_quartet_rejected(schema_conn: psycopg.Connection) -> None:
    commit = _sample_commit()
    qm = next(r for r in _sample_records() if r["kind"] == "query_measurement")
    qm = dict(qm, peak_physical=10)  # only one of the four memory columns set
    with pytest.raises(SystemExit, match="all four or none"):
        post_ingest.ingest_postgres(schema_conn, commit, [qm])


def test_full_memory_quartet_accepted(schema_conn: psycopg.Connection) -> None:
    commit = _sample_commit()
    qm = next(r for r in _sample_records() if r["kind"] == "query_measurement")
    qm = dict(qm, peak_physical=10, peak_virtual=20, physical_delta=1, virtual_delta=2)
    inserted, updated = post_ingest.ingest_postgres(schema_conn, commit, [qm])
    assert (inserted, updated) == (1, 0)
    row = schema_conn.execute(
        "SELECT peak_physical, peak_virtual, physical_delta, virtual_delta FROM query_measurements"
    ).fetchone()
    assert row == (10, 20, 1, 2)


def test_commit_sha_mismatch_rejected(schema_conn: psycopg.Connection) -> None:
    commit = _sample_commit()
    qm = next(r for r in _sample_records() if r["kind"] == "query_measurement")
    qm = dict(qm, commit_sha="f" * 40)  # does not match envelope commit.sha
    with pytest.raises(SystemExit, match="does not match envelope"):
        post_ingest.ingest_postgres(schema_conn, commit, [qm])


# -- non-Docker unit tests --------------------------------------------------


def test_schema_version_matches_rust_source() -> None:
    """`post-ingest.py`'s SCHEMA_VERSION stays in lockstep with `schema.rs`.

    `vortex-bench/src/v3.rs` (the record producer) does not declare its own
    SCHEMA_VERSION constant; `benchmarks-website/server/src/schema.rs` is the
    source of truth, so the lockstep check is against it.
    """
    schema_rs = (REPO_ROOT / "benchmarks-website/server/src/schema.rs").read_text()
    m = re.search(r"pub const SCHEMA_VERSION:\s*i32\s*=\s*(\d+)", schema_rs)
    assert m is not None, "could not find SCHEMA_VERSION in schema.rs"
    assert post_ingest.SCHEMA_VERSION == int(m.group(1))


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("vortex-bench-prod.abc123.us-east-1.rds.amazonaws.com", "us-east-1"),
        ("vortex-bench-proxy.proxy-abc.us-west-2.rds.amazonaws.com", "us-west-2"),
        ("localhost", None),
        ("foo.bar.example.com", None),
    ],
)
def test_region_from_host(host: str, expected: str | None) -> None:
    assert post_ingest._region_from_host(host) == expected


def _fake_conn(ssl_in_use: bool = True):
    """Stand-in psycopg connection for connect_postgres tests.

    connect_postgres reads only `.pgconn.ssl_in_use` (the post-connect TLS check) and calls
    `.close()` (on the non-TLS rejection path) on the object psycopg.connect returns. The shape
    mirrors the REAL psycopg API: `ssl_in_use` is on the low-level `conn.pgconn` (a `pq.PGconn`),
    NOT on the high-level `conn.info` (`ConnectionInfo`); see test_pgconn_ssl_in_use_accessor.
    """
    return types.SimpleNamespace(pgconn=types.SimpleNamespace(ssl_in_use=ssl_in_use), close=lambda: None)


def test_connect_postgres_mints_iam_token_when_passwordless(monkeypatch) -> None:
    captured: dict = {}

    class FakeClient:
        def generate_db_auth_token(self, **kwargs):
            captured["token_args"] = kwargs
            return "iam-token-xyz"

    class FakeSession:
        region_name = None

        def client(self, service, region_name):
            captured["client"] = (service, region_name)
            return FakeClient()

    fake_boto3 = types.ModuleType("boto3")
    fake_boto3.session = types.SimpleNamespace(Session=lambda: FakeSession())
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)

    def fake_connect(**params):
        captured["connect_params"] = params
        return _fake_conn()

    monkeypatch.setattr(psycopg, "connect", fake_connect)

    dsn = "postgresql://bench_ingest@db.abc.us-east-1.rds.amazonaws.com:5432/benchmarks?sslrootcert=/ca.pem"
    result = post_ingest.connect_postgres(dsn, region=None)

    assert result.pgconn.ssl_in_use is True
    assert captured["token_args"] == {
        "DBHostname": "db.abc.us-east-1.rds.amazonaws.com",
        "Port": 5432,
        "DBUsername": "bench_ingest",
        "Region": "us-east-1",
    }
    cp = captured["connect_params"]
    assert cp["password"] == "iam-token-xyz"
    assert cp["sslmode"] == "verify-full"
    # The non-credential libpq params must flow through to psycopg.connect so the
    # verify-full handshake validates against the right host + CA bundle.
    assert cp["host"] == "db.abc.us-east-1.rds.amazonaws.com"
    assert cp["dbname"] == "benchmarks"
    assert cp["user"] == "bench_ingest"
    assert cp["sslrootcert"] == "/ca.pem"


def test_connect_postgres_uses_dsn_password_without_token(monkeypatch) -> None:
    # If boto3 were imported here it would fail, proving the token path is not
    # taken when the DSN already carries a password.
    monkeypatch.setitem(sys.modules, "boto3", None)

    captured: dict = {}

    def fake_connect(**params):
        captured.update(params)
        return _fake_conn()

    monkeypatch.setattr(psycopg, "connect", fake_connect)

    # A password-bearing DSN does not mint an IAM token. The user must be
    # bench_ingest (always enforced); verify-full is still required.
    dsn = "postgresql://bench_ingest:secret@localhost:5432/db?sslmode=verify-full"
    post_ingest.connect_postgres(dsn, region="us-east-1")

    assert captured["password"] == "secret"
    assert captured["sslmode"] == "verify-full"


@pytest.mark.parametrize(
    "dsn",
    [
        # Previously-bypassing shapes: loopback host, hostaddr-only (no host=, so the
        # old DSN-host locality check saw None), and a real RDS host. All must be
        # rejected now that bench_ingest is required unconditionally.
        "postgresql://migrator:secret@localhost:5432/db?sslmode=verify-full",
        "postgresql://migrator:secret@/db?hostaddr=203.0.113.10&sslmode=verify-full",
        "postgresql://postgres:secret@db.abc.us-east-1.rds.amazonaws.com:5432/benchmarks?sslmode=verify-full",
    ],
)
def test_connect_postgres_always_requires_bench_ingest_user(monkeypatch, dsn: str) -> None:
    # The least-privilege check no longer relies on a (bypassable) host heuristic:
    # any non-bench_ingest user is refused regardless of host / hostaddr / auth.
    def boom(**params):
        raise AssertionError("psycopg.connect must not be reached for a non-bench_ingest user")

    monkeypatch.setattr(psycopg, "connect", boom)

    with pytest.raises(SystemExit, match="bench_ingest"):
        post_ingest.connect_postgres(dsn, region="us-east-1")


@pytest.mark.parametrize("weak", ["require", "disable", "prefer"])
def test_connect_postgres_rejects_weak_sslmode(monkeypatch, weak: str) -> None:
    # The sslmode contract is enforced before connect: a DSN that downgrades it
    # must fail loudly, not silently weaken the ingest TLS posture.
    def boom(**params):
        raise AssertionError("psycopg.connect must not be reached for a weak sslmode")

    monkeypatch.setattr(psycopg, "connect", boom)

    dsn = f"postgresql://bench_ingest@db.abc.us-east-1.rds.amazonaws.com:5432/benchmarks?sslmode={weak}"
    with pytest.raises(SystemExit, match="sslmode=verify-full"):
        post_ingest.connect_postgres(dsn, region="us-east-1")


@pytest.mark.parametrize("bad_user", ["migrator", "postgres", "GitHubBenchmarkSchemaRole"])
def test_connect_postgres_iam_path_rejects_non_bench_ingest_user(monkeypatch, bad_user: str) -> None:
    # The IAM path mints a token for the DSN user; enforce least-privilege so a
    # misconfigured DSN cannot ingest as migrator/postgres. boto3=None proves
    # the token path is refused before it is reached.
    monkeypatch.setitem(sys.modules, "boto3", None)

    def boom(**params):
        raise AssertionError("psycopg.connect must not be reached for a non-bench_ingest IAM user")

    monkeypatch.setattr(psycopg, "connect", boom)

    dsn = f"postgresql://{bad_user}@db.abc.us-east-1.rds.amazonaws.com:5432/benchmarks"
    with pytest.raises(SystemExit, match="bench_ingest"):
        post_ingest.connect_postgres(dsn, region="us-east-1")


def test_connect_postgres_forces_public_search_path(monkeypatch) -> None:
    # The writer uses unqualified table names, so connect_postgres pins search_path=public to
    # defend against DSN/PGOPTIONS search_path drift. libpq applies repeated -c settings
    # last-wins, so a hostile DSN search_path must be overridden by ours appended last.
    captured: dict = {}

    def fake_connect(**params):
        captured["connect_params"] = params
        return _fake_conn()

    monkeypatch.setattr(psycopg, "connect", fake_connect)

    dsn = (
        "host=db.example.com port=5432 dbname=benchmarks user=bench_ingest "
        "password=secret sslrootcert=/ca.pem options='-c search_path=evil'"
    )
    post_ingest.connect_postgres(dsn, region=None)
    options = captured["connect_params"]["options"]
    assert options.rstrip().endswith("-c search_path=public"), options


def test_connect_postgres_rejects_non_tls_connection(monkeypatch) -> None:
    # The verify-full contract must hold for the password branch too: a hostless / Unix-socket DSN
    # bypasses sslmode (libpq does not TLS over a local socket), so connect_postgres asserts the
    # RESOLVED connection actually used TLS (conn.pgconn.ssl_in_use) and fails loud + closes the
    # connection otherwise -- the host check alone (IAM branch only) cannot catch this.
    monkeypatch.setitem(sys.modules, "boto3", None)
    closed = {"n": 0}
    fake = types.SimpleNamespace(
        pgconn=types.SimpleNamespace(ssl_in_use=False),
        close=lambda: closed.__setitem__("n", closed["n"] + 1),
    )
    monkeypatch.setattr(psycopg, "connect", lambda **params: fake)
    dsn = "user=bench_ingest password=secret dbname=db sslmode=verify-full host=/var/run/postgresql"
    with pytest.raises(SystemExit, match="not using TLS"):
        post_ingest.connect_postgres(dsn, region=None)
    assert closed["n"] == 1  # the rejected (non-TLS) connection is closed


def test_pgconn_ssl_in_use_accessor() -> None:
    # Pin where ssl_in_use actually lives in psycopg so connect_postgres's post-connect TLS check
    # cannot silently use the wrong accessor again (a prior cycle shipped conn.info.ssl_in_use,
    # which raises AttributeError on every real connect): ssl_in_use is on the low-level pq.PGconn
    # (reached via conn.pgconn), NOT on the high-level ConnectionInfo (conn.info). A psycopg upgrade
    # that moves it trips this loudly, and it guarantees the _fake_conn mock shape matches reality.
    assert hasattr(psycopg.pq.PGconn, "ssl_in_use")
    assert not hasattr(psycopg.ConnectionInfo, "ssl_in_use")


def test_real_connection_exposes_pgconn_ssl_in_use(schema_conn: psycopg.Connection) -> None:
    # Pin the conn.pgconn INTERMEDIATE traversal connect_postgres walks (not just the ssl_in_use
    # leaf): a real psycopg connection must expose conn.pgconn.ssl_in_use, so a future psycopg
    # rename of the .pgconn attribute trips loudly rather than being masked by the SimpleNamespace
    # mocks the connect_postgres unit tests use (the cycle-9/10 regression class).
    assert hasattr(schema_conn, "pgconn")
    assert isinstance(schema_conn.pgconn.ssl_in_use, bool)


def test_main_postgres_composition(monkeypatch, capsys) -> None:
    # Pin the production --postgres CLI wiring (no DB): read_records -> build_commit ->
    # connect_postgres -> ingest_postgres -> compact JSON to stdout -> conn.close() in finally.
    # Explicitly clear both refresh env vars so this test is robust to ambient env:
    # the refresh branch is gated on both being set, and testing its skip is
    # covered by test_post_ingest_revalidate.py.
    monkeypatch.delenv("BENCH_SITE_BASE_URL", raising=False)
    monkeypatch.delenv("BENCH_REVALIDATE_TOKEN", raising=False)
    closed = {"n": 0}
    conn = types.SimpleNamespace(close=lambda: closed.__setitem__("n", closed["n"] + 1))
    monkeypatch.setattr(post_ingest, "read_records", lambda path: [{"kind": "compression_size"}])
    monkeypatch.setattr(post_ingest, "build_commit", lambda *a, **k: _sample_commit())
    monkeypatch.setattr(post_ingest, "connect_postgres", lambda dsn, region: conn)
    monkeypatch.setattr(post_ingest, "ingest_postgres", lambda c, commit, records: (3, 2))
    args = types.SimpleNamespace(
        jsonl_path=Path("x.jsonl"),
        commit_sha=COMMIT_SHA,
        repo_url="https://example.com/repo",
        git_dir=None,
        postgres="dsn",
        region=None,
        timeout=30.0,
    )
    rc = post_ingest._main_postgres(args)
    assert rc == 0
    assert closed["n"] == 1  # conn closed in the finally block
    assert json.loads(capsys.readouterr().out) == {"records": 1, "inserted": 3, "updated": 2}


# -- v3 --server path: argparse contract + dispatch (the refactor's new surface) --

_ARGV_BASE = ["post-ingest.py", "x.jsonl", "--commit-sha", "a" * 40, "--benchmark-id", "b"]


def test_server_and_postgres_are_mutually_exclusive(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", _ARGV_BASE + ["--server", "http://x", "--postgres", "postgresql://h/d"])
    with pytest.raises(SystemExit):
        post_ingest.parse_args()


def test_exactly_one_mode_is_required(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", _ARGV_BASE)  # neither --server nor --postgres
    with pytest.raises(SystemExit):
        post_ingest.parse_args()


def test_main_dispatches_to_server_vs_postgres(monkeypatch) -> None:
    # The post-refactor main() routes by mode; pin that --server -> _main_server
    # and --postgres -> _main_postgres without exercising the real ingest paths.
    calls: list[str] = []
    monkeypatch.setattr(post_ingest, "_main_postgres", lambda args: calls.append("postgres") or 0)
    monkeypatch.setattr(post_ingest, "_main_server", lambda args: calls.append("server") or 0)

    monkeypatch.setattr(sys, "argv", _ARGV_BASE + ["--postgres", "postgresql://bench_ingest@h:5432/d"])
    assert post_ingest.main() == 0
    monkeypatch.setattr(sys, "argv", _ARGV_BASE + ["--server", "http://x"])
    assert post_ingest.main() == 0

    assert calls == ["postgres", "server"]


# Argv base without --benchmark-id, for the v4-optional / v3-required tests.
_ARGV_NO_BENCH_ID = ["post-ingest.py", "x.jsonl", "--commit-sha", "a" * 40]


def test_postgres_mode_does_not_require_benchmark_id(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", _ARGV_NO_BENCH_ID + ["--postgres", "postgresql://bench_ingest@h:5432/d"])
    args = post_ingest.parse_args()  # must not raise -- --benchmark-id is v3-only
    assert args.postgres == "postgresql://bench_ingest@h:5432/d"
    assert args.benchmark_id is None


def test_server_mode_requires_benchmark_id(monkeypatch, capsys) -> None:
    # parse_args accepts the omission (it's globally optional), but _main_server rejects it:
    # --benchmark-id feeds the v3 envelope's run_meta. Set a valid token so the missing-benchmark-id
    # check is the ONLY failing condition: _main_server checks benchmark_id BEFORE the token, so
    # without a token a deleted benchmark-id check would still return 2 via the token path and pass
    # for the wrong reason. Assert stderr names --benchmark-id to pin WHICH check rejected.
    monkeypatch.setenv("INGEST_BEARER_TOKEN", "tok")
    monkeypatch.setattr(sys, "argv", _ARGV_NO_BENCH_ID + ["--server", "http://x"])
    assert post_ingest.main() == 2
    assert "--benchmark-id" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# PR-3.5 cross-check harness: the Python --postgres writer UPDATEs (does not
# duplicate-INSERT) a pre-seeded row and the values round-trip. The local-
# container tests below validate the harness's discrimination; the PROD run
# (against the Rust-seeded RDS) is an operator gate.
# ---------------------------------------------------------------------------


def test_cross_check_clean_when_writer_updates_seeded_row(schema_conn: psycopg.Connection) -> None:
    commit = _sample_commit()
    rec = _sample_records()[0]  # query_measurement
    mid = cross_check_mod.measurement_id_for(post_ingest._measurement_id_module(), rec)
    # Seed the row DIRECTLY (independent of the writer -- a stand-in for a Rust-
    # loaded row), with a deliberately-different value_ns/all_runtimes_ns so a clean
    # cross-check proves the writer's UPDATE actually overwrote the seeded values.
    schema_conn.execute(
        """
        INSERT INTO query_measurements
          (measurement_id, commit_sha, dataset, scale_factor, query_idx, storage,
           engine, format, value_ns, all_runtimes_ns)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::bigint[])
        """,
        (
            mid,
            rec["commit_sha"],
            rec["dataset"],
            rec["scale_factor"],
            rec["query_idx"],
            rec["storage"],
            rec["engine"],
            rec["format"],
            99,
            [99],
        ),
    )

    report = cross_check_mod.cross_check(schema_conn, commit, [rec])

    assert report.is_clean(), str(report)
    assert (report.inserted, report.updated) == (0, 1)
    assert _count(schema_conn, "query_measurements") == 1  # UPDATE, not a duplicate
    # The seeded value_ns (99) was overwritten by the envelope's value.
    stored = schema_conn.execute(
        "SELECT value_ns FROM query_measurements WHERE measurement_id = %s", (mid,)
    ).fetchone()[0]
    assert stored == rec["value_ns"]


def test_cross_check_clean_over_all_kinds(schema_conn: psycopg.Connection) -> None:
    commit = _sample_commit()
    records = _sample_records()  # one record per fact-table kind
    # Seed once (the stand-in for PR-3.4's prod seed), then the harness re-ingests:
    # every record must UPDATE its seeded row, exercising all five measurement_id
    # paths + value comparisons.
    post_ingest.ingest_postgres(schema_conn, commit, records)

    report = cross_check_mod.cross_check(schema_conn, commit, records)

    assert report.is_clean(), str(report)
    assert (report.inserted, report.updated) == (0, len(records))
    for table in _FACT_TABLES:
        assert _count(schema_conn, table) == 1, f"{table} row count drifted"


def test_cross_check_flags_insert_when_row_not_seeded(schema_conn: psycopg.Connection) -> None:
    # No pre-seed: the writer INSERTs (a duplicate, in prod terms) rather than
    # UPDATEs, so the harness must FLAG it -- this is the discrimination that
    # catches a Python-vs-Rust measurement_id divergence against the live seed.
    commit = _sample_commit()
    rec = _sample_records()[0]

    report = cross_check_mod.cross_check(schema_conn, commit, [rec])

    assert not report.is_clean()
    assert report.inserted == 1
    assert any("INSERT" in problem for problem in report.problems), report.problems


def test_cross_check_value_mismatches_discriminate() -> None:
    rec = _sample_records()[0]  # query_measurement: value_ns + all_runtimes_ns
    matching = {"value_ns": rec["value_ns"], "all_runtimes_ns": rec["all_runtimes_ns"]}
    assert cross_check_mod.value_mismatches(matching, rec) == []

    wrong = {"value_ns": rec["value_ns"] + 1, "all_runtimes_ns": rec["all_runtimes_ns"]}
    problems = cross_check_mod.value_mismatches(wrong, rec)
    assert len(problems) == 1 and "value_ns" in problems[0]

    # all_runtimes_ns is compared element-wise and order-sensitively: a reorder mismatches.
    reordered = {"value_ns": rec["value_ns"], "all_runtimes_ns": list(reversed(rec["all_runtimes_ns"]))}
    assert any("all_runtimes_ns" in p for p in cross_check_mod.value_mismatches(reordered, rec))

    # env_triple (omitted by the base sample) is discriminated when a record carries it.
    rec_env = {**rec, "env_triple": "x86_64-linux"}
    db_env = {
        "value_ns": rec["value_ns"],
        "all_runtimes_ns": rec["all_runtimes_ns"],
        "env_triple": "aarch64-darwin",
    }
    assert any("env_triple" in p for p in cross_check_mod.value_mismatches(db_env, rec_env))

    # vector_search side counters (matches/rows_scanned/bytes_scanned/iterations) discriminate.
    vsr = _sample_records()[4]
    counter_cols = ("value_ns", "all_runtimes_ns", "matches", "rows_scanned", "bytes_scanned", "iterations")
    db_vsr = {c: vsr[c] for c in counter_cols}
    db_vsr["matches"] = vsr["matches"] + 1
    assert any("matches" in p for p in cross_check_mod.value_mismatches(db_vsr, vsr))


def test_cross_check_compares_env_and_memory_columns(schema_conn: psycopg.Connection) -> None:
    # The base sample omits env_triple + the memory quartet, so the integration
    # round-trip (writer UPDATE -> harness re-read -> compare) is otherwise never
    # exercised on them. Seed a row whose env/memory/value DIFFER from the envelope,
    # so a clean cross-check proves the writer UPDATEd those columns AND the harness
    # compared them -- a writer SET-list omission of env_triple/memory would fail it.
    commit = _sample_commit()
    rec = {
        **_sample_records()[0],
        "env_triple": "aarch64-darwin",
        "peak_physical": 11,
        "peak_virtual": 22,
        "physical_delta": 33,
        "virtual_delta": 44,
    }
    mid = cross_check_mod.measurement_id_for(post_ingest._measurement_id_module(), rec)
    schema_conn.execute(
        """
        INSERT INTO query_measurements
          (measurement_id, commit_sha, dataset, scale_factor, query_idx, storage, engine,
           format, value_ns, all_runtimes_ns, peak_physical, peak_virtual, physical_delta,
           virtual_delta, env_triple)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::bigint[], %s, %s, %s, %s, %s)
        """,
        (
            mid,
            rec["commit_sha"],
            rec["dataset"],
            rec["scale_factor"],
            rec["query_idx"],
            rec["storage"],
            rec["engine"],
            rec["format"],
            1,
            [1],
            1,
            1,
            1,
            1,
            "x86_64-linux",
        ),
    )

    report = cross_check_mod.cross_check(schema_conn, commit, [rec])

    assert report.is_clean(), str(report)
    assert (report.inserted, report.updated) == (0, 1)
    # The seeded env_triple + a memory column were overwritten by the envelope's values.
    row = schema_conn.execute(
        "SELECT env_triple, peak_physical FROM query_measurements WHERE measurement_id = %s",
        (mid,),
    ).fetchone()
    assert row == ("aarch64-darwin", 11)
