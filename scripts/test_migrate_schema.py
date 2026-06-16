# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Tests for `scripts/migrate-schema.py`.

The migrate-schema runner is intentionally substrate-agnostic (no AWS / IAM
auth code paths inside the runner itself), so the tests exercise it against
a vanilla Postgres testcontainer with password auth.

The file under test is hyphenated (`migrate-schema.py`), so it cannot be
imported via the normal module path; we load it via `importlib.util` and
exercise the internal `apply`, `status`, `discover`, and `_applied_set`
functions directly. The CLI entry point (`main`) is exercised via a
subprocess invocation in `test_cli_apply_via_subprocess`.
"""

import importlib.util
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import psycopg
import pytest
from psycopg import conninfo
from psycopg import sql as pg_sql
from testcontainers.postgres import PostgresContainer

SCRIPT_PATH = Path(__file__).resolve().parent / "migrate-schema.py"

# Subprocess timeout for the runner tests. Generous to tolerate slow CI
# machines but finite so a hung runner or psycopg.connect retry surfaces
# as a clear pytest failure instead of blocking the whole CI job.
_SUBPROCESS_TIMEOUT_S = 60

# The repository's real migrations directory (PR-1.3's `001_initial_schema.sql`
# and `002_iam_db_user.sql`). The `test_real_migrations_*` tests apply these
# against a vanilla Postgres testcontainer to prove the DDL is valid Postgres
# and that the schema shape matches the authoritative DuckDB DDL in
# `benchmarks-website/server/src/schema.rs`.
REPO_MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"

# Tables `001_initial_schema.sql` must create, in boot/creation order (the
# `commits` dim first, then the five fact families). Mirrors
# `benchmarks-website/server/src/schema.rs` and its `TABLES` ordering.
_EXPECTED_TABLES = [
    "commits",
    "query_measurements",
    "compression_times",
    "compression_sizes",
    "random_access_times",
    "vector_search_runs",
]

# Read-path indexes `001_initial_schema.sql` must create: the `commits`
# timestamp-ordering index plus one chart-filter index per fact table.
_EXPECTED_INDEXES = [
    "idx_commits_timestamp",
    "idx_query_measurements_chart",
    "idx_compression_times_chart",
    "idx_compression_sizes_chart",
    "idx_random_access_times_chart",
    "idx_vector_search_runs_chart",
]

# Ordered indexed-column names per index (DESC/ASC qualifiers and quoting
# stripped). Pins the dim-leading read-path order: the ratified Key decision is
# that these indexes follow the chart-query filter columns, NOT the
# `measurement_id` hash field order. Regression guard so a future edit cannot
# silently reorder or drop an indexed column (index-name-only checks miss that).
# Per-index `(table, [columns in order])`. Pinning the table as well as the
# columns guards against an index created with the right name/columns on the
# WRONG table (which would leave the intended table unindexed yet pass a
# name+columns-only check).
_EXPECTED_INDEX_COLUMNS = {
    "idx_commits_timestamp": ("commits", ["timestamp", "commit_sha"]),
    "idx_query_measurements_chart": (
        "query_measurements",
        ["dataset", "dataset_variant", "scale_factor", "storage", "query_idx"],
    ),
    "idx_compression_times_chart": ("compression_times", ["dataset", "dataset_variant"]),
    "idx_compression_sizes_chart": ("compression_sizes", ["dataset", "dataset_variant"]),
    "idx_random_access_times_chart": ("random_access_times", ["dataset"]),
    "idx_vector_search_runs_chart": ("vector_search_runs", ["dataset", "layout", "threshold"]),
    # Read-path-perf indexes added by migration 006 (PR-5.1.5). The summary index
    # backs the latest-per-series skip scan (fix c; its 007 INCLUDE payload is
    # pinned separately by `_EXPECTED_INDEX_INCLUDES`); the engine/format indexes
    # back collectFilterUniverse's loose-index skip scan (fix d).
    "idx_query_measurements_summary": (
        "query_measurements",
        [
            "dataset",
            "dataset_variant",
            "scale_factor",
            "storage",
            "query_idx",
            "engine",
            "format",
            "commit_timestamp",
        ],
    ),
    "idx_query_measurements_engine": ("query_measurements", ["engine"]),
    "idx_query_measurements_format": ("query_measurements", ["format"]),
    "idx_compression_times_format": ("compression_times", ["format"]),
    "idx_compression_sizes_format": ("compression_sizes", ["format"]),
    "idx_random_access_times_format": ("random_access_times", ["format"]),
}

# Expected INCLUDE (non-key payload) columns per index; indexes absent from this
# map must have none. Migration 007 rebuilt the summary index with INCLUDE
# (value_ns) so the latest-per-series skip scan is an Index Only Scan (the
# value_ns filter and projection read from the index leaf, no heap fetch);
# losing the payload would merge green on a key-columns-only check while
# regressing summaries to the pre-007 heap-fetch plan.
_EXPECTED_INDEX_INCLUDES = {
    "idx_query_measurements_summary": ["value_ns"],
}

# Per-table `(column_name, is_nullable)` in ordinal order, mirroring the DuckDB
# DDL exactly. Behavior-preservation: the plan's `Out of scope` forbids changing
# column order, nullability, or dim-tuple membership, so this is a regression
# pin against accidental shape drift in the Postgres translation.
_EXPECTED_COLUMNS: dict[str, list[tuple[str, str]]] = {
    "commits": [
        ("commit_sha", "NO"),
        ("timestamp", "NO"),
        ("message", "YES"),
        ("author_name", "YES"),
        ("author_email", "YES"),
        ("committer_name", "YES"),
        ("committer_email", "YES"),
        ("tree_sha", "NO"),
        ("url", "NO"),
    ],
    "query_measurements": [
        ("measurement_id", "NO"),
        ("commit_sha", "NO"),
        ("dataset", "NO"),
        ("dataset_variant", "YES"),
        ("scale_factor", "YES"),
        ("query_idx", "NO"),
        ("storage", "NO"),
        ("engine", "NO"),
        ("format", "NO"),
        ("value_ns", "NO"),
        ("all_runtimes_ns", "NO"),
        ("peak_physical", "YES"),
        ("peak_virtual", "YES"),
        ("physical_delta", "YES"),
        ("virtual_delta", "YES"),
        ("env_triple", "YES"),
        # Denormalized commit timestamp, appended by migration 006 (PR-5.1.5 fix
        # c). Nullable: writers populate it + the migration backfills, but the
        # column tolerates a transient NULL before the post-deploy re-backfill.
        ("commit_timestamp", "YES"),
    ],
    "compression_times": [
        ("measurement_id", "NO"),
        ("commit_sha", "NO"),
        ("dataset", "NO"),
        ("dataset_variant", "YES"),
        ("format", "NO"),
        ("op", "NO"),
        ("value_ns", "NO"),
        ("all_runtimes_ns", "NO"),
        ("env_triple", "YES"),
    ],
    "compression_sizes": [
        ("measurement_id", "NO"),
        ("commit_sha", "NO"),
        ("dataset", "NO"),
        ("dataset_variant", "YES"),
        ("format", "NO"),
        ("value_bytes", "NO"),
    ],
    "random_access_times": [
        ("measurement_id", "NO"),
        ("commit_sha", "NO"),
        ("dataset", "NO"),
        ("format", "NO"),
        ("value_ns", "NO"),
        ("all_runtimes_ns", "NO"),
        ("env_triple", "YES"),
    ],
    "vector_search_runs": [
        ("measurement_id", "NO"),
        ("commit_sha", "NO"),
        ("dataset", "NO"),
        ("layout", "NO"),
        ("flavor", "NO"),
        ("threshold", "NO"),
        ("value_ns", "NO"),
        ("all_runtimes_ns", "NO"),
        ("matches", "NO"),
        ("rows_scanned", "NO"),
        ("bytes_scanned", "NO"),
        ("iterations", "NO"),
        ("env_triple", "YES"),
    ],
}


def _load_runner():
    """Import `migrate-schema.py` by file path (the hyphen blocks `import`)."""
    spec = importlib.util.spec_from_file_location("migrate_schema", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = _load_runner()


def _docker_available() -> bool:
    """Cheap probe: try to connect to the Docker daemon via `docker info`."""
    try:
        result = subprocess.run(
            ["docker", "info"], capture_output=True, timeout=5, check=False
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _require_docker_for_testcontainers() -> None:
    # The testcontainer suite MUST run in CI: a missing Docker daemon is a HARD FAILURE there,
    # not a silent skip. A green CI job with skipped testcontainer tests would let a migration-runner
    # regression merge undetected, which is exactly what wiring `scripts/` into CI (PR-2.3) closes.
    # Locally (no CI env) we skip so developers without Docker can still run the pure-unit tests.
    if _docker_available():
        return
    if os.environ.get("CI"):
        pytest.fail("Docker unavailable in CI (`docker info` failed); the testcontainer suite must run, not skip")
    pytest.skip("Docker not running; skipping Postgres testcontainer tests")


@pytest.fixture(scope="module")
def postgres_dsn() -> Iterator[str]:
    """Spin up a Postgres testcontainer for the module and yield a libpq DSN.

    testcontainers' `get_connection_url` returns a SQLAlchemy-style URL
    (`postgresql+psycopg2://...`); psycopg wants a libpq URI, so we rebuild
    the URI from the container's exposed accessors.

    Skipped locally when Docker isn't available; FAILS in CI (where the `CI` env var is set)
    so the testcontainer suite can never silently skip on the CI runner.
    """
    _require_docker_for_testcontainers()
    with PostgresContainer("postgres:16-alpine") as container:
        host = container.get_container_host_ip()
        port = container.get_exposed_port(5432)
        dsn = (
            f"postgresql://{container.username}:{container.password}"
            f"@{host}:{port}/{container.dbname}"
        )
        yield dsn


@pytest.fixture
def migrations_dir(tmp_path: Path) -> Path:
    """A clean, empty migrations directory rooted at a tmp path."""
    d = tmp_path / "migrations"
    d.mkdir()
    return d


@pytest.fixture
def conn(postgres_dsn: str) -> Iterator[psycopg.Connection]:
    """Per-test connection scrubbed of any prior `public._applied_migrations` state.

    `DROP TABLE IF EXISTS` runs as the test's first action so each test sees
    a fresh `public._applied_migrations` ledger even though the testcontainer is
    module-scoped (one container, many tests).
    """
    with psycopg.connect(postgres_dsn) as c:
        with c.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS public._applied_migrations CASCADE")
            # Drop any leftover tables created by prior tests' SQL fixtures.
            cur.execute(
                """
                DO $$
                DECLARE
                    r RECORD;
                BEGIN
                    FOR r IN (
                        SELECT tablename FROM pg_tables WHERE schemaname = 'public'
                    ) LOOP
                        EXECUTE 'DROP TABLE IF EXISTS public.' || quote_ident(r.tablename) || ' CASCADE';
                    END LOOP;
                END$$;
                """
            )
        c.commit()
        yield c


def _write_migration(d: Path, name: str, sql: str) -> Path:
    p = d / name
    p.write_text(sql)
    return p


def test_require_docker_fails_loud_in_ci(monkeypatch) -> None:
    # In CI a missing Docker daemon must FAIL the testcontainer suite, not skip it: a green job with
    # skipped migration-runner tests would let a regression merge undetected (the gap PR-2.3 closes).
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


def test_apply_creates_applied_migrations_table_and_runs_migrations(
    conn: psycopg.Connection, migrations_dir: Path
) -> None:
    _write_migration(
        migrations_dir,
        "001_initial.sql",
        "CREATE TABLE widgets (id BIGINT PRIMARY KEY, name TEXT NOT NULL)",
    )

    count = runner.apply(conn, migrations_dir)

    assert count == 1
    with conn.cursor() as cur:
        cur.execute("SELECT filename FROM public._applied_migrations ORDER BY filename")
        assert [row[0] for row in cur.fetchall()] == ["001_initial.sql"]
        cur.execute("SELECT to_regclass('public.widgets')")
        assert cur.fetchone()[0] == "widgets"


def test_apply_is_idempotent(conn: psycopg.Connection, migrations_dir: Path) -> None:
    _write_migration(
        migrations_dir,
        "001_initial.sql",
        "CREATE TABLE widgets (id BIGINT PRIMARY KEY)",
    )

    first = runner.apply(conn, migrations_dir)
    second = runner.apply(conn, migrations_dir)

    assert first == 1
    assert second == 0  # second run sees the same applied set; 0 pending
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM public._applied_migrations")
        assert cur.fetchone()[0] == 1


def test_apply_applies_new_migration_in_order(
    conn: psycopg.Connection, migrations_dir: Path
) -> None:
    _write_migration(
        migrations_dir,
        "001_initial.sql",
        "CREATE TABLE widgets (id BIGINT PRIMARY KEY)",
    )
    runner.apply(conn, migrations_dir)

    # Add a second migration; only the second one should run on next apply.
    _write_migration(
        migrations_dir,
        "002_add_gizmos.sql",
        "CREATE TABLE gizmos (id BIGINT PRIMARY KEY)",
    )

    count = runner.apply(conn, migrations_dir)

    assert count == 1
    with conn.cursor() as cur:
        cur.execute("SELECT filename FROM public._applied_migrations ORDER BY filename")
        assert [row[0] for row in cur.fetchall()] == [
            "001_initial.sql",
            "002_add_gizmos.sql",
        ]
        cur.execute("SELECT to_regclass('public.gizmos')")
        assert cur.fetchone()[0] == "gizmos"


def test_apply_rolls_back_on_failure_subprocess(
    postgres_dsn: str, conn: psycopg.Connection, migrations_dir: Path
) -> None:
    """Exercises the production close-on-exception path.

    A previous version of this test ran the runner against the same
    `conn` that verified state and wrapped the failing call in
    `pytest.raises`. That passed for the wrong reason: pytest's
    exception-swallowing kept the connection's `with` block exiting
    cleanly (commit), masking the fact that the runner's own
    `with psycopg.connect(...)` in `main()` exits with the exception
    and rolls back the outer transaction. Invoking the CLI via
    subprocess + verifying via a FRESH connection actually exercises
    the production path.
    """
    _write_migration(
        migrations_dir,
        "001_good.sql",
        "CREATE TABLE widgets (id BIGINT PRIMARY KEY)",
    )
    _write_migration(
        migrations_dir,
        "002_bad.sql",
        "CREATE TABLE gizmos (id BIGINT PRIMARY KEY); SELECT 1/0",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "apply",
            f"--target={postgres_dsn}",
            f"--migrations={migrations_dir}",
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        timeout=_SUBPROCESS_TIMEOUT_S,
    )

    assert result.returncode != 0, (
        f"runner should exit non-zero on failed migration; got 0\nstderr: {result.stderr}"
    )
    assert "applied 001_good.sql" in result.stderr

    # Open an independent verification connection. Anything visible here is
    # what would persist across the runner's process boundary in production.
    with psycopg.connect(postgres_dsn) as verify:
        with verify.cursor() as cur:
            cur.execute("SELECT filename FROM public._applied_migrations ORDER BY filename")
            assert [row[0] for row in cur.fetchall()] == ["001_good.sql"], (
                "Migration 1 must persist after migration 2 fails; otherwise the "
                "runner re-applies migration 1 on the next run"
            )
            cur.execute("SELECT to_regclass('public.widgets')")
            assert cur.fetchone()[0] == "widgets", (
                "Migration 1's DDL must persist across migration 2's failure"
            )
            cur.execute("SELECT to_regclass('public.gizmos')")
            assert cur.fetchone()[0] is None, (
                "Migration 2's DDL must roll back on its own failure"
            )


def test_status_reports_applied_and_pending(
    conn: psycopg.Connection, migrations_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_migration(migrations_dir, "001_initial.sql", "SELECT 1")
    _write_migration(migrations_dir, "002_pending.sql", "SELECT 1")

    # Apply only the first migration manually.
    with conn.cursor() as cur:
        cur.execute(runner.APPLIED_MIGRATIONS_DDL)
        cur.execute(
            "INSERT INTO public._applied_migrations (filename) VALUES (%s)",
            ("001_initial.sql",),
        )
    conn.commit()

    rc = runner.status(conn, migrations_dir)

    assert rc == 1  # one pending file -> non-zero drift exit code
    captured = capsys.readouterr()
    assert "[x] 001_initial.sql" in captured.out
    assert "[ ] 002_pending.sql" in captured.out


def test_status_clean_returns_zero(
    conn: psycopg.Connection, migrations_dir: Path
) -> None:
    _write_migration(migrations_dir, "001_initial.sql", "SELECT 1")
    with conn.cursor() as cur:
        cur.execute(runner.APPLIED_MIGRATIONS_DDL)
        cur.execute(
            "INSERT INTO public._applied_migrations (filename) VALUES (%s)",
            ("001_initial.sql",),
        )
    conn.commit()

    assert runner.status(conn, migrations_dir) == 0


def test_status_flags_orphaned_applied_files(
    conn: psycopg.Connection, migrations_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Filename in the ledger but not on disk is drift; status exits non-zero."""
    # No file on disk; ledger has a row for a deleted-from-repo migration.
    with conn.cursor() as cur:
        cur.execute(runner.APPLIED_MIGRATIONS_DDL)
        cur.execute(
            "INSERT INTO public._applied_migrations (filename) VALUES (%s)",
            ("001_deleted.sql",),
        )
    conn.commit()

    rc = runner.status(conn, migrations_dir)

    assert rc == 1
    captured = capsys.readouterr()
    assert "[!] 001_deleted.sql -- applied but missing from disk" in captured.out


def test_status_does_not_create_table(
    conn: psycopg.Connection, migrations_dir: Path
) -> None:
    """status against a database with no ledger table treats it as empty.

    Pure read so the runner can run as a read-only role (e.g., a Vercel
    read-side check for migration drift).
    """
    # The fixture's setup already dropped any prior ledger; do not seed.
    rc = runner.status(conn, migrations_dir)

    assert rc == 0
    # public._applied_migrations must NOT have been created by `status`.
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass('public._applied_migrations')::text")
        assert cur.fetchone()[0] is None


def test_discover_missing_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="migrations directory not found"):
        runner.discover(tmp_path / "does-not-exist")


def test_discover_case_insensitive_suffix(migrations_dir: Path) -> None:
    """Mixed-case .SQL files authored on case-insensitive macOS are still
    discovered on case-sensitive Linux CI hosts."""
    _write_migration(migrations_dir, "001_lower.sql", "SELECT 1")
    _write_migration(migrations_dir, "002_UPPER.SQL", "SELECT 1")
    _write_migration(migrations_dir, "003_Mixed.Sql", "SELECT 1")

    files = runner.discover(migrations_dir)

    assert [p.name for p in files] == [
        "001_lower.sql",
        "002_UPPER.SQL",
        "003_Mixed.Sql",
    ]


def test_discover_sorts_lexicographically(migrations_dir: Path) -> None:
    _write_migration(migrations_dir, "002_b.sql", "SELECT 1")
    _write_migration(migrations_dir, "001_a.sql", "SELECT 1")
    _write_migration(migrations_dir, "010_c.sql", "SELECT 1")
    # Non-sql files are ignored.
    (migrations_dir / "README.md").write_text("not a migration")

    files = runner.discover(migrations_dir)

    assert [p.name for p in files] == ["001_a.sql", "002_b.sql", "010_c.sql"]


def test_cli_apply_via_subprocess(
    postgres_dsn: str, conn: psycopg.Connection, migrations_dir: Path
) -> None:
    """Exercise the CLI entry point end-to-end via subprocess."""
    _write_migration(
        migrations_dir,
        "001_initial.sql",
        "CREATE TABLE widgets (id BIGINT PRIMARY KEY)",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "apply",
            f"--target={postgres_dsn}",
            f"--migrations={migrations_dir}",
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        timeout=_SUBPROCESS_TIMEOUT_S,
    )

    assert result.returncode == 0, result.stderr
    assert "applied 001_initial.sql" in result.stderr
    with conn.cursor() as cur:
        cur.execute("SELECT filename FROM public._applied_migrations")
        assert [row[0] for row in cur.fetchall()] == ["001_initial.sql"]


def test_apply_rejects_empty_sql_file(
    conn: psycopg.Connection, migrations_dir: Path
) -> None:
    """An empty migration file is an authoring slip; the runner must reject
    it loudly rather than silently leaving it in `pending` on every subsequent
    run. Per the cycle-2 must-fix finding."""
    _write_migration(migrations_dir, "001_initial.sql", "CREATE TABLE widgets (id BIGINT PRIMARY KEY)")
    _write_migration(migrations_dir, "002_empty.sql", "")

    with pytest.raises(ValueError, match="empty migration file"):
        runner.apply(conn, migrations_dir)

    # Migration 1 still committed (each migration is its own top-level txn).
    with conn.cursor() as cur:
        cur.execute("SELECT filename FROM public._applied_migrations")
        assert {row[0] for row in cur.fetchall()} == {"001_initial.sql"}


def test_apply_rejects_whitespace_only_sql_file(
    conn: psycopg.Connection, migrations_dir: Path
) -> None:
    """Whitespace-only migration is treated as empty."""
    _write_migration(migrations_dir, "001_ws.sql", "  \n\t  \n")

    with pytest.raises(ValueError, match="empty migration file"):
        runner.apply(conn, migrations_dir)


def test_discover_ignores_subdirectory_with_sql_suffix(
    migrations_dir: Path,
) -> None:
    """A subdirectory whose name happens to end in `.sql` must not be
    enumerated as a migration; otherwise `apply` crashes mid-run with
    IsADirectoryError after earlier migrations have committed."""
    (migrations_dir / "001_real.sql").write_text("SELECT 1")
    (migrations_dir / "002_subdir.sql").mkdir()

    files = runner.discover(migrations_dir)

    assert [p.name for p in files] == ["001_real.sql"]


def test_apply_uses_public_schema_under_custom_search_path(
    postgres_dsn: str, migrations_dir: Path
) -> None:
    """Cycle-2 must-fix: _applied_set / apply write to `public._applied_migrations`
    and _read_applied_filenames reads from `public._applied_migrations`. A
    non-default search_path (likely on the PR-1.3 `migrator` IAM role under
    standard rds_iam least-privilege patterns) must not silently split the
    ledger between writer and reader.

    Connects with `options='-c search_path=foo,public'`, runs apply + status,
    asserts both see the same ledger and that the ledger lives under
    `public._applied_migrations`.
    """
    _write_migration(migrations_dir, "001_initial.sql", "CREATE TABLE widgets (id BIGINT PRIMARY KEY)")

    # Use libpq's `options` parameter to set search_path on connect.
    with psycopg.connect(postgres_dsn, options="-c search_path=foo,public") as c:
        with c.cursor() as cur:
            # Create a separate schema so search_path resolution has somewhere
            # to look BEFORE public.
            cur.execute("CREATE SCHEMA IF NOT EXISTS foo")
        c.commit()
        applied_count = runner.apply(c, migrations_dir)
        assert applied_count == 1

    # Re-open with the same search_path and verify status sees the ledger.
    with psycopg.connect(postgres_dsn, options="-c search_path=foo,public") as c:
        rc = runner.status(c, migrations_dir)
        assert rc == 0, "status under non-default search_path must see the ledger written by apply"

    # Verify the ledger truly lives in public.
    with psycopg.connect(postgres_dsn) as c:
        with c.cursor() as cur:
            cur.execute("SELECT to_regclass('public._applied_migrations')::text")
            assert cur.fetchone()[0] == "_applied_migrations"
            cur.execute("SELECT to_regclass('foo._applied_migrations')::text")
            assert cur.fetchone()[0] is None, (
                "ledger must NOT have been created in `foo` schema under search_path resolution"
            )
            # Phase-end cycle-2 must-fix: the migration DDL itself must also land
            # in `public`, matching the ledger. Without the runner's
            # `SET LOCAL search_path TO public`, the unqualified `CREATE TABLE
            # widgets` would resolve to `foo` (first on the search_path) while the
            # public ledger still records the migration as applied -- a silent
            # table mislocation the reader's `public.<table>` would never find.
            cur.execute("SELECT to_regclass('public.widgets')::text")
            assert cur.fetchone()[0] == "widgets", (
                "migration table must be created in `public` regardless of the caller's search_path"
            )
            cur.execute("SELECT to_regclass('foo.widgets')::text")
            assert cur.fetchone()[0] is None, (
                "migration table must NOT land in `foo` under search_path resolution"
            )


def test_real_migrations_apply_cleanly(conn: psycopg.Connection) -> None:
    """The real migrations (`001` through `007`) apply against vanilla Postgres
    and are recorded in the ledger in order."""
    applied = runner.apply(conn, REPO_MIGRATIONS_DIR)

    assert applied == 7
    with conn.cursor() as cur:
        cur.execute("SELECT filename FROM public._applied_migrations ORDER BY filename")
        assert [row[0] for row in cur.fetchall()] == [
            "001_initial_schema.sql",
            "002_iam_db_user.sql",
            "003_migrator_ledger_grant.sql",
            "004_ingest_role.sql",
            "005_read_role.sql",
            "006_read_path_perf.sql",
            "007_summary_covering_index.sql",
        ]


def test_real_migrations_idempotent(conn: psycopg.Connection) -> None:
    """Re-applying the real migration set is a no-op (ledger-tracked)."""
    first = runner.apply(conn, REPO_MIGRATIONS_DIR)
    second = runner.apply(conn, REPO_MIGRATIONS_DIR)

    assert first == 7
    assert second == 0


def test_real_migrations_create_expected_tables(conn: psycopg.Connection) -> None:
    runner.apply(conn, REPO_MIGRATIONS_DIR)

    with conn.cursor() as cur:
        for table in _EXPECTED_TABLES:
            cur.execute("SELECT to_regclass(%s)::text", (f"public.{table}",))
            assert cur.fetchone()[0] == table, f"table {table} was not created"


def test_real_migrations_create_expected_indexes(conn: psycopg.Connection) -> None:
    runner.apply(conn, REPO_MIGRATIONS_DIR)

    with conn.cursor() as cur:
        cur.execute("SELECT indexname FROM pg_indexes WHERE schemaname = 'public'")
        names = {row[0] for row in cur.fetchall()}

    for index in _EXPECTED_INDEXES:
        assert index in names, f"index {index} missing; have {sorted(names)}"


def test_real_migrations_index_columns(conn: psycopg.Connection) -> None:
    """Pin the indexed columns AND their order, not just the index names.

    The composite indexes are deliberately dim-leading (the read-path chart
    filter columns), NOT the `measurement_id` hash field order -- a ratified
    Key decision. An index-name-only check cannot catch a silent reorder or a
    dropped column, so assert the column list of each index definition."""
    runner.apply(conn, REPO_MIGRATIONS_DIR)

    for index, (expected_table, expected_cols) in _EXPECTED_INDEX_COLUMNS.items():
        with conn.cursor() as cur:
            cur.execute(
                "SELECT tablename, indexdef FROM pg_indexes "
                "WHERE schemaname = 'public' AND indexname = %s",
                (index,),
            )
            row = cur.fetchone()
        assert row is not None, f"index {index} missing"
        tablename, indexdef = row
        assert tablename == expected_table, (
            f"{index}: expected on table {expected_table!r}, got {tablename!r}"
        )
        # Anchor on the btree column list (robust vs a future expression /
        # partial / opclass-with-parens index, where a trailing-paren scan would
        # grab the wrong group). Take the leading token of each comma part
        # (drops DESC/ASC), unquoting reserved-word columns like "timestamp".
        assert " USING btree (" in indexdef, (
            f"{index}: not a plain btree index, parser assumption broken: {indexdef}"
        )
        inner = indexdef.split(" USING btree (", 1)[1]
        # Split any INCLUDE payload off BEFORE parsing the key columns: a bare
        # rsplit(")") on the whole tail silently swallows ") INCLUDE (value_ns"
        # into the last comma part, which both corrupts the key-column parse
        # and leaves the covering payload entirely unchecked.
        if ") INCLUDE (" in inner:
            key_part, include_part = inner.split(") INCLUDE (", 1)
            include_cols = [
                c.strip().strip('"') for c in include_part.rsplit(")", 1)[0].split(",")
            ]
        else:
            key_part = inner.rsplit(")", 1)[0]
            include_cols = []
        cols = [part.strip().split()[0].strip('"') for part in key_part.split(",")]
        assert cols == expected_cols, (
            f"{index}: expected columns {expected_cols}, got {cols} "
            f"(indexdef: {indexdef})"
        )
        assert include_cols == _EXPECTED_INDEX_INCLUDES.get(index, []), (
            f"{index}: expected INCLUDE {_EXPECTED_INDEX_INCLUDES.get(index, [])}, "
            f"got {include_cols} (indexdef: {indexdef})"
        )

    # The column-name check above strips ASC/DESC; separately pin the DESC/DESC
    # ordering of the time-series index (it is what makes recency-window scans
    # over `commits` fast -- a silent drop to ASC would pass the check above).
    with conn.cursor() as cur:
        cur.execute(
            "SELECT indexdef FROM pg_indexes "
            "WHERE schemaname = 'public' AND indexname = 'idx_commits_timestamp'"
        )
        ts_row = cur.fetchone()
    assert ts_row is not None, "index idx_commits_timestamp missing"
    ts_def = ts_row[0]
    ts_inner = ts_def.split(" USING btree (", 1)[1].rsplit(")", 1)[0]
    # Require DESC as a qualifier token somewhere after the column name
    # (`split()[1:]`), not a substring of the column expression, so a future
    # column whose name merely contains "desc" cannot false-pass.
    assert all("DESC" in part.strip().upper().split()[1:] for part in ts_inner.split(",")), (
        f"idx_commits_timestamp must order both columns DESC: {ts_def}"
    )

    # Likewise pin `commit_timestamp DESC` on the summary index: the
    # latest-per-series probe's index-ordered descent (newest first) depends on
    # the trailing key column's direction, and the column-name check strips it.
    with conn.cursor() as cur:
        cur.execute(
            "SELECT indexdef FROM pg_indexes "
            "WHERE schemaname = 'public' AND indexname = 'idx_query_measurements_summary'"
        )
        summary_row = cur.fetchone()
    assert summary_row is not None, "index idx_query_measurements_summary missing"
    summary_def = summary_row[0]
    summary_inner = summary_def.split(" USING btree (", 1)[1].split(") INCLUDE (", 1)[0]
    last_key = summary_inner.split(",")[-1].strip()
    assert last_key.split()[0].strip('"') == "commit_timestamp", (
        f"idx_query_measurements_summary must end its key list with commit_timestamp: {summary_def}"
    )
    assert "DESC" in last_key.upper().split()[1:], (
        f"idx_query_measurements_summary must order commit_timestamp DESC: {summary_def}"
    )


def test_006_backfills_preexisting_query_measurements(
    conn: psycopg.Connection, tmp_path: Path
) -> None:
    """006's one-time backfill UPDATE fills `commit_timestamp` on PRE-EXISTING rows.

    This is the statement that stamped the 4.85M prod rows at apply time, and it is
    invisible to every other test (they apply all migrations before seeding, so the
    backfill always runs against empty tables). Recreate the prod shape: apply only
    001, seed commits + query_measurements rows that predate the column, then apply
    the full migration set and assert every row was stamped from its commit."""
    staged = tmp_path / "pre_006"
    staged.mkdir()
    first = REPO_MIGRATIONS_DIR / "001_initial_schema.sql"
    (staged / first.name).write_text(first.read_text())
    runner.apply(conn, staged)

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO commits (commit_sha, timestamp, tree_sha, url) VALUES
              ('sha-a', TIMESTAMPTZ '2026-01-02 03:04:05.123456+00', 't1', 'https://x/a'),
              ('sha-b', TIMESTAMPTZ '2026-02-03 04:05:06+00',        't2', 'https://x/b')
            """
        )
        cur.execute(
            """
            INSERT INTO query_measurements (
                measurement_id, commit_sha, dataset, dataset_variant, scale_factor,
                query_idx, storage, engine, format, value_ns, all_runtimes_ns
            ) VALUES
              (1, 'sha-a', 'tpch', NULL, '1', 1, 'nvme', 'duckdb', 'parquet', 10, '{10}'),
              (2, 'sha-b', 'tpch', NULL, '1', 2, 'nvme', 'duckdb', 'parquet', 20, '{20}')
            """
        )
    conn.commit()

    runner.apply(conn, REPO_MIGRATIONS_DIR)

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT count(*) FILTER (WHERE q.commit_timestamp IS NULL),
                   count(*) FILTER (WHERE q.commit_timestamp <> c.timestamp)
              FROM query_measurements q
              JOIN commits c USING (commit_sha)
            """
        )
        unstamped, drifted = cur.fetchone()
    assert unstamped == 0, "006 backfill left NULL commit_timestamp on pre-existing rows"
    assert drifted == 0, "006 backfill stamped a timestamp that disagrees with commits"


def test_schema_deploy_targets_instance_endpoint() -> None:
    """Guard the PR-1.6 endpoint repoint (behavioral-drift BAN).

    The schema-deploy workflow MUST connect to the public RDS *instance*
    endpoint (`RDS_BENCH_INSTANCE_ENDPOINT`), not the VPC-internal RDS Proxy
    (`RDS_BENCH_ENDPOINT`, unreachable from off-VPC GitHub runners). A silent
    revert of PGHOST back to the proxy var would otherwise merge green and
    break every schema deploy. No DB needed -- this is a static workflow check."""
    workflow = (
        Path(__file__).resolve().parent.parent
        / ".github"
        / "workflows"
        / "schema-deploy.yml"
    )
    pghost_lines = [
        ln for ln in workflow.read_text().splitlines() if ln.strip().startswith("PGHOST:")
    ]
    assert len(pghost_lines) == 1, f"expected exactly one PGHOST line, got {pghost_lines}"
    line = pghost_lines[0]
    assert "RDS_BENCH_INSTANCE_ENDPOINT" in line, (
        f"schema-deploy PGHOST must be the instance-endpoint var, got: {line}"
    )
    # `RDS_BENCH_ENDPOINT` is NOT a substring of `RDS_BENCH_INSTANCE_ENDPOINT`,
    # so this catches a revert to the proxy var without a false positive.
    assert "RDS_BENCH_ENDPOINT" not in line, (
        f"schema-deploy PGHOST must NOT be the VPC-internal proxy var: {line}"
    )
    # `PGSSLMODE` must be `verify-full`: the workflow connects with an IAM auth
    # token, so the server certificate MUST be authenticated (not merely
    # encrypted) to prevent a MITM from harvesting the token. A silent downgrade
    # to `require`/`prefer` would weaken that and otherwise merge green. (The
    # README master-password bootstrap path is pinned separately by
    # `test_readme_bootstrap_pins_verify_full`.)
    sslmode_lines = [
        ln for ln in workflow.read_text().splitlines() if ln.strip().startswith("PGSSLMODE:")
    ]
    assert len(sslmode_lines) == 1, f"expected exactly one PGSSLMODE line, got {sslmode_lines}"
    assert sslmode_lines[0].split(":", 1)[1].strip() == "verify-full", (
        f"schema-deploy PGSSLMODE must be verify-full, got: {sslmode_lines[0]}"
    )


def test_provision_emits_instance_endpoint_var() -> None:
    """Guard the PR-1.6 provision.sh repo-var output (behavioral-drift BAN).

    `provision.sh`'s summary MUST emit `gh variable set
    RDS_BENCH_INSTANCE_ENDPOINT` (the public instance endpoint CI's schema-deploy
    consumes) and MUST NOT set `RDS_BENCH_ENDPOINT` as a GitHub Actions variable
    (the VPC-internal proxy endpoint is Vercel-only, not a GitHub variable). A
    silent revert of the summary to the proxy var would pass the PGHOST-only
    workflow check above yet leave schema-deploy without the instance endpoint.
    No DB needed -- static script check."""
    provision = (
        Path(__file__).resolve().parent.parent
        / "benchmarks-website"
        / "infra"
        / "provision.sh"
    )
    # Parse `gh variable set <NAME> --body "<BODY>"` into {name: body} so we can
    # check both the variable NAMES and the values they are set FROM.
    pairs = {}
    for ln in provision.read_text().splitlines():
        ln = ln.strip()
        if not ln.startswith("gh variable set "):
            continue
        name = ln.split()[3]
        pairs[name] = ln.split("--body", 1)[1].strip() if "--body" in ln else ""
    assert "RDS_BENCH_INSTANCE_ENDPOINT" in pairs, (
        f"provision.sh must emit `gh variable set RDS_BENCH_INSTANCE_ENDPOINT`, got: {sorted(pairs)}"
    )
    # The instance var must carry the INSTANCE endpoint (`${DB_ENDPOINT}`), not
    # the proxy endpoint -- a name-only check would pass `--body "${PROXY_ENDPOINT}"`.
    instance_body = pairs["RDS_BENCH_INSTANCE_ENDPOINT"]
    assert "DB_ENDPOINT" in instance_body, (
        f"RDS_BENCH_INSTANCE_ENDPOINT must be set from ${{DB_ENDPOINT}}, got body: {instance_body}"
    )
    assert "RDS_BENCH_ENDPOINT" not in pairs, (
        f"provision.sh must NOT set the VPC-internal proxy var as a GitHub variable: {sorted(pairs)}"
    )
    # No GitHub repo-var may be set from the proxy endpoint; it goes to Vercel env.
    assert not any("PROXY_ENDPOINT" in body for body in pairs.values()), (
        f"no GitHub repo-var may be set from ${{PROXY_ENDPOINT}}: {pairs}"
    )


def test_provision_grants_instance_dbuser() -> None:
    """Guard the IAM half of the PR-1.6 endpoint repoint (behavioral-drift BAN).

    The repoint couples two changes: `schema-deploy` connects to the instance
    endpoint (PGHOST) AND the `rds-db:connect` policy must grant the instance
    dbuser resource (`dbuser:${DB_RESOURCE_ID}/${PG_MIGRATOR_ROLE}`). The
    PGHOST/repo-var guards do not pin the IAM side; a regression that dropped the
    instance grant (leaving the policy proxy-only) would pass them yet deny every
    schema deploy. No DB needed -- static script check."""
    provision = (
        Path(__file__).resolve().parent.parent
        / "benchmarks-website"
        / "infra"
        / "provision.sh"
    ).read_text()
    assert "dbuser:${DB_RESOURCE_ID}/${PG_MIGRATOR_ROLE}" in provision, (
        "provision.sh rds-db:connect policy must grant the instance dbuser resource "
        "`dbuser:${DB_RESOURCE_ID}/${PG_MIGRATOR_ROLE}` (CI authenticates against the instance)"
    )


def test_provision_creates_ingest_role_with_instance_dbuser() -> None:
    """PR-2.1: `provision.sh` provisions the dedicated `GitHubBenchmarkIngestRole`
    and scopes its `rds-db:connect` to the `bench_ingest` dbuser on the INSTANCE
    resource (the dual-write CI path connects to the public instance endpoint,
    never the VPC-internal proxy), and emits the `GH_BENCH_INGEST_ROLE_ARN` repo
    var. Static script check -- no AWS needed."""
    provision = (
        Path(__file__).resolve().parent.parent
        / "benchmarks-website"
        / "infra"
        / "provision.sh"
    ).read_text()
    assert "GitHubBenchmarkIngestRole" in provision, (
        "provision.sh must provision the GitHubBenchmarkIngestRole OIDC role"
    )
    assert 'PG_INGEST_ROLE="bench_ingest"' in provision, (
        "PG_INGEST_ROLE must be hardcoded to `bench_ingest` (matches migrations/004)"
    )
    assert "rds-db-connect-ingest" in provision, (
        "provision.sh must attach an `rds-db-connect-ingest` inline policy to the ingest role"
    )
    assert "dbuser:${DB_RESOURCE_ID}/${PG_INGEST_ROLE}" in provision, (
        "ingest role rds-db:connect must grant the instance dbuser resource "
        "`dbuser:${DB_RESOURCE_ID}/${PG_INGEST_ROLE}`"
    )
    assert "GH_BENCH_INGEST_ROLE_ARN" in provision, (
        "provision.sh summary must emit the `GH_BENCH_INGEST_ROLE_ARN` repo var"
    )


def test_provision_schema_role_drops_dead_proxy_grant() -> None:
    """PR-2.1 least-privilege cleanup: the schema role's `rds-db:connect` policy
    must no longer grant the VPC-internal proxy resource (dead surface through
    PR-1.6). The `proxy_resource_id` lookup existed ONLY to build that dead grant,
    so its complete removal is the precise, false-positive-free signal that the
    grant is gone. Static script check."""
    provision = (
        Path(__file__).resolve().parent.parent
        / "benchmarks-website"
        / "infra"
        / "provision.sh"
    ).read_text()
    assert "proxy_resource_id" not in provision, (
        "the dead proxy `rds-db:connect` grant (and its `proxy_resource_id` lookup) "
        "must be removed from the schema role in PR-2.1's least-privilege cleanup"
    )
    # Neither OIDC role may scope rds-db:connect to a proxy dbuser. The only
    # dbuser ARN resources in the file must be the two INSTANCE grants. Match the
    # ARN substring `:dbuser:` so prose mentions of "dbuser" don't false-trip.
    dbuser_lines = [ln for ln in provision.splitlines() if ":dbuser:" in ln]
    assert dbuser_lines, "expected at least one rds-db:connect dbuser ARN resource"
    assert all("${DB_RESOURCE_ID}" in ln for ln in dbuser_lines), (
        f"every rds-db:connect dbuser grant must target the instance resource, got: {dbuser_lines}"
    )


def test_readme_bootstrap_pins_verify_full() -> None:
    """Guard the README master-password bootstrap TLS setting (security-critical).

    The one-time master bootstrap documented in the README is the path that
    transmits the RDS master password, so `PGSSLMODE=verify-full` (authenticate
    the server certificate, not merely encrypt) is mandatory there. The workflow
    PGSSLMODE guard (`test_schema_deploy_targets_instance_endpoint`) does NOT
    cover this README command; a silent downgrade to require/prefer/allow in the
    runbook would weaken MITM protection on the master password and otherwise go
    unnoticed. No DB needed -- static doc check."""
    readme = (
        Path(__file__).resolve().parent.parent
        / "benchmarks-website"
        / "infra"
        / "README.md"
    ).read_text()
    assert "export PGSSLMODE=verify-full" in readme, (
        "README bootstrap must `export PGSSLMODE=verify-full` (master-password path)"
    )
    assert "export PGSSLROOTCERT=" in readme, (
        "README bootstrap must `export PGSSLROOTCERT=` (CA bundle for verify-full)"
    )
    for weak in (
        "export PGSSLMODE=require",
        "export PGSSLMODE=prefer",
        "export PGSSLMODE=allow",
        "export PGSSLMODE=disable",
        # `verify-ca` validates the CA chain but skips hostname verification --
        # a real (if weaker) downgrade from verify-full on the master-password path.
        "export PGSSLMODE=verify-ca",
    ):
        assert weak not in readme, (
            f"README bootstrap must not downgrade TLS below verify-full: found `{weak}`"
        )


def test_readme_bootstrap_password_fetch_is_safe() -> None:
    """Guard the README master-password bootstrap fetch (fail-fast + non-interactive).

    The bootstrap fetches the RDS master password from Secrets Manager. Two
    failure modes the runbook must not regress into:
    - `export PGPASSWORD=$(...)` masks the command substitution's exit code
      (export's own status wins), so a failed fetch silently continues with an
      empty password. The runbook must assign first, then export.
    - an interactive `stty -echo; read` prompt consumes following pasted lines
      and can strand the terminal with echo off.
    Pin the non-interactive Secrets-Manager fetch so neither regression merges
    green. No DB needed -- static doc check."""
    readme = (
        Path(__file__).resolve().parent.parent
        / "benchmarks-website"
        / "infra"
        / "README.md"
    ).read_text()
    assert "aws secretsmanager get-secret-value" in readme, (
        "README bootstrap must fetch the master password via "
        "`aws secretsmanager get-secret-value` (non-interactive)"
    )
    assert "jq -er" in readme, (
        "README bootstrap must parse the secret with `jq -er` (fatal on a missing key)"
    )
    # The negative checks scan only CODE lines: full-line `#` shell-comments
    # legitimately MENTION `stty`/`export PGPASSWORD=$(` to explain why they are
    # avoided, so a whole-text substring check would false-positive on the
    # explanatory comment. Lines like `export PGHOST=... # note` keep their
    # leading code token and are correctly retained.
    code = "\n".join(
        s for ln in readme.splitlines()
        if (s := ln.strip()) and not s.startswith("#")
    )
    assert "export PGPASSWORD=$(" not in code, (
        "README bootstrap must NOT use `export PGPASSWORD=$(...)` -- it masks the "
        "substitution's exit code; assign first, then `export PGPASSWORD`"
    )
    for interactive in ("stty -echo", "stty echo", "read -rsp", "IFS= read -r PGPASSWORD"):
        assert interactive not in code, (
            f"README bootstrap must stay non-interactive: found `{interactive}` in a command line"
        )


@pytest.mark.parametrize("table", list(_EXPECTED_COLUMNS))
def test_real_migrations_preserve_column_shape(
    conn: psycopg.Connection, table: str
) -> None:
    """The Postgres translation must preserve the DuckDB column order and
    nullability exactly (behavior-preservation; see plan `Out of scope`)."""
    runner.apply(conn, REPO_MIGRATIONS_DIR)

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name, is_nullable
              FROM information_schema.columns
             WHERE table_schema = 'public' AND table_name = %s
             ORDER BY ordinal_position
            """,
            (table,),
        )
        actual = [(row[0], row[1]) for row in cur.fetchall()]

    assert actual == _EXPECTED_COLUMNS[table]


def test_real_migrations_key_column_types(conn: psycopg.Connection) -> None:
    """Spot-check the type translations that differ or matter for round-trip:
    DuckDB `DOUBLE` -> Postgres `double precision`, `BIGINT[]` stays an array,
    and `measurement_id` is a `bigint` primary key."""
    runner.apply(conn, REPO_MIGRATIONS_DIR)

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT data_type FROM information_schema.columns
             WHERE table_schema = 'public'
               AND table_name = 'vector_search_runs'
               AND column_name = 'threshold'
            """
        )
        assert cur.fetchone()[0] == "double precision"

        cur.execute(
            """
            SELECT data_type FROM information_schema.columns
             WHERE table_schema = 'public'
               AND table_name = 'query_measurements'
               AND column_name = 'all_runtimes_ns'
            """
        )
        assert cur.fetchone()[0] == "ARRAY"

        cur.execute(
            """
            SELECT data_type FROM information_schema.columns
             WHERE table_schema = 'public'
               AND table_name = 'query_measurements'
               AND column_name = 'measurement_id'
            """
        )
        assert cur.fetchone()[0] == "bigint"

        cur.execute(
            """
            SELECT a.attname
              FROM pg_index i
              JOIN pg_attribute a
                ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
             WHERE i.indrelid = 'public.query_measurements'::regclass
               AND i.indisprimary
            """
        )
        assert [row[0] for row in cur.fetchall()] == ["measurement_id"]


def test_real_migrations_create_migrator_role(conn: psycopg.Connection) -> None:
    """`002_iam_db_user.sql` creates a login-capable `migrator` role. The
    `rds_iam` grant is skipped on vanilla Postgres (the role does not exist
    there); on real RDS it binds `migrator` to IAM-token auth."""
    runner.apply(conn, REPO_MIGRATIONS_DIR)

    with conn.cursor() as cur:
        cur.execute("SELECT rolcanlogin FROM pg_roles WHERE rolname = 'migrator'")
        row = cur.fetchone()
        assert row is not None, "migrator role was not created"
        assert row[0] is True, "migrator role must be able to log in"


def test_real_migrations_grant_migrator_ledger_access(conn: psycopg.Connection) -> None:
    """`003_migrator_ledger_grant.sql` gives `migrator` exactly SELECT + INSERT on
    the append-only `public._applied_migrations` ledger. CI runs `apply` AS
    `migrator`, so without these grants the apply path cannot record applied
    migrations against a master-owned ledger. DELETE/UPDATE are intentionally NOT
    granted (the ledger is append-only; least-privilege)."""
    runner.apply(conn, REPO_MIGRATIONS_DIR)

    with conn.cursor() as cur:
        cur.execute(
            "SELECT has_table_privilege('migrator', 'public._applied_migrations', 'SELECT')"
        )
        assert cur.fetchone()[0] is True, "migrator needs SELECT for `status`"
        cur.execute(
            "SELECT has_table_privilege('migrator', 'public._applied_migrations', 'INSERT')"
        )
        assert cur.fetchone()[0] is True, "migrator needs INSERT to record applied migrations"
        cur.execute(
            "SELECT has_table_privilege('migrator', 'public._applied_migrations', 'DELETE')"
        )
        assert cur.fetchone()[0] is False, (
            "migrator must NOT have DELETE on the append-only ledger (least-privilege)"
        )
        cur.execute(
            "SELECT has_table_privilege('migrator', 'public._applied_migrations', 'UPDATE')"
        )
        assert cur.fetchone()[0] is False, (
            "migrator must NOT have UPDATE on the append-only ledger (least-privilege)"
        )


# Minimal valid `INSERT ... ON CONFLICT DO UPDATE` per data table (only NOT NULL
# columns populated), used to prove the `bench_ingest` role can perform the ingest
# write path's upsert on every table. The conflict target is `measurement_id` for
# the fact tables and `commit_sha` for the `commits` dim.
_INGEST_UPSERTS = {
    "commits": (
        "INSERT INTO commits (commit_sha, timestamp, tree_sha, url) "
        "VALUES ('sha-bench-ingest', now(), 'tree-x', 'https://example/x') "
        "ON CONFLICT (commit_sha) DO UPDATE SET url = EXCLUDED.url"
    ),
    "query_measurements": (
        "INSERT INTO query_measurements (measurement_id, commit_sha, dataset, "
        "query_idx, storage, engine, format, value_ns, all_runtimes_ns) "
        "VALUES (1, 'sha-bench-ingest', 'ds', 0, 'st', 'en', 'fmt', 1, ARRAY[1]::bigint[]) "
        "ON CONFLICT (measurement_id) DO UPDATE SET value_ns = EXCLUDED.value_ns"
    ),
    "compression_times": (
        "INSERT INTO compression_times (measurement_id, commit_sha, dataset, "
        "format, op, value_ns, all_runtimes_ns) "
        "VALUES (1, 'sha-bench-ingest', 'ds', 'fmt', 'encode', 1, ARRAY[1]::bigint[]) "
        "ON CONFLICT (measurement_id) DO UPDATE SET value_ns = EXCLUDED.value_ns"
    ),
    "compression_sizes": (
        "INSERT INTO compression_sizes (measurement_id, commit_sha, dataset, "
        "format, value_bytes) "
        "VALUES (1, 'sha-bench-ingest', 'ds', 'fmt', 1) "
        "ON CONFLICT (measurement_id) DO UPDATE SET value_bytes = EXCLUDED.value_bytes"
    ),
    "random_access_times": (
        "INSERT INTO random_access_times (measurement_id, commit_sha, dataset, "
        "format, value_ns, all_runtimes_ns) "
        "VALUES (1, 'sha-bench-ingest', 'ds', 'fmt', 1, ARRAY[1]::bigint[]) "
        "ON CONFLICT (measurement_id) DO UPDATE SET value_ns = EXCLUDED.value_ns"
    ),
    "vector_search_runs": (
        "INSERT INTO vector_search_runs (measurement_id, commit_sha, dataset, "
        "layout, flavor, threshold, value_ns, all_runtimes_ns, matches, "
        "rows_scanned, bytes_scanned, iterations) "
        "VALUES (1, 'sha-bench-ingest', 'ds', 'lay', 'fla', 0.5, 1, ARRAY[1]::bigint[], "
        "1, 1, 1, 1) "
        "ON CONFLICT (measurement_id) DO UPDATE SET value_ns = EXCLUDED.value_ns"
    ),
}


def test_real_migrations_create_bench_ingest_role(conn: psycopg.Connection) -> None:
    """`004_ingest_role.sql` creates a login-capable `bench_ingest` role. The
    `rds_iam` grant is skipped on vanilla Postgres (the role does not exist there);
    on real RDS it binds `bench_ingest` to IAM-token auth, mirroring `migrator`."""
    runner.apply(conn, REPO_MIGRATIONS_DIR)

    with conn.cursor() as cur:
        cur.execute("SELECT rolcanlogin FROM pg_roles WHERE rolname = 'bench_ingest'")
        row = cur.fetchone()
        assert row is not None, "bench_ingest role was not created"
        assert row[0] is True, "bench_ingest role must be able to log in"


def test_bench_ingest_has_dml_only_on_data_tables(conn: psycopg.Connection) -> None:
    """`004_ingest_role.sql` grants `bench_ingest` exactly SELECT/INSERT/UPDATE on
    all six data tables (the upsert write path needs INSERT + UPDATE for
    `ON CONFLICT DO UPDATE`, and SELECT for read-back/reconciliation), plus USAGE
    (not CREATE) on `public`. DELETE/TRUNCATE and schema CREATE are withheld: the
    ingest path is data-DML-only, never DDL (least-privilege separation from the
    schema-deploy `migrator` identity)."""
    runner.apply(conn, REPO_MIGRATIONS_DIR)

    with conn.cursor() as cur:
        for table in _EXPECTED_TABLES:
            qualified = f"public.{table}"
            for priv in ("SELECT", "INSERT", "UPDATE"):
                cur.execute(
                    "SELECT has_table_privilege('bench_ingest', %s, %s)", (qualified, priv)
                )
                assert cur.fetchone()[0] is True, f"bench_ingest needs {priv} on {table}"
            for priv in ("DELETE", "TRUNCATE"):
                cur.execute(
                    "SELECT has_table_privilege('bench_ingest', %s, %s)", (qualified, priv)
                )
                assert cur.fetchone()[0] is False, (
                    f"bench_ingest must NOT have {priv} on {table} (data-DML-only)"
                )
        cur.execute("SELECT has_schema_privilege('bench_ingest', 'public', 'USAGE')")
        assert cur.fetchone()[0] is True, "bench_ingest needs USAGE on public to reach the tables"


def test_real_migrations_create_bench_read_role(conn: psycopg.Connection) -> None:
    """`005_read_role.sql` creates a login-capable `bench_read` role with NO
    `rds_iam` grant: on RDS that membership makes IAM auth mandatory (password
    auth fails), and the read service authenticates with a static password."""
    runner.apply(conn, REPO_MIGRATIONS_DIR)

    with conn.cursor() as cur:
        cur.execute("SELECT rolcanlogin FROM pg_roles WHERE rolname = 'bench_read'")
        row = cur.fetchone()
        assert row is not None, "bench_read role was not created"
        assert row[0] is True, "bench_read role must be able to log in"


def test_bench_read_select_only_on_data_tables(conn: psycopg.Connection) -> None:
    """`005_read_role.sql` grants `bench_read` exactly SELECT on the six data
    tables: every write privilege is withheld (the read service never writes),
    and USAGE on `public` is present so the tables are reachable."""
    runner.apply(conn, REPO_MIGRATIONS_DIR)

    with conn.cursor() as cur:
        for table in _EXPECTED_TABLES:
            qualified = f"public.{table}"
            cur.execute("SELECT has_table_privilege('bench_read', %s, 'SELECT')", (qualified,))
            assert cur.fetchone()[0] is True, f"bench_read needs SELECT on {table}"
            for priv in ("INSERT", "UPDATE", "DELETE", "TRUNCATE"):
                cur.execute(
                    "SELECT has_table_privilege('bench_read', %s, %s)", (qualified, priv)
                )
                assert cur.fetchone()[0] is False, (
                    f"bench_read must NOT have {priv} on {table} (read-only)"
                )
        cur.execute("SELECT has_schema_privilege('bench_read', 'public', 'USAGE')")
        assert cur.fetchone()[0] is True, "bench_read needs USAGE on public to reach the tables"
        cur.execute("SELECT has_schema_privilege('bench_ingest', 'public', 'CREATE')")
        assert cur.fetchone()[0] is False, (
            "bench_ingest must NOT have CREATE on public (no DDL; least-privilege)"
        )


def test_005_revokes_rds_iam_from_preexisting_bench_read(conn: psycopg.Connection) -> None:
    """`005_read_role.sql` enforces its no-`rds_iam` (password-auth) invariant
    idempotently. The `CREATE ROLE` guard only covers a FRESH role; a `bench_read`
    that pre-exists as a member of `rds_iam` (an earlier apply that granted it, or
    manual setup) must end up WITHOUT `rds_iam` after a (re-)apply -- on RDS a
    lingering `rds_iam` membership forces IAM-only auth and silently breaks the read
    service's password auth. Models the live failure class that produced 8b85a3d3f."""
    runner.apply(conn, REPO_MIGRATIONS_DIR)
    conn.autocommit = True

    sql_005 = next(
        p.read_text(encoding="utf-8")
        for p in runner.discover(REPO_MIGRATIONS_DIR)
        if p.name == "005_read_role.sql"
    )

    with conn.cursor() as cur:
        cur.execute("SET search_path TO public")
        try:
            # Model the pre-existing bad state: a cluster-global `rds_iam` role (as
            # on RDS) with `bench_read` already a member.
            cur.execute("CREATE ROLE rds_iam")
            cur.execute("GRANT rds_iam TO bench_read")
            cur.execute("SELECT pg_has_role('bench_read', 'rds_iam', 'MEMBER')")
            assert cur.fetchone()[0] is True, (
                "precondition: bench_read should start as an rds_iam member"
            )

            # Re-applying 005 must REVOKE the membership (idempotent invariant
            # enforcement), restoring the password-auth contract.
            cur.execute(sql_005)
            cur.execute("SELECT pg_has_role('bench_read', 'rds_iam', 'MEMBER')")
            assert cur.fetchone()[0] is False, (
                "005 must REVOKE rds_iam from a pre-existing bench_read so password auth keeps working"
            )
        finally:
            # `rds_iam` is cluster-global; drop it so it does not leak into other
            # tests (whose 002/004/005 applies would otherwise grant it).
            cur.execute("REVOKE rds_iam FROM bench_read")
            cur.execute("DROP ROLE IF EXISTS rds_iam")


def test_bench_ingest_can_upsert_and_is_denied_ddl_delete(
    conn: psycopg.Connection,
) -> None:
    """Round-trip under the real ownership split: the data tables are owned by the
    bootstrapping superuser (modeling the RDS master) and `bench_ingest` is a
    non-owner whose only privileges come from `004`'s grants. AS `bench_ingest`, an
    `INSERT ... ON CONFLICT DO UPDATE` (run twice to exercise both the INSERT and
    the DO UPDATE branch, i.e. both privileges) succeeds on all six tables, while a
    DELETE and a DDL attempt are denied.

    `SET ROLE` drops session privileges to `bench_ingest` -- the portable way to
    test the role without a password, since `bench_ingest` is IAM-auth-only (no
    password) on real RDS."""
    runner.apply(conn, REPO_MIGRATIONS_DIR)
    # `apply` leaves the connection in autocommit; make that explicit so each
    # statement is its own transaction and an expected failure does not poison the
    # session. `SET ROLE` (session-level) persists across autocommit statements.
    conn.autocommit = True

    with conn.cursor() as cur:
        cur.execute("SET ROLE bench_ingest")
        try:
            for stmt in _INGEST_UPSERTS.values():
                cur.execute(stmt)  # first run inserts
                cur.execute(stmt)  # second run hits ON CONFLICT -> DO UPDATE
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                cur.execute("DELETE FROM query_measurements")
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                cur.execute("CREATE TABLE bench_ingest_should_not_exist (x integer)")
        finally:
            cur.execute("RESET ROLE")


def test_bench_ingest_default_privileges_cover_future_migrator_tables(
    conn: psycopg.Connection,
) -> None:
    """`004` sets `ALTER DEFAULT PRIVILEGES FOR ROLE migrator ... GRANT
    SELECT,INSERT,UPDATE ON TABLES TO bench_ingest`, so a future data table created
    by a `migrator`-run migration auto-grants the ingest role its DML without a
    follow-up explicit grant. Pins that non-obvious clause."""
    runner.apply(conn, REPO_MIGRATIONS_DIR)
    conn.autocommit = True

    with conn.cursor() as cur:
        cur.execute("SET ROLE migrator")
        try:
            cur.execute(
                "CREATE TABLE public.future_fact (measurement_id bigint primary key)"
            )
        finally:
            cur.execute("RESET ROLE")
        try:
            for priv in ("SELECT", "INSERT", "UPDATE"):
                cur.execute(
                    "SELECT has_table_privilege('bench_ingest', 'public.future_fact', %s)",
                    (priv,),
                )
                assert cur.fetchone()[0] is True, (
                    f"bench_ingest should auto-receive {priv} on a future migrator-created table"
                )
        finally:
            cur.execute("DROP TABLE IF EXISTS public.future_fact")


def test_bench_read_default_privileges_cover_future_migrator_tables(
    conn: psycopg.Connection,
) -> None:
    """`005` sets `ALTER DEFAULT PRIVILEGES FOR ROLE migrator ... GRANT SELECT ON
    TABLES TO bench_read`, so a future data table created by a `migrator`-run
    migration auto-grants the read role SELECT -- and nothing else -- without a
    follow-up explicit grant. Pins that non-obvious clause: deleting 005's ADP
    block would otherwise leave the existing 005 tests green (the read-role
    counterpart of `test_bench_ingest_default_privileges_cover_future_migrator_tables`)."""
    runner.apply(conn, REPO_MIGRATIONS_DIR)
    conn.autocommit = True

    with conn.cursor() as cur:
        cur.execute("SET ROLE migrator")
        try:
            cur.execute(
                "CREATE TABLE public.future_read_fact (measurement_id bigint primary key)"
            )
        finally:
            cur.execute("RESET ROLE")
        try:
            cur.execute(
                "SELECT has_table_privilege('bench_read', 'public.future_read_fact', 'SELECT')"
            )
            assert cur.fetchone()[0] is True, (
                "bench_read should auto-receive SELECT on a future migrator-created table"
            )
            for priv in ("INSERT", "UPDATE", "DELETE", "TRUNCATE"):
                cur.execute(
                    "SELECT has_table_privilege('bench_read', 'public.future_read_fact', %s)",
                    (priv,),
                )
                assert cur.fetchone()[0] is False, (
                    f"bench_read must NOT auto-receive {priv} on a future migrator-created table (read-only)"
                )
        finally:
            cur.execute("DROP TABLE IF EXISTS public.future_read_fact")


# Password for the simulated RDS master login. The testcontainer authenticates
# host connections with a password, so the modeled master needs one to connect as
# a REAL login (not `SET ROLE`) -- see the test docstring for why that fidelity
# matters.
_RDS_MASTER_SIM_PASSWORD = "rds_master_sim_pw"  # noqa: S105 -- test-only literal


def _rds_master_sim_dsn(postgres_dsn: str) -> str:
    """Rewrite the superuser DSN to log in AS `rds_master_sim` with its password."""
    info = conninfo.conninfo_to_dict(postgres_dsn)
    info["user"] = "rds_master_sim"
    info["password"] = _RDS_MASTER_SIM_PASSWORD
    return conninfo.make_conninfo(**info)


def _scrub_bootstrap_roles(cur: psycopg.Cursor) -> None:
    """Drop the bootstrap roles (and anything they own) so a non-superuser master
    can re-create `migrator`/`bench_ingest` from scratch and thereby hold the ADMIN
    membership the `004` default-privilege self-grant relies on. Idempotent.

    The explicit `REVOKE ... ON SCHEMA public` is load-bearing: `DROP OWNED BY`
    does not reliably clear a `public` privilege one bootstrap role granted to
    another (e.g. `migrator`'s USAGE/CREATE granted by the master), which would
    otherwise block the `DROP ROLE`."""
    cur.execute(
        """
        DO $$
        DECLARE r text;
        BEGIN
            FOR r IN SELECT unnest(ARRAY['migrator', 'bench_ingest', 'bench_read', 'rds_master_sim']) LOOP
                IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = r) THEN
                    EXECUTE format('REVOKE ALL PRIVILEGES ON SCHEMA public FROM %I CASCADE', r);
                END IF;
            END LOOP;
            FOR r IN SELECT unnest(ARRAY['migrator', 'bench_ingest', 'bench_read', 'rds_master_sim']) LOOP
                IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = r) THEN
                    EXECUTE format('DROP OWNED BY %I CASCADE', r);
                    EXECUTE format('DROP ROLE %I', r);
                END IF;
            END LOOP;
        END$$;
        """
    )


def test_real_migrations_apply_as_non_superuser_createrole_master(
    conn: psycopg.Connection, postgres_dsn: str
) -> None:
    """Apply `001`..`005` AS a real non-superuser CREATEROLE login (the RDS master).

    The module's `conn` fixture connects as the testcontainer's built-in user, a
    TRUE superuser (`rolsuper = true`) that bypasses every privilege check. Real RDS
    runs the bootstrap as the master, `rds_superuser` -- a `rolsuper = false` role
    with CREATEROLE. That gap matters for `004`'s `ALTER DEFAULT PRIVILEGES FOR ROLE
    migrator`: PostgreSQL 16 grants a CREATEROLE creator its new role WITH INHERIT
    FALSE, SET FALSE (the `createrole_self_grant` default), so the master neither
    inherits `migrator` nor can `SET ROLE migrator`; the bare ADP rolls back the
    whole `004` transaction under such a master.

    The reproduction MUST be a real login, not a `SET ROLE` from the superuser
    `conn`: `SET ROLE` is checked against the *session* user, so a superuser session
    that `SET ROLE`s to the master keeps superuser bypass and masks the failure
    exactly as the `conn` fixture does. This test therefore creates a NOSUPERUSER
    CREATEROLE master (with `createrole_self_grant` pinned to the default empty
    value), logs in AS it, applies the real migration set, and asserts both a clean
    apply and that the default-privilege rule took effect. It fails
    (`InsufficientPrivilege`) against the pre-fix `004` and passes once `004`
    self-grants the membership the ADP needs via the ADMIN option the master holds
    on `migrator`.
    """
    conn.autocommit = True
    # Build the modeled master, scrubbing any roles a sibling test left behind so
    # THIS master creates `migrator` itself (and thus holds ADMIN on it).
    with conn.cursor() as cur:
        _scrub_bootstrap_roles(cur)
        cur.execute(
            pg_sql.SQL(
                "CREATE ROLE rds_master_sim WITH LOGIN NOSUPERUSER "
                "CREATEROLE PASSWORD {}"
            ).format(pg_sql.Literal(_RDS_MASTER_SIM_PASSWORD))
        )
        # Pin the self-grant policy to the PG/RDS default so creating `migrator`
        # yields INHERIT FALSE / SET FALSE membership deterministically.
        cur.execute("ALTER ROLE rds_master_sim SET createrole_self_grant = ''")
        # The master owns its bootstrap objects: it needs CREATE on `public` to
        # create the six tables, and the GRANT OPTION to re-grant schema access to
        # `migrator` in `002`.
        cur.execute(
            "GRANT CREATE, USAGE ON SCHEMA public TO rds_master_sim WITH GRANT OPTION"
        )

    try:
        # Apply the real migrations AS the non-superuser master (a real login).
        with psycopg.connect(_rds_master_sim_dsn(postgres_dsn)) as master:
            master.autocommit = True
            current_role, is_super = master.execute(
                "SELECT current_user, "
                "(SELECT rolsuper FROM pg_roles WHERE rolname = current_user)"
            ).fetchone()
            assert current_role == "rds_master_sim"
            assert is_super is False, "fidelity requires a non-superuser master login"

            applied = runner.apply(master, REPO_MIGRATIONS_DIR)
            assert applied == 7, (
                "all seven real migrations must apply under the non-superuser master"
            )

        # Verify, on the superuser connection, that the bootstrap produced a usable
        # default-privilege rule for future migrator-created tables.
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT a.privilege_type
                FROM pg_default_acl da
                JOIN pg_roles dr ON dr.oid = da.defaclrole
                JOIN pg_namespace n ON n.oid = da.defaclnamespace
                CROSS JOIN LATERAL aclexplode(da.defaclacl) a
                JOIN pg_roles g ON g.oid = a.grantee
                WHERE dr.rolname = 'migrator' AND n.nspname = 'public'
                  AND da.defaclobjtype = 'r' AND g.rolname = 'bench_ingest'
                """
            )
            granted = {row[0] for row in cur.fetchall()}
            assert granted == {"SELECT", "INSERT", "UPDATE"}, (
                "004 must default-privilege bench_ingest SELECT/INSERT/UPDATE on "
                f"future migrator-created tables; got {sorted(granted)}"
            )
            # The temporary INHERIT self-grant `004` uses must be revoked: the
            # master must not be left inheriting `migrator`.
            cur.execute("SELECT pg_has_role('rds_master_sim', 'migrator', 'USAGE')")
            assert cur.fetchone()[0] is False, (
                "004's temporary INHERIT self-grant must be revoked; the master must "
                "not be left inheriting migrator"
            )
            # And the REVOKE must remove ONLY the self-grant, not the creator's ADMIN
            # auto-grant: the self-grant (grantor = master) and the CREATE ROLE
            # auto-grant (grantor = bootstrap superuser) are SEPARATE pg_auth_members
            # rows, so exactly one membership row must survive -- the auto-grant, with
            # ADMIN TRUE / INHERIT FALSE intact -- proving the master can still
            # administer `migrator` after the bootstrap. A regression that revoked the
            # wrong row (e.g. the auto-grant) would not be caught by the USAGE check
            # above, which passes either way.
            cur.execute(
                """
                SELECT am.admin_option, am.inherit_option
                FROM pg_auth_members am
                JOIN pg_roles r ON r.oid = am.roleid
                JOIN pg_roles m ON m.oid = am.member
                WHERE r.rolname = 'migrator' AND m.rolname = 'rds_master_sim'
                """
            )
            rows = cur.fetchall()
            assert rows == [(True, False)], (
                "the master must retain exactly its CREATE-ROLE ADMIN auto-grant on "
                f"migrator (admin=True, inherit=False) after 004; got {rows}"
            )
    finally:
        # Leave the module-scoped container clean for sibling tests.
        with conn.cursor() as cur:
            _scrub_bootstrap_roles(cur)


# ---------------------------------------------------------------------------
# `requires-superuser` bootstrap-ordering guard (gap #4 / PR-3.1 re-plan).
#
# The marker-detection tests are pure (no Docker); the rejection test is
# testcontainer-gated like the rest of the suite.
# ---------------------------------------------------------------------------


def test_migration_requires_superuser_detects_marker() -> None:
    """The marker is recognised anywhere in the header comment block."""
    assert runner._migration_requires_superuser(
        "-- SPDX-License-Identifier: Apache-2.0\n"
        "-- migrate-schema: requires-superuser\n"
        "CREATE ROLE foo WITH LOGIN;\n"
    )
    # Extra dashes + surrounding whitespace still match the exact directive.
    assert runner._migration_requires_superuser(
        "   ---  migrate-schema: requires-superuser  \nSELECT 1;\n"
    )


def test_migration_requires_superuser_false_without_exact_marker() -> None:
    """Additive migrations, near-misses, and in-string text do NOT match."""
    assert not runner._migration_requires_superuser(
        "-- SPDX-License-Identifier: Apache-2.0\nCREATE TABLE t (x int);\n"
    )
    # Missing colon -> not the exact directive.
    assert not runner._migration_requires_superuser(
        "-- migrate-schema: requires superuser\nSELECT 1;\n"
    )
    # The directive text inside a non-comment SQL line must not trigger it.
    assert not runner._migration_requires_superuser(
        "INSERT INTO t VALUES ('migrate-schema: requires-superuser');\n"
    )


def test_real_bootstrap_migrations_carry_superuser_marker() -> None:
    """002/004/005 (role/grant bootstrap) + 006/007 (master-owned-table DDL) are marked;
    001 + 003 (additive / owned-grant) are not."""
    by_name = {
        p.name: p.read_text(encoding="utf-8") for p in runner.discover(REPO_MIGRATIONS_DIR)
    }
    for marked in (
        "002_iam_db_user.sql",
        "004_ingest_role.sql",
        "005_read_role.sql",
        "006_read_path_perf.sql",
        "007_summary_covering_index.sql",
    ):
        assert marked in by_name, f"expected {marked} in migrations/"
        assert runner._migration_requires_superuser(by_name[marked]), (
            f"{marked} creates roles / runs ALTER DEFAULT PRIVILEGES / ALTERs a master-owned "
            "table and must carry the requires-superuser marker"
        )
    for unmarked in ("001_initial_schema.sql", "003_migrator_ledger_grant.sql"):
        assert unmarked in by_name, f"expected {unmarked} in migrations/"
        assert not runner._migration_requires_superuser(by_name[unmarked]), (
            f"{unmarked} needs neither superuser nor CREATEROLE and must NOT carry the marker"
        )


_LEAST_PRIV_SIM_PASSWORD = "least_priv_sim_pw"  # noqa: S105 -- test-only literal


def _least_priv_sim_dsn(postgres_dsn: str) -> str:
    """DSN that logs in AS the modeled least-privilege role `least_priv_sim`."""
    info = conninfo.conninfo_to_dict(postgres_dsn)
    info["user"] = "least_priv_sim"
    info["password"] = _LEAST_PRIV_SIM_PASSWORD
    return conninfo.make_conninfo(**info)


def test_master_capable_true_for_superuser_conn(conn: psycopg.Connection) -> None:
    """Positive control: the testcontainer's superuser login is master-capable."""
    _require_docker_for_testcontainers()
    assert runner._role_is_master_capable(conn) is True


def test_requires_superuser_migration_rejected_for_non_master_role(
    conn: psycopg.Connection, postgres_dsn: str, migrations_dir: Path
) -> None:
    """A marked migration is rejected BEFORE any DDL when applied by a non-master role.

    Models the misconfiguration the gap-#4 guard exists for: a least-privilege
    `migrator`-class login (NOSUPERUSER NOCREATEROLE) reaching a `requires-superuser`
    bootstrap migration. The guard must fail loud + early -- the marked migration's
    DDL must NOT have run -- so the operator fixes the bootstrap ordering instead of
    debugging a mid-`DO`-block InsufficientPrivilege rollback. The probe body
    (`CREATE TABLE`) is one the least-priv role COULD run given CREATE on `public`, so
    the ONLY thing that can block it is the preflight, not a DDL permission error.
    """
    _require_docker_for_testcontainers()
    conn.autocommit = True

    def _drop_least_priv() -> None:
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS probe_marked")
            # DROP OWNED first: the role may own the ledger it created in `apply`, and
            # it holds schema privileges -- both block a bare DROP ROLE.
            cur.execute(
                "DO $$ BEGIN "
                "  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'least_priv_sim') THEN "
                "    EXECUTE 'DROP OWNED BY least_priv_sim CASCADE'; "
                "    EXECUTE 'DROP ROLE least_priv_sim'; "
                "  END IF; "
                "END $$;"
            )

    _drop_least_priv()
    with conn.cursor() as cur:
        cur.execute(
            pg_sql.SQL(
                "CREATE ROLE least_priv_sim WITH LOGIN NOSUPERUSER NOCREATEROLE PASSWORD {}"
            ).format(pg_sql.Literal(_LEAST_PRIV_SIM_PASSWORD))
        )
        # CREATE on public so it COULD create the ledger + the probe table; this proves
        # the rejection is the preflight, not a DDL permission error on the body.
        cur.execute("GRANT CREATE, USAGE ON SCHEMA public TO least_priv_sim")

    marked = migrations_dir / "001_probe_requires_superuser.sql"
    marked.write_text(
        "-- migrate-schema: requires-superuser\nCREATE TABLE probe_marked (x int);\n",
        encoding="utf-8",
    )

    try:
        with psycopg.connect(_least_priv_sim_dsn(postgres_dsn)) as least_priv:
            least_priv.autocommit = True
            is_super, can_createrole = least_priv.execute(
                "SELECT rolsuper, rolcreaterole FROM pg_roles WHERE rolname = current_user"
            ).fetchone()
            assert (is_super, can_createrole) == (False, False), (
                "fidelity requires a NOSUPERUSER NOCREATEROLE login"
            )
            assert runner._role_is_master_capable(least_priv) is False

            with pytest.raises(PermissionError, match="requires-superuser"):
                runner.apply(least_priv, migrations_dir)

        with conn.cursor() as cur:
            # The marked migration's DDL must NOT have run: the preflight fired first.
            cur.execute("SELECT to_regclass('public.probe_marked') IS NULL")
            assert cur.fetchone()[0] is True, (
                "preflight must reject before the marked migration's CREATE TABLE runs"
            )
            # And it must NOT be recorded as applied.
            cur.execute("SELECT to_regclass('public._applied_migrations') IS NOT NULL")
            if cur.fetchone()[0]:
                cur.execute(
                    "SELECT count(*) FROM public._applied_migrations "
                    "WHERE filename = '001_probe_requires_superuser.sql'"
                )
                assert cur.fetchone()[0] == 0
    finally:
        _drop_least_priv()
