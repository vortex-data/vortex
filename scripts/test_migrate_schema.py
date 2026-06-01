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
from testcontainers.postgres import PostgresContainer

SCRIPT_PATH = Path(__file__).resolve().parent / "migrate-schema.py"

# Subprocess timeout for the runner tests. Generous to tolerate slow CI
# machines but finite so a hung runner or psycopg.connect retry surfaces
# as a clear pytest failure instead of blocking the whole CI job.
_SUBPROCESS_TIMEOUT_S = 60


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


@pytest.fixture(scope="module")
def postgres_dsn() -> Iterator[str]:
    """Spin up a Postgres testcontainer for the module and yield a libpq DSN.

    testcontainers' `get_connection_url` returns a SQLAlchemy-style URL
    (`postgresql+psycopg2://...`); psycopg wants a libpq URI, so we rebuild
    the URI from the container's exposed accessors.

    Skipped (not failed) when Docker isn't available locally — CI runs this
    via the testcontainers-friendly job (Docker socket mounted).
    """
    if not _docker_available():
        pytest.skip("Docker not running; skipping Postgres testcontainer tests")
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
