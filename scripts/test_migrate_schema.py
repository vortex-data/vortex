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
    """The real Phase-1 migrations (`001` + `002` + `003`) apply against vanilla
    Postgres and are recorded in the ledger in order."""
    applied = runner.apply(conn, REPO_MIGRATIONS_DIR)

    assert applied == 3
    with conn.cursor() as cur:
        cur.execute("SELECT filename FROM public._applied_migrations ORDER BY filename")
        assert [row[0] for row in cur.fetchall()] == [
            "001_initial_schema.sql",
            "002_iam_db_user.sql",
            "003_migrator_ledger_grant.sql",
        ]


def test_real_migrations_idempotent(conn: psycopg.Connection) -> None:
    """Re-applying the real migration set is a no-op (ledger-tracked)."""
    first = runner.apply(conn, REPO_MIGRATIONS_DIR)
    second = runner.apply(conn, REPO_MIGRATIONS_DIR)

    assert first == 3
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
        inner = indexdef.split(" USING btree (", 1)[1].rsplit(")", 1)[0]
        cols = [part.strip().split()[0].strip('"') for part in inner.split(",")]
        assert cols == expected_cols, (
            f"{index}: expected columns {expected_cols}, got {cols} "
            f"(indexdef: {indexdef})"
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
    assert "DB_ENDPOINT" in pairs["RDS_BENCH_INSTANCE_ENDPOINT"], (
        f"RDS_BENCH_INSTANCE_ENDPOINT must be set from ${{DB_ENDPOINT}}, got body: {pairs['RDS_BENCH_INSTANCE_ENDPOINT']}"
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
