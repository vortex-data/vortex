#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["psycopg[binary]>=3.2"]
# ///

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

# `psycopg[binary]` appears in two places: the PEP 723 inline metadata block
# above (so `uv run scripts/migrate-schema.py` resolves the dep standalone)
# and the workspace `dev` group in `pyproject.toml` (so pytest + the rest of
# the workspace tooling pick it up under `uv run --all-packages`). Keep both
# in lockstep when bumping the pin.

"""Apply forward-only Postgres migrations from `migrations/*.sql`.

Files are applied in filename order; the runner tracks applied filenames in
the `public._applied_migrations` table on the target database and is idempotent
(re-running with no pending files is a no-op).

Connection is via standard libpq environment variables (`PGHOST`,
`PGDATABASE`, `PGUSER`, `PGPASSWORD`, `PGPORT`, `PGSSLMODE`) or an
explicit DSN passed via `--target=<dsn>`. CI generates IAM-auth tokens
out-of-band via `aws rds generate-db-auth-token` and exports them as
`PGPASSWORD`; the runner stays substrate-agnostic and never touches AWS
APIs itself.

Exit codes
----------
- `apply`  : 0 on success (zero or more migrations applied).
- `status` : 0 when the on-disk migration set matches the ledger;
             1 when there is drift (pending files OR applied-but-deleted
             files). CI uses this for clean-tree gates.
"""

import argparse
import sys
from pathlib import Path

import psycopg

_PARSER_DESCRIPTION = "Apply forward-only Postgres migrations from `migrations/*.sql`."

# Idempotent on re-create; PRIMARY KEY guards against duplicate ledger rows
# when two CI runs race for the same migration file. The migration DDL itself
# may still execute twice on concurrent runs, so each migration must be
# idempotent or serialization must be enforced upstream (e.g. via the
# `concurrency:` group in the CI workflow).
# All ledger references are schema-qualified to `public._applied_migrations`
# so that a non-default `search_path` (likely on the PR-1.3 `migrator` IAM
# role under standard `rds_iam` least-privilege patterns) cannot silently
# split the ledger between `apply` (which writes) and `status` (which reads).
APPLIED_MIGRATIONS_DDL = """
CREATE TABLE IF NOT EXISTS public._applied_migrations (
    filename TEXT PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""


def discover(migrations_dir: Path) -> list[Path]:
    """Return migration files sorted by filename (lexicographic == numeric under
    the `NNN_<desc>.sql` naming convention documented in `migrations/README.md`).

    Only regular files match; subdirectories whose names happen to end in
    `.sql` are ignored so the downstream `path.read_text()` in [`apply`] does
    not crash with [`IsADirectoryError`] partway through a run after earlier
    migrations have already committed under the autocommit-per-transaction
    discipline.

    Case-insensitive on the suffix so a file authored as `001_FOO.SQL` on a
    case-insensitive filesystem (macOS APFS default) is still discovered on
    case-sensitive Linux CI hosts.

    Raises [`FileNotFoundError`] if the directory does not exist; [`main`]
    translates this to a clean SystemExit at the CLI boundary so callers that
    `import` the runner get a normal exception instead of a process abort.
    """
    if not migrations_dir.is_dir():
        raise FileNotFoundError(f"migrations directory not found: {migrations_dir}")
    return sorted(
        p
        for p in migrations_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".sql"
    )


def _applied_set(conn: psycopg.Connection) -> set[str]:
    """Read the `public._applied_migrations` ledger, ensuring the table exists.

    Private because the function relies on `conn.autocommit = True`; exposing
    it as a public API would let importers re-introduce the cycle-1 implicit-
    outer-transaction bug by calling this on a non-autocommit connection.
    [`apply`] enforces the precondition before calling this helper.
    """
    with conn.cursor() as cur:
        cur.execute(APPLIED_MIGRATIONS_DDL)
        cur.execute("SELECT filename FROM public._applied_migrations")
        return {row[0] for row in cur.fetchall()}


def _read_applied_filenames(conn: psycopg.Connection) -> set[str] | None:
    """Pure read of the `public._applied_migrations` ledger; returns `None`
    when the table does not exist yet (a fresh database the runner has never
    touched).

    Used by [`status`] so that running the status command against a read-only
    role does not require CREATE-table privileges.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT to_regclass('public._applied_migrations') IS NOT NULL"
        )
        exists = cur.fetchone()[0]
        if not exists:
            return None
        cur.execute("SELECT filename FROM public._applied_migrations")
        return {row[0] for row in cur.fetchall()}


def apply(conn: psycopg.Connection, migrations_dir: Path) -> int:
    """Apply all pending migrations in order. Returns the count applied."""
    # Force autocommit BEFORE any execute so each `with conn.transaction()`
    # below is a real top-level Postgres transaction, not a SAVEPOINT nested
    # inside an implicit outer transaction. Without this, the
    # `CREATE TABLE IF NOT EXISTS` in `_applied_set` would lazily open an
    # outer transaction that never commits, and a failing migration N would
    # propagate the exception out of `psycopg.connect()`'s context manager,
    # which rolls back the outer transaction and discards every prior
    # successfully-applied migration in the same run.
    conn.autocommit = True
    applied = _applied_set(conn)
    pending = [p for p in discover(migrations_dir) if p.name not in applied]
    applied_count = 0
    for path in pending:
        sql = path.read_text(encoding="utf-8")
        if not sql.strip():
            # An empty migration file is almost always an authoring slip; if
            # it were silently skipped without a ledger row, every subsequent
            # `apply` would rediscover it as pending and `status` would
            # permanently report drift. Treat as a hard error so the operator
            # fixes the file (delete it or add the intended SQL) before
            # re-running.
            raise ValueError(
                f"empty migration file: {path}. Migrations must contain at "
                "least one SQL statement; delete the file or add the intended "
                "DDL."
            )
        # Each migration runs in its OWN top-level transaction (autocommit is
        # True on the connection, so `conn.transaction()` opens a fresh
        # transaction here). On success the transaction commits before the
        # next iteration; on failure it rolls back THIS migration only and
        # re-raises -- earlier migrations are already committed and persist
        # across the failure.
        with conn.transaction():
            with conn.cursor() as cur:
                cur.execute(sql)
                cur.execute(
                    "INSERT INTO public._applied_migrations (filename) VALUES (%s)",
                    (path.name,),
                )
        applied_count += 1
        print(f"applied {path.name}", file=sys.stderr)
    return applied_count


def status(conn: psycopg.Connection, migrations_dir: Path) -> int:
    """Print applied + pending + drifted migrations.

    Returns 0 when the on-disk set matches the ledger; 1 when there is drift
    (pending files OR applied-but-deleted files). Performs no DDL so this can
    run against a read-only role; the `public._applied_migrations` table is treated
    as "empty" when absent.
    """
    applied = _read_applied_filenames(conn) or set()
    on_disk = [p.name for p in discover(migrations_dir)]
    on_disk_set = set(on_disk)
    pending = [name for name in on_disk if name not in applied]
    orphaned = sorted(applied - on_disk_set)

    for name in on_disk:
        mark = "x" if name in applied else " "
        print(f"[{mark}] {name}")
    for name in orphaned:
        print(f"[!] {name} -- applied but missing from disk")

    summary = (
        f"{len(applied)} applied, {len(pending)} pending, "
        f"{len(orphaned)} orphaned"
    )
    print(summary, file=sys.stderr)

    return 0 if not pending and not orphaned else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=_PARSER_DESCRIPTION)
    parser.add_argument("command", choices=["apply", "status"])
    parser.add_argument(
        "--target",
        default="",
        help="psycopg connection string; defaults to libpq env vars (PGHOST etc.)",
    )
    parser.add_argument(
        "--migrations",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "migrations",
        help="path to migrations directory (default: <repo>/migrations)",
    )
    args = parser.parse_args()

    try:
        with psycopg.connect(args.target) as conn:
            if args.command == "apply":
                count = apply(conn, args.migrations)
                print(f"{count} migration(s) applied", file=sys.stderr)
                return 0
            return status(conn, args.migrations)
    except FileNotFoundError as e:
        # Translate the typed exception from `discover` into a clean CLI
        # error; callers that import this module get the original exception
        # via `discover` instead of an opaque SystemExit.
        print(str(e), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
