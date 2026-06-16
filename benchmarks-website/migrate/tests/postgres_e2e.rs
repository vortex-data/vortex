// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! End-to-end rehearsal of the DuckDB -> Postgres migration against a real
//! `postgres:16` container.
//!
//! This is the first RUNTIME validation of the loader's Postgres-execution path
//! (PR-3.1: connect, per-table `COPY`, one-transaction atomicity) and the value
//! verifier's Postgres-read path (PR-3.2: the epoch-microsecond timestamp SQL and
//! the `BIGINT[]` reads) -- both COMPILE in their own crates but were never run
//! against a live Postgres until here. It proves the load + verify CODE works
//! before the PR-3.4 real-snapshot LOCAL rehearsal and the PR-5.0 one-shot prod load.
//!
//! Docker gating mirrors the Python testcontainer suite (`scripts/`): the test
//! SKIPS locally when Docker is unavailable, but FAILS LOUD in CI (the `CI` env
//! var is set) so the suite can never silently skip on the runner. The schema is
//! created from the authoritative Postgres DDL (`migrations/001`) via the
//! container's init-SQL entrypoint.

use std::path::Path;
use std::process::Command;

use anyhow::Result;
use tempfile::TempDir;
use testcontainers::Container;
use testcontainers::ImageExt;
use testcontainers::runners::SyncRunner;
use testcontainers_modules::postgres::Postgres;
use vortex_bench_migrate::postgres::load;
use vortex_bench_migrate::verify::run_postgres_value_verify;
use vortex_bench_server::family;
use vortex_bench_server::schema::COMMITS_DDL;

/// The Postgres schema applied to the container at init: the schema-shape
/// migrations the loader touches -- the 001 base DDL, the 006 read-path
/// migration (whose denormalized `query_measurements.commit_timestamp` column
/// the loader's post-COPY denormalization UPDATE requires), and the 007
/// covering-index swap (index-only, but kept so the container's index set
/// matches prod's). The 002-005 role/grant migrations are deliberately
/// omitted: they configure RDS auth, which a throwaway container neither has
/// nor needs.
const SCHEMA_SQL: &str = concat!(
    include_str!("../../../migrations/001_initial_schema.sql"),
    include_str!("../../../migrations/006_read_path_perf.sql"),
    include_str!("../../../migrations/007_summary_covering_index.sql"),
);

/// Per-table row counts the fixture loads. Drives the count assertions.
const FIXTURE_COUNTS: &[(&str, u64)] = &[
    ("commits", 3),
    ("query_measurements", 2),
    ("compression_times", 2),
    ("compression_sizes", 1),
    ("random_access_times", 1),
    ("vector_search_runs", 1),
];

/// Representative rows across all six tables, exercising the value-fidelity edge
/// cases end to end: a sub-second timestamp (`c2`) and a pre-1970 timestamp (`c3`)
/// for the epoch path; a literal `\N` message (`c3`) for COPY-NULL disambiguation;
/// multibyte text in a VALUE-compared column (`c1`'s `commits.message`, so the
/// verify actually discriminates a multibyte-COPY corruption); empty +
/// multi-element + negative `BIGINT[]`; NULL memory columns; and `measurement_id`
/// 7777 in `vector_search_runs`, which the rollback test pre-seeds to force a
/// mid-load conflict.
const FIXTURE_INSERTS: &str = r#"
INSERT INTO commits VALUES
  ('c1', TIMESTAMPTZ '2024-01-15 12:34:56+00', 'café ☕ commit', 'Ann', 'ann@x', 'Cory', 'cory@x', 'tree1', 'https://x/c1'),
  ('c2', TIMESTAMPTZ '2024-06-15 08:30:45.123456+00', NULL, NULL, NULL, NULL, NULL, 'tree2', 'https://x/c2'),
  ('c3', TIMESTAMPTZ '1969-07-20 20:17:00+00', chr(92) || 'N', 'Buzz', 'buzz@moon', 'Neil', 'neil@moon', 'tree3', 'https://x/c3');
INSERT INTO query_measurements VALUES
  (101, 'c1', 'tpch', NULL, '1', 3, 'nvme', 'vortex', 'vortex', 1000, [1000,1100,1050], 5, 6, 7, 8, 'x86_64-linux'),
  (102, 'c2', 'clickbench', 'café', NULL, -1, 's3', 'duckdb', 'parquet', 2000, [2000], NULL, NULL, NULL, NULL, NULL);
INSERT INTO compression_times VALUES
  (201, 'c1', 'taxi', NULL, 'vortex-file-compressed', 'encode', 100, []::BIGINT[], 'x86_64-linux'),
  (202, 'c1', 'taxi', NULL, 'parquet', 'decode', 200, [-5, 7, 7], NULL);
INSERT INTO compression_sizes VALUES
  (301, 'c1', 'taxi', NULL, 'vortex-file-compressed', 4096);
INSERT INTO random_access_times VALUES
  (401, 'c2', 'taxi', 'parquet', 777, [700,777,800], 'aarch64-darwin');
INSERT INTO vector_search_runs VALUES
  (7777, 'c1', 'sift', 'flat', 'f32', 0.95, 500, [500,510], 42, 1000, 64000, 3, 'x86_64-linux');
"#;

/// True when the testcontainer tests should run. Skips locally when Docker is
/// absent; FAILS LOUD in CI so the suite can never silently skip on the runner.
fn require_docker() -> bool {
    let available = Command::new("docker")
        .arg("info")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    if available {
        return true;
    }
    if std::env::var_os("CI").is_some() {
        panic!(
            "Docker unavailable in CI (`docker info` failed); the Postgres \
             testcontainer suite must run, not skip"
        );
    }
    eprintln!("Docker not running; skipping Postgres testcontainer rehearsal");
    false
}

/// Build a file-based v3 DuckDB fixture at `path`, then drop the connection so the
/// loader can open it read-only.
fn build_fixture_duckdb(path: &Path) -> Result<()> {
    let conn = duckdb::Connection::open(path)?;
    conn.execute_batch("SET TimeZone='UTC';")?;
    conn.execute_batch(COMMITS_DDL)?;
    for fam in family::FAMILIES {
        conn.execute_batch(fam.schema_ddl)?;
    }
    conn.execute_batch(FIXTURE_INSERTS)?;
    Ok(())
}

/// Start a `postgres:16-alpine` container with the schema applied via init SQL.
fn start_postgres() -> Result<Container<Postgres>> {
    let container = Postgres::default()
        .with_init_sql(SCHEMA_SQL.as_bytes().to_vec())
        .with_tag("16-alpine")
        .start()?;
    Ok(container)
}

fn dsn_for(container: &Container<Postgres>) -> Result<String> {
    let host = container.get_host()?;
    let port = container.get_host_port_ipv4(5432)?;
    Ok(format!(
        "postgresql://postgres:postgres@{host}:{port}/postgres"
    ))
}

fn table_count(client: &mut postgres::Client, table: &str) -> Result<i64> {
    let row = client.query_one(&format!("SELECT count(*) FROM {table}"), &[])?;
    Ok(row.get(0))
}

#[test]
fn rehearsal_load_then_verify_is_clean() -> Result<()> {
    if !require_docker() {
        return Ok(());
    }
    let dir = TempDir::new()?;
    let duckdb_path = dir.path().join("fixture.duckdb");
    build_fixture_duckdb(&duckdb_path)?;

    let container = start_postgres()?;
    let dsn = dsn_for(&container)?;

    // The loader reports exactly the fixture's per-table row counts.
    let summary = load(&duckdb_path, &dsn, None, false)?;
    for &(table, expected) in FIXTURE_COUNTS {
        let got = summary.per_table.iter().find(|e| e.0 == table).map(|e| e.1);
        assert_eq!(got, Some(expected), "loader count for {table}");
    }

    // The value verify is clean: every non-hashed column matches per id, including
    // the sub-second + pre-1970 epoch timestamps, the literal `\N` message, the
    // multibyte text, and the empty / negative arrays.
    let report = run_postgres_value_verify(&duckdb_path, &dsn, None)?;
    assert!(report.is_clean(), "value verify not clean:\n{report}");

    // The target's per-table counts equal the source's.
    let mut client = postgres::Client::connect(&dsn, postgres::NoTls)?;
    for &(table, expected) in FIXTURE_COUNTS {
        assert_eq!(
            table_count(&mut client, table)? as u64,
            expected,
            "target count for {table}"
        );
    }

    // The loader denormalized `commit_timestamp` onto every `query_measurements`
    // row (migration 006, the read path's latest-per-series sort key): no NULLs
    // remain and each value equals the joined `commits.timestamp`.
    let unstamped: i64 = client
        .query_one(
            "SELECT count(*) FROM query_measurements WHERE commit_timestamp IS NULL",
            &[],
        )?
        .get(0);
    assert_eq!(unstamped, 0, "rows missing denormalized commit_timestamp");
    let mismatched: i64 = client
        .query_one(
            "SELECT count(*) FROM query_measurements q JOIN commits c USING (commit_sha)
              WHERE q.commit_timestamp <> c.timestamp",
            &[],
        )?
        .get(0);
    assert_eq!(mismatched, 0, "denormalized commit_timestamp drifted");
    Ok(())
}

#[test]
fn rehearsal_mid_load_failure_rolls_back_to_empty() -> Result<()> {
    if !require_docker() {
        return Ok(());
    }
    let dir = TempDir::new()?;
    let duckdb_path = dir.path().join("fixture.duckdb");
    build_fixture_duckdb(&duckdb_path)?;

    let container = start_postgres()?;
    let dsn = dsn_for(&container)?;

    // Pre-seed `vector_search_runs` (loaded LAST) with `measurement_id` 7777, which
    // the fixture also loads. The single-transaction load COPYs the five earlier
    // tables, then the `vector_search_runs` COPY hits the duplicate primary key and
    // the whole transaction aborts.
    let mut client = postgres::Client::connect(&dsn, postgres::NoTls)?;
    client.execute(
        "INSERT INTO vector_search_runs \
         (measurement_id, commit_sha, dataset, layout, flavor, threshold, value_ns, \
          all_runtimes_ns, matches, rows_scanned, bytes_scanned, iterations) \
         VALUES (7777, 'preexist', 'd', 'l', 'f', 0.5, 1, '{1}', 1, 1, 1, 1)",
        &[],
    )?;

    let result = load(&duckdb_path, &dsn, None, false);
    assert!(
        result.is_err(),
        "the conflicting load should fail, got {result:?}"
    );

    // The five earlier tables' COPYs rolled back, so they are empty;
    // `vector_search_runs` retains ONLY the pre-seeded row (the fixture's row
    // rolled back with the rest).
    for table in [
        "commits",
        "query_measurements",
        "compression_times",
        "compression_sizes",
        "random_access_times",
    ] {
        assert_eq!(
            table_count(&mut client, table)?,
            0,
            "{table} should be empty after the atomic rollback"
        );
    }
    assert_eq!(
        table_count(&mut client, "vector_search_runs")?,
        1,
        "vector_search_runs keeps only the pre-seeded row"
    );
    Ok(())
}

#[test]
fn rehearsal_replace_load_reseeds_a_populated_target() -> Result<()> {
    if !require_docker() {
        return Ok(());
    }
    let dir = TempDir::new()?;
    let duckdb_path = dir.path().join("fixture.duckdb");
    build_fixture_duckdb(&duckdb_path)?;

    let container = start_postgres()?;
    let dsn = dsn_for(&container)?;
    let mut client = postgres::Client::connect(&dsn, postgres::NoTls)?;

    // A first plain load populates the target with the fixture.
    load(&duckdb_path, &dsn, None, false)?;
    // A second plain load MUST fail: every primary key is already present, so the
    // first COPY aborts on the duplicate. This is the data-refresh footgun the
    // `replace` flag exists to remove -- a re-load over a populated target.
    assert!(
        load(&duckdb_path, &dsn, None, false).is_err(),
        "a plain re-load over a populated target must abort on duplicate keys"
    );

    // The replace load empties every table inside the transaction, then reloads, so
    // it succeeds and the per-table counts equal the fixture's exactly -- no
    // duplicated rows and no rows left over from the first load.
    let summary = load(&duckdb_path, &dsn, None, true)?;
    for &(table, expected) in FIXTURE_COUNTS {
        let got = summary.per_table.iter().find(|e| e.0 == table).map(|e| e.1);
        assert_eq!(got, Some(expected), "replace-load count for {table}");
        assert_eq!(
            table_count(&mut client, table)? as u64,
            expected,
            "target count for {table} after replace"
        );
    }

    // The post-COPY denormalization runs on the replace path too: no
    // `query_measurements` row is left with a NULL `commit_timestamp`.
    let unstamped: i64 = client
        .query_one(
            "SELECT count(*) FROM query_measurements WHERE commit_timestamp IS NULL",
            &[],
        )?
        .get(0);
    assert_eq!(unstamped, 0, "replace load left commit_timestamp NULLs");
    Ok(())
}
