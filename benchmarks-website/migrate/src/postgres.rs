// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! One-shot bulk loader from an existing v3 DuckDB snapshot into Postgres (the
//! v3 -> v4 historical-data migration, PR-3.1).
//!
//! Each of the six tables is read from DuckDB as Arrow batches and streamed into
//! Postgres via `COPY ... FROM STDIN` (text format). The whole load runs inside
//! ONE Postgres transaction: a mid-load failure rolls the target back to empty
//! rather than leaving it half-seeded. `measurement_id` (and `commit_sha` for
//! `commits`) are copied verbatim from DuckDB -- the loader never recomputes the
//! hash, so the existing primary keys the upsert-not-duplicate invariant depends
//! on are preserved exactly. Value fidelity beyond PK presence is checked
//! separately by `Verify --postgres-target` (PR-3.2).
//!
//! TLS: the local rehearsal (PR-3.3 / PR-3.4) connects with `NoTls`; the prod
//! load (PR-5.0) uses `--ca-cert` to trust the RDS CA bundle and verify the host
//! (verify-full-equivalent), via rustls. rustls rather than native-tls because
//! the RDS leaf certificate carries no `serverAuth` Extended Key Usage, which
//! macOS Secure Transport (native-tls's macOS backend) rejects outright;
//! rustls/webpki treats a missing EKU as unrestricted (matching OpenSSL/libpq),
//! so the chain validates on every OS. The verify-full path cannot be exercised
//! without a live RDS, so it is covered by compile + the prod runbook, not an
//! automated test. The DuckDB-read + COPY-text formatting -- where the
//! value-fidelity risk lives -- is covered by the embedded-DuckDB unit tests
//! below (no Docker).

use std::io::Write as _;
use std::path::Path;

use anyhow::Context as _;
use anyhow::Result;
use anyhow::bail;
use arrow_array::Array;
use arrow_array::Float64Array;
use arrow_array::Int32Array;
use arrow_array::Int64Array;
use arrow_array::ListArray;
use arrow_array::RecordBatch;
use arrow_array::StringArray;
use tracing::info;

use self::ColKind::BigInt;
use self::ColKind::BigIntArray;
use self::ColKind::Double;
use self::ColKind::Int;
use self::ColKind::Text;

/// How a column's non-null Arrow value is rendered into a COPY text field.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ColKind {
    /// 32-bit integer (`query_idx`, `iterations`).
    Int,
    /// 64-bit integer (`measurement_id`, `value_ns`, memory + side counters).
    BigInt,
    /// UTF-8 text. Also covers `commits.timestamp`, which is `CAST` to VARCHAR in
    /// the DuckDB SELECT so Postgres re-parses the string into TIMESTAMPTZ.
    Text,
    /// `BIGINT[]` rendered as a Postgres array literal `{a,b,c}`.
    BigIntArray,
    /// 64-bit float (`threshold`); formatted with Rust's shortest round-trip
    /// `Display`, which Postgres re-parses to the same bits.
    Double,
}

/// One column in a table's COPY plan.
struct Column {
    /// Column name. The DuckDB and Postgres shapes are identical by the
    /// migration's Behavior-preservation invariant, so one name serves both the
    /// `SELECT` (read) and the `COPY (...)` column list (write).
    name: &'static str,
    kind: ColKind,
    /// When true the DuckDB SELECT wraps the column in `CAST(... AS VARCHAR)` so a
    /// TIMESTAMPTZ comes back as a UTC string. Only `commits.timestamp`.
    cast_to_text: bool,
}

const fn col(name: &'static str, kind: ColKind) -> Column {
    Column {
        name,
        kind,
        cast_to_text: false,
    }
}

/// One table's load plan: name + ordered columns. Column order mirrors the
/// authoritative DDL in `vortex_bench_server::schema` (and `migrations/001`).
struct TableSpec {
    name: &'static str,
    columns: &'static [Column],
}

const COMMITS: &[Column] = &[
    col("commit_sha", Text),
    Column {
        name: "timestamp",
        kind: Text,
        cast_to_text: true,
    },
    col("message", Text),
    col("author_name", Text),
    col("author_email", Text),
    col("committer_name", Text),
    col("committer_email", Text),
    col("tree_sha", Text),
    col("url", Text),
];

const QUERY_MEASUREMENTS: &[Column] = &[
    col("measurement_id", BigInt),
    col("commit_sha", Text),
    col("dataset", Text),
    col("dataset_variant", Text),
    col("scale_factor", Text),
    col("query_idx", Int),
    col("storage", Text),
    col("engine", Text),
    col("format", Text),
    col("value_ns", BigInt),
    col("all_runtimes_ns", BigIntArray),
    col("peak_physical", BigInt),
    col("peak_virtual", BigInt),
    col("physical_delta", BigInt),
    col("virtual_delta", BigInt),
    col("env_triple", Text),
];

const COMPRESSION_TIMES: &[Column] = &[
    col("measurement_id", BigInt),
    col("commit_sha", Text),
    col("dataset", Text),
    col("dataset_variant", Text),
    col("format", Text),
    col("op", Text),
    col("value_ns", BigInt),
    col("all_runtimes_ns", BigIntArray),
    col("env_triple", Text),
];

const COMPRESSION_SIZES: &[Column] = &[
    col("measurement_id", BigInt),
    col("commit_sha", Text),
    col("dataset", Text),
    col("dataset_variant", Text),
    col("format", Text),
    col("value_bytes", BigInt),
];

const RANDOM_ACCESS_TIMES: &[Column] = &[
    col("measurement_id", BigInt),
    col("commit_sha", Text),
    col("dataset", Text),
    col("format", Text),
    col("value_ns", BigInt),
    col("all_runtimes_ns", BigIntArray),
    col("env_triple", Text),
];

const VECTOR_SEARCH_RUNS: &[Column] = &[
    col("measurement_id", BigInt),
    col("commit_sha", Text),
    col("dataset", Text),
    col("layout", Text),
    col("flavor", Text),
    col("threshold", Double),
    col("value_ns", BigInt),
    col("all_runtimes_ns", BigIntArray),
    col("matches", BigInt),
    col("rows_scanned", BigInt),
    col("bytes_scanned", BigInt),
    col("iterations", Int),
    col("env_triple", Text),
];

/// Every table, in the `commits`-dim-first order a fresh schema creates them.
/// Mirrors `vortex_bench_server::schema::TABLES`. `commits` is loaded first so
/// the fact tables' `commit_sha` references already exist (there is no FK at
/// alpha, but keeping the order matches the server + reads naturally).
const TABLE_SPECS: &[TableSpec] = &[
    TableSpec {
        name: "commits",
        columns: COMMITS,
    },
    TableSpec {
        name: "query_measurements",
        columns: QUERY_MEASUREMENTS,
    },
    TableSpec {
        name: "compression_times",
        columns: COMPRESSION_TIMES,
    },
    TableSpec {
        name: "compression_sizes",
        columns: COMPRESSION_SIZES,
    },
    TableSpec {
        name: "random_access_times",
        columns: RANDOM_ACCESS_TIMES,
    },
    TableSpec {
        name: "vector_search_runs",
        columns: VECTOR_SEARCH_RUNS,
    },
];

/// Per-table row counts from a completed load.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadSummary {
    pub per_table: Vec<(&'static str, u64)>,
}

impl std::fmt::Display for LoadSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (table, rows) in &self.per_table {
            writeln!(f, "{table}: {rows} rows")?;
        }
        Ok(())
    }
}

/// Bulk-load every table from the v3 DuckDB at `duckdb_path` into Postgres at
/// `dsn`, atomically. `ca_cert`, when set, is a PEM CA bundle to trust for a
/// TLS connection (the RDS CA for the prod load); when `None` the connection is
/// plaintext (`NoTls`, the local rehearsal).
///
/// When `replace` is true, every target table is `TRUNCATE`d at the start of the
/// load transaction so the COPYs below replace the existing rows instead of
/// colliding with their primary keys; this is the data-refresh / re-migration
/// path (re-seeding an already-populated target). The TRUNCATE runs inside the
/// same transaction as the COPYs, so a later failure rolls back to the ORIGINAL
/// data rather than leaving the target empty. `TRUNCATE` requires table
/// ownership, so the replace path must connect as the table owner (the RDS
/// master), not the `migrator` role. When `replace` is false the load is the
/// one-shot empty-seed contract and aborts on the first duplicate
/// `measurement_id`.
///
/// The target schema must already be applied -- `migrations/001_initial_schema.sql`
/// plus `006_read_path_perf.sql`, whose denormalized
/// `query_measurements.commit_timestamp` column the post-COPY denormalization
/// UPDATE below writes: `load` only `COPY`s into (optionally first `TRUNCATE`-ing)
/// the existing tables and never creates or alters schema objects.
pub fn load(
    duckdb_path: &Path,
    dsn: &str,
    ca_cert: Option<&Path>,
    replace: bool,
) -> Result<LoadSummary> {
    let config = duckdb::Config::default()
        .access_mode(duckdb::AccessMode::ReadOnly)
        .context("configuring read-only DuckDB access")?;
    let duck = duckdb::Connection::open_with_flags(duckdb_path, config)
        .with_context(|| format!("opening source DuckDB at {}", duckdb_path.display()))?;
    // Render TIMESTAMPTZ -> VARCHAR deterministically in UTC so `commits.timestamp`
    // copies as an unambiguous `+00` string Postgres parses back to the same instant.
    duck.execute_batch("SET TimeZone='UTC';")
        .context("setting DuckDB session timezone to UTC")?;

    let mut client = connect_postgres(dsn, ca_cert)?;
    // One transaction for the whole load: a failure on any table rolls back every
    // table, so the target is never left half-seeded.
    let mut tx = client
        .transaction()
        .context("opening the load transaction")?;

    // For a re-load into an already-populated target, empty every table FIRST,
    // inside this same transaction, so the COPYs below cannot collide with the
    // existing primary keys. Building the list from `TABLE_SPECS` keeps it in sync
    // with the tables actually loaded; a single `TRUNCATE` covers all six (there is
    // no FK at alpha, so no `CASCADE` is needed).
    if replace {
        let tables = TABLE_SPECS
            .iter()
            .map(|spec| spec.name)
            .collect::<Vec<_>>()
            .join(", ");
        tx.batch_execute(&format!("TRUNCATE TABLE {tables}"))
            .context("truncating target tables for the replace load")?;
        info!(tables = %tables, "truncated target tables for replace load");
    }

    let mut per_table = Vec::with_capacity(TABLE_SPECS.len());
    for spec in TABLE_SPECS {
        let rows = load_table(&duck, &mut tx, spec)?;
        info!(table = spec.name, rows, "bulk-loaded table");
        per_table.push((spec.name, rows));
    }

    // Populate the denormalized `query_measurements.commit_timestamp` (migration
    // 006, the read path's latest-per-series sort key). The v3 DuckDB source has
    // no such column, so it cannot ride the COPY's 1:1 column mapping; it is
    // derived here from the already-loaded `commits` dim (`TABLE_SPECS` loads
    // `commits` first), inside the same all-or-nothing transaction. `IS DISTINCT
    // FROM` (rather than `IS NULL`) makes the stamp drift-repairing as well as
    // NULL-filling, the same repair form migrations/README.md documents for
    // operational re-stamping.
    let stamped = tx
        .execute(
            "UPDATE query_measurements q
                SET commit_timestamp = c.timestamp
               FROM commits c
              WHERE c.commit_sha = q.commit_sha
                AND q.commit_timestamp IS DISTINCT FROM c.timestamp",
            &[],
        )
        .context("denormalizing commit_timestamp onto query_measurements")?;
    info!(rows = stamped, "denormalized commit_timestamp");

    tx.commit().context("committing the load transaction")?;
    Ok(LoadSummary { per_table })
}

/// Read one table from DuckDB and COPY it into Postgres within `tx`. Returns the
/// row count, and fails loud if `COPY` reports a different count than was sent.
fn load_table(
    duck: &duckdb::Connection,
    tx: &mut postgres::Transaction<'_>,
    spec: &TableSpec,
) -> Result<u64> {
    let batches = read_batches(duck, spec)?;
    let copy_sql = format!("COPY {} ({}) FROM STDIN", spec.name, column_list(spec));
    let mut writer = tx
        .copy_in(copy_sql.as_str())
        .with_context(|| format!("starting COPY into {}", spec.name))?;
    let mut sent: u64 = 0;
    for batch in &batches {
        let text = batch_to_copy(spec, batch)?;
        writer
            .write_all(text.as_bytes())
            .with_context(|| format!("streaming COPY data for {}", spec.name))?;
        sent += batch.num_rows() as u64;
    }
    let copied = writer
        .finish()
        .with_context(|| format!("finishing COPY into {}", spec.name))?;
    if copied != sent {
        bail!(
            "table {}: COPY committed {copied} rows but {sent} were sent",
            spec.name
        );
    }
    Ok(copied)
}

/// Connect to Postgres, plaintext (`NoTls`) when `ca_cert` is `None` or with a
/// host-verifying TLS connection trusting the given CA PEM otherwise. Shared with
/// the value-verify read path ([`crate::verify::run_postgres_value_verify`]) so
/// the loader and the verifier connect identically.
pub(crate) fn connect_postgres(dsn: &str, ca_cert: Option<&Path>) -> Result<postgres::Client> {
    match ca_cert {
        None => postgres::Client::connect(dsn, postgres::NoTls).context("connecting to Postgres"),
        Some(ca) => {
            // Install a process-default rustls crypto provider once. `install_default`
            // errors if one is already installed (for example a second
            // `connect_postgres` call within the same process), which is harmless
            // here, so the result is intentionally ignored.
            let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
            let pem = std::fs::read(ca)
                .with_context(|| format!("reading CA certificate {}", ca.display()))?;
            // The RDS CA file is a bundle (a root per signing algorithm); add every
            // certificate it contains to the trust set. Host verification is on by
            // construction -- the default `WebPkiServerVerifier` checks the leaf SAN
            // against the connection host -- so trusting the RDS roots is
            // verify-full-equivalent. The DSN should request `sslmode=require`
            // (tokio-postgres understands prefer/require/disable, not verify-full).
            let mut roots = rustls::RootCertStore::empty();
            for cert in rustls_pemfile::certs(&mut pem.as_slice()) {
                let cert = cert.context("parsing a certificate from the CA PEM bundle")?;
                roots
                    .add(cert)
                    .context("adding a CA certificate to the rustls root store")?;
            }
            // Fail loud and early if the `--ca-cert` file held no certificates (an
            // empty file, a key-only PEM, or the wrong path): an empty root store
            // builds fine but then rejects every server cert with an opaque
            // handshake error, which is a confusing way to learn the bundle is wrong.
            if roots.is_empty() {
                bail!("no certificates found in the CA bundle {}", ca.display());
            }
            let config = rustls::ClientConfig::builder()
                .with_root_certificates(roots)
                .with_no_client_auth();
            let connector = tokio_postgres_rustls::MakeRustlsConnect::new(config);
            postgres::Client::connect(dsn, connector).context("connecting to Postgres over TLS")
        }
    }
}

/// The `COPY (...)` column list for `spec`, in DDL order.
fn column_list(spec: &TableSpec) -> String {
    spec.columns
        .iter()
        .map(|c| c.name)
        .collect::<Vec<_>>()
        .join(", ")
}

/// The [`ColKind`] of `column` in `table`, or `None` if the table or column is
/// unknown. Exposed so the value-verify path
/// ([`crate::verify::run_postgres_value_verify`]) reads each value column with
/// the loader's authoritative column type rather than maintaining a second,
/// drift-prone type table.
pub(crate) fn column_kind(table: &str, column: &str) -> Option<ColKind> {
    TABLE_SPECS
        .iter()
        .find(|s| s.name == table)?
        .columns
        .iter()
        .find(|c| c.name == column)
        .map(|c| c.kind)
}

/// The DuckDB `SELECT` that reads `spec`'s columns in DDL order, casting the one
/// TIMESTAMPTZ column to VARCHAR so it arrives as a string.
fn select_sql(spec: &TableSpec) -> String {
    let cols: Vec<String> = spec
        .columns
        .iter()
        .map(|c| {
            if c.cast_to_text {
                format!("CAST({0} AS VARCHAR) AS {0}", c.name)
            } else {
                c.name.to_string()
            }
        })
        .collect();
    format!("SELECT {} FROM {}", cols.join(", "), spec.name)
}

/// Read `spec`'s rows from DuckDB as Arrow batches.
fn read_batches(duck: &duckdb::Connection, spec: &TableSpec) -> Result<Vec<RecordBatch>> {
    let sql = select_sql(spec);
    let mut stmt = duck
        .prepare(&sql)
        .with_context(|| format!("preparing `{sql}`"))?;
    let batches: Vec<RecordBatch> = stmt
        .query_arrow([])
        .with_context(|| format!("reading {} from DuckDB", spec.name))?
        .collect();
    Ok(batches)
}

/// Render one Arrow batch into COPY text-format rows (tab-separated columns,
/// newline-terminated rows). NULLs become `\N`; every other value is formatted
/// per its column kind.
fn batch_to_copy(spec: &TableSpec, batch: &RecordBatch) -> Result<String> {
    if batch.num_columns() != spec.columns.len() {
        bail!(
            "table {}: DuckDB returned {} columns, expected {}",
            spec.name,
            batch.num_columns(),
            spec.columns.len()
        );
    }
    let mut out = String::new();
    for row in 0..batch.num_rows() {
        for (idx, column) in spec.columns.iter().enumerate() {
            if idx > 0 {
                out.push('\t');
            }
            let array = batch.column(idx).as_ref();
            out.push_str(&cell_to_copy(spec.name, column, array, row)?);
        }
        out.push('\n');
    }
    Ok(out)
}

/// Render one cell. `\N` for NULL, else the kind-specific text.
fn cell_to_copy(table: &str, column: &Column, array: &dyn Array, row: usize) -> Result<String> {
    if array.is_null(row) {
        return Ok("\\N".to_string());
    }
    let downcast_err = || {
        format!(
            "table {table}: column {} has an unexpected Arrow type",
            column.name
        )
    };
    let field = match column.kind {
        Int => {
            let a = array
                .as_any()
                .downcast_ref::<Int32Array>()
                .with_context(downcast_err)?;
            a.value(row).to_string()
        }
        BigInt => {
            let a = array
                .as_any()
                .downcast_ref::<Int64Array>()
                .with_context(downcast_err)?;
            a.value(row).to_string()
        }
        Text => {
            let a = array
                .as_any()
                .downcast_ref::<StringArray>()
                .with_context(downcast_err)?;
            escape_copy_text(a.value(row))
        }
        Double => {
            let a = array
                .as_any()
                .downcast_ref::<Float64Array>()
                .with_context(downcast_err)?;
            format_f64(a.value(row))?
        }
        BigIntArray => {
            let a = array
                .as_any()
                .downcast_ref::<ListArray>()
                .with_context(downcast_err)?;
            // `format_bigint_array` attaches its own context on its only fallible
            // step, so the outer `downcast_err` wrap here would be redundant.
            format_bigint_array(a, row)?
        }
    };
    Ok(field)
}

/// Escape a string for a COPY text field: backslash, tab, newline, and carriage
/// return are the COPY-significant characters.
fn escape_copy_text(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '\t' => out.push_str("\\t"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            other => out.push(other),
        }
    }
    out
}

/// Format an `f64` with Rust's shortest round-trip `Display`. Non-finite values
/// are rejected loud: the source DuckDB never holds one (the ingest path guards
/// `is_finite`), and `NaN`/`Inf` have no faithful COPY representation.
fn format_f64(x: f64) -> Result<String> {
    if !x.is_finite() {
        bail!(
            "non-finite f64 ({x}) cannot be bulk-loaded; the source DuckDB should never hold one"
        );
    }
    Ok(format!("{x}"))
}

/// Render a `BIGINT[]` row as the Postgres array literal `{a,b,c}` (`{}` empty).
fn format_bigint_array(arr: &ListArray, row: usize) -> Result<String> {
    let values = arr.value(row);
    let ints = values
        .as_any()
        .downcast_ref::<Int64Array>()
        .context("expected Int64 list items")?;
    let mut out = String::from("{");
    for i in 0..ints.len() {
        if i > 0 {
            out.push(',');
        }
        // The schema declares `all_runtimes_ns` items NOT NULL; emit the Postgres
        // array NULL token defensively if one ever appears rather than a wrong value.
        if ints.is_null(i) {
            out.push_str("NULL");
        } else {
            out.push_str(&ints.value(i).to_string());
        }
    }
    out.push('}');
    Ok(out)
}

#[cfg(test)]
mod tests {
    use vortex_bench_server::family;
    use vortex_bench_server::schema::COMMITS_DDL;

    use super::*;

    fn spec(name: &str) -> &'static TableSpec {
        TABLE_SPECS
            .iter()
            .find(|s| s.name == name)
            .expect("known table")
    }

    fn in_memory_v3() -> duckdb::Connection {
        let conn = duckdb::Connection::open_in_memory().expect("open in-memory duckdb");
        conn.execute_batch("SET TimeZone='UTC';").expect("utc");
        conn.execute_batch(COMMITS_DDL).expect("commits ddl");
        for fam in family::FAMILIES {
            conn.execute_batch(fam.schema_ddl).expect("fact ddl");
        }
        conn
    }

    #[test]
    fn escape_handles_copy_significant_chars() {
        assert_eq!(escape_copy_text("plain"), "plain");
        assert_eq!(escape_copy_text("a\\b"), "a\\\\b");
        assert_eq!(escape_copy_text("a\tb"), "a\\tb");
        assert_eq!(escape_copy_text("a\nb"), "a\\nb");
        assert_eq!(escape_copy_text("a\rb"), "a\\rb");
        // A multibyte char passes through untouched.
        assert_eq!(escape_copy_text("café"), "café");
    }

    #[test]
    fn f64_round_trips_and_rejects_non_finite() {
        assert_eq!(format_f64(0.95).unwrap(), "0.95");
        assert_eq!(format_f64(2.0).unwrap(), "2");
        assert!(format_f64(f64::NAN).is_err());
        assert!(format_f64(f64::INFINITY).is_err());
    }

    #[test]
    fn select_sql_casts_only_the_timestamp() {
        assert_eq!(
            select_sql(spec("commits")),
            "SELECT commit_sha, CAST(timestamp AS VARCHAR) AS timestamp, message, \
             author_name, author_email, committer_name, committer_email, tree_sha, url \
             FROM commits"
        );
        assert_eq!(
            select_sql(spec("compression_sizes")),
            "SELECT measurement_id, commit_sha, dataset, dataset_variant, format, value_bytes \
             FROM compression_sizes"
        );
    }

    #[test]
    fn column_list_is_ddl_order() {
        assert_eq!(
            column_list(spec("compression_sizes")),
            "measurement_id, commit_sha, dataset, dataset_variant, format, value_bytes"
        );
    }

    #[test]
    fn copy_text_for_query_measurements_handles_arrays_nulls_and_escapes() {
        let duck = in_memory_v3();
        // Row 1: optional dims + all four memory columns NULL; backslash in a dim.
        // Row 2: variant NULL, scale_factor present, memory columns set, negative idx.
        duck.execute_batch(
            r#"
            INSERT INTO query_measurements VALUES
              (1, 'abc', 'tpch', 'a\b', '1', 3, 'nvme', 'vortex', 'vortex',
               1000, [1000, 1100], NULL, NULL, NULL, NULL, NULL),
              (2, 'def', 'tpch', NULL, NULL, -5, 's3', 'duckdb', 'parquet',
               2000, [2000], 10, 20, 30, 40, 'x86_64-linux');
            "#,
        )
        .expect("insert query rows");

        let batches = read_batches(&duck, spec("query_measurements")).unwrap();
        let text: String = batches
            .iter()
            .map(|b| batch_to_copy(spec("query_measurements"), b).unwrap())
            .collect();

        let expected = "\
1\tabc\ttpch\ta\\\\b\t1\t3\tnvme\tvortex\tvortex\t1000\t{1000,1100}\t\\N\t\\N\t\\N\t\\N\t\\N\n\
2\tdef\ttpch\t\\N\t\\N\t-5\ts3\tduckdb\tparquet\t2000\t{2000}\t10\t20\t30\t40\tx86_64-linux\n";
        assert_eq!(text, expected);
    }

    #[test]
    fn copy_text_for_commits_renders_utc_timestamp_string() {
        let duck = in_memory_v3();
        duck.execute_batch(
            r#"
            INSERT INTO commits VALUES
              ('sha1', TIMESTAMPTZ '2024-01-15 12:34:56.789+00', 'msg', 'an', 'ae',
               'cn', 'ce', 'tree1', 'http://x'),
              ('sha2', TIMESTAMPTZ '2024-03-02 00:00:00+00', NULL, NULL, NULL,
               NULL, NULL, 'tree2', 'http://y');
            "#,
        )
        .expect("insert commits");

        let batches = read_batches(&duck, spec("commits")).unwrap();
        let text: String = batches
            .iter()
            .map(|b| batch_to_copy(spec("commits"), b).unwrap())
            .collect();

        let expected = "\
sha1\t2024-01-15 12:34:56.789+00\tmsg\tan\tae\tcn\tce\ttree1\thttp://x\n\
sha2\t2024-03-02 00:00:00+00\t\\N\t\\N\t\\N\t\\N\t\\N\ttree2\thttp://y\n";
        assert_eq!(text, expected);
    }

    #[test]
    fn copy_text_for_vector_search_renders_double_and_side_counters() {
        let duck = in_memory_v3();
        duck.execute_batch(
            r#"
            INSERT INTO vector_search_runs VALUES
              (7, 'sha', 'sift', 'flat', 'f32', 0.95, 500, [500, 510], 42, 1000, 64000, 3, NULL);
            "#,
        )
        .expect("insert vector row");

        let batches = read_batches(&duck, spec("vector_search_runs")).unwrap();
        let text: String = batches
            .iter()
            .map(|b| batch_to_copy(spec("vector_search_runs"), b).unwrap())
            .collect();

        assert_eq!(
            text,
            "7\tsha\tsift\tflat\tf32\t0.95\t500\t{500,510}\t42\t1000\t64000\t3\t\\N\n"
        );
    }

    #[test]
    fn empty_table_yields_no_copy_rows() {
        let duck = in_memory_v3();
        let batches = read_batches(&duck, spec("random_access_times")).unwrap();
        let text: String = batches
            .iter()
            .map(|b| batch_to_copy(spec("random_access_times"), b).unwrap())
            .collect();
        assert_eq!(text, "");
    }

    #[test]
    fn copy_text_preserves_literal_backslash_n_and_distinguishes_null() {
        // A `commits.message` whose literal value is the two characters `\N` must
        // escape to `\\N` in COPY text (read back as literal text), NOT the bare
        // `\N` token COPY interprets as SQL NULL; a genuine NULL renders the bare
        // token. This disambiguation is the load-bearing property of text COPY.
        let duck = in_memory_v3();
        duck.execute_batch(
            r#"
            INSERT INTO commits VALUES
              ('s1', TIMESTAMPTZ '2024-01-01 00:00:00+00', chr(92) || 'N', NULL, NULL,
               NULL, NULL, 't1', 'u1'),
              ('s2', TIMESTAMPTZ '2024-01-01 00:00:00+00', NULL, NULL, NULL,
               NULL, NULL, 't2', 'u2');
            "#,
        )
        .expect("insert commits");

        let batches = read_batches(&duck, spec("commits")).unwrap();
        let text: String = batches
            .iter()
            .map(|b| batch_to_copy(spec("commits"), b).unwrap())
            .collect();

        // Row 1's message is the escaped literal `\\N`; row 2's is the `\N` NULL token.
        assert_eq!(
            text,
            "s1\t2024-01-01 00:00:00+00\t\\\\N\t\\N\t\\N\t\\N\t\\N\tt1\tu1\n\
             s2\t2024-01-01 00:00:00+00\t\\N\t\\N\t\\N\t\\N\t\\N\tt2\tu2\n"
        );
    }

    #[test]
    fn copy_text_normalizes_non_utc_offset_timestamp_to_utc() {
        // A non-UTC TIMESTAMPTZ offset must normalize to a `+00` UTC string under
        // the loader's `SET TimeZone='UTC'`, so the same instant copies identically
        // regardless of source offset (a wrong offset would silently shift every
        // commit timestamp).
        let duck = in_memory_v3();
        duck.execute_batch(
            r#"
            INSERT INTO commits VALUES
              ('s', TIMESTAMPTZ '2024-06-15 08:00:00+05:30', 'm', NULL, NULL,
               NULL, NULL, 't', 'u');
            "#,
        )
        .expect("insert commit");

        let batches = read_batches(&duck, spec("commits")).unwrap();
        let text: String = batches
            .iter()
            .map(|b| batch_to_copy(spec("commits"), b).unwrap())
            .collect();

        // 08:00:00+05:30 == 02:30:00 UTC.
        assert_eq!(
            text,
            "s\t2024-06-15 02:30:00+00\tm\t\\N\t\\N\t\\N\t\\N\tt\tu\n"
        );
    }

    #[test]
    fn copy_text_renders_empty_and_negative_arrays() {
        // Empty `all_runtimes_ns` (`[]` -> `{}`) and a negative element
        // (`[-5, 7]` -> `{-5,7}`).
        let duck = in_memory_v3();
        duck.execute_batch(
            r#"
            INSERT INTO compression_times VALUES
              (1, 'sha', 'ds', NULL, 'fmt', 'encode', 100, []::BIGINT[], NULL),
              (2, 'sha', 'ds', NULL, 'fmt', 'decode', 200, [-5, 7], NULL);
            "#,
        )
        .expect("insert compression_times");

        let batches = read_batches(&duck, spec("compression_times")).unwrap();
        let text: String = batches
            .iter()
            .map(|b| batch_to_copy(spec("compression_times"), b).unwrap())
            .collect();

        assert_eq!(
            text,
            "1\tsha\tds\t\\N\tfmt\tencode\t100\t{}\t\\N\n\
             2\tsha\tds\t\\N\tfmt\tdecode\t200\t{-5,7}\t\\N\n"
        );
    }
}
