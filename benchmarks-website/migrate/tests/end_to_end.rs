// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Inline JSONL fixtures driven through the full migration into a
//! tempdir DuckDB. No live S3.

use std::fs::File;
use std::io::Write;
use std::path::Path;

use duckdb::Connection;
use flate2::Compression;
use flate2::write::GzEncoder;
use tempfile::TempDir;
use vortex_bench_migrate::migrate;
use vortex_bench_migrate::source::Source;

const COMMITS_JSONL: &str = r#"{"id":"deadbeef","timestamp":"2026-04-25T00:00:00Z","message":"fixture commit","author":{"name":"A","email":"a@example.com"},"committer":{"name":"C","email":"c@example.com"},"tree_id":"abcd0001","url":"https://example.com/commit/deadbeef"}
"#;

const DATA_JSONL: &str = r#"{"name":"clickbench_q07/datafusion:parquet","commit_id":"deadbeef","unit":"ns","value":42000,"all_runtimes":[41000,42000,43000]}
{"name":"compress time/clickbench","commit_id":"deadbeef","unit":"ns","value":99}
{"name":"vortex size/clickbench","commit_id":"deadbeef","unit":"bytes","value":1024}
{"name":"random-access/taxi/take/parquet-tokio-local-disk","commit_id":"deadbeef","unit":"ns","value":777,"all_runtimes":[700,777,800]}
"#;

/// Build a local-source fixture directory. Caller supplies the contents
/// of `commits.json`, `data.json.gz`, and any number of
/// `file-sizes-*.json.gz` files (name → contents).
fn build_fixture(commits: &str, data: &str, file_sizes: &[(&str, &str)]) -> TempDir {
    let dir = TempDir::new().expect("tempdir");
    write_text(&dir.path().join("commits.json"), commits);
    write_gz(&dir.path().join("data.json.gz"), data);
    for (name, body) in file_sizes {
        write_gz(&dir.path().join(name), body);
    }
    dir
}

fn write_text(path: &Path, body: &str) {
    let mut f = File::create(path).unwrap();
    f.write_all(body.as_bytes()).unwrap();
}

fn write_gz(path: &Path, body: &str) {
    let f = File::create(path).unwrap();
    let mut gz = GzEncoder::new(f, Compression::default());
    gz.write_all(body.as_bytes()).unwrap();
    gz.finish().unwrap();
}

#[test]
fn migrate_inline_fixture_populates_each_table() {
    let src_dir = build_fixture(COMMITS_JSONL, DATA_JSONL, &[]);
    let target_dir = TempDir::new().unwrap();
    let target = target_dir.path().join("v3.duckdb");

    let summary = migrate::run(&Source::Local(src_dir.path().into()), &target).unwrap();

    assert_eq!(summary.records_read, 4, "summary={summary}");
    assert_eq!(summary.uncategorized, 0, "summary={summary}");
    assert_eq!(summary.commits_inserted, 1);
    assert_eq!(summary.query_inserted, 1);
    assert_eq!(summary.compression_time_inserted, 1);
    assert_eq!(summary.compression_size_inserted, 1);
    assert_eq!(summary.random_access_inserted, 1);

    let conn = Connection::open(&target).unwrap();
    let count = |table: &str| -> i64 {
        conn.query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |r| r.get(0))
            .unwrap()
    };
    assert_eq!(count("commits"), 1);
    assert_eq!(count("query_measurements"), 1);
    assert_eq!(count("compression_times"), 1);
    assert_eq!(count("compression_sizes"), 1);
    assert_eq!(count("random_access_times"), 1);

    // Spot-check the v3 column values for each kind.
    let (engine, format, query_idx, value_ns): (String, String, i32, i64) = conn
        .query_row(
            "SELECT engine, format, query_idx, value_ns FROM query_measurements",
            [],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
        )
        .unwrap();
    assert_eq!(engine, "datafusion");
    assert_eq!(format, "parquet");
    assert_eq!(query_idx, 7);
    assert_eq!(value_ns, 42000);

    let (dataset, format, op): (String, String, String) = conn
        .query_row(
            "SELECT dataset, format, op FROM compression_times",
            [],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )
        .unwrap();
    assert_eq!(dataset, "clickbench");
    assert_eq!(format, "vortex-file-compressed");
    assert_eq!(op, "encode");

    let (dataset, format, value_bytes): (String, String, i64) = conn
        .query_row(
            "SELECT dataset, format, value_bytes FROM compression_sizes",
            [],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )
        .unwrap();
    assert_eq!(dataset, "clickbench");
    assert_eq!(format, "vortex-file-compressed");
    assert_eq!(value_bytes, 1024);

    let (dataset, format): (String, String) = conn
        .query_row("SELECT dataset, format FROM random_access_times", [], |r| {
            Ok((r.get(0)?, r.get(1)?))
        })
        .unwrap();
    assert_eq!(dataset, "taxi/take");
    assert_eq!(format, "parquet");
}

#[test]
fn dedup_collision_keeps_one_row() {
    // Two data.json.gz lines whose query-measurement dim columns are
    // identical (same commit / dataset / engine / format / query_idx,
    // and `storage` collapses to "nvme" since `storage` is unset).
    // Different `value`s. The accumulator's HashSet<measurement_id>
    // should drop the second one and bump `summary.deduped`.
    const DATA: &str = r#"{"name":"clickbench_q07/datafusion:parquet","commit_id":"deadbeef","unit":"ns","value":111}
{"name":"clickbench_q07/datafusion:parquet","commit_id":"deadbeef","unit":"ns","value":222}
"#;

    let src_dir = build_fixture(COMMITS_JSONL, DATA, &[]);
    let target_dir = TempDir::new().unwrap();
    let target = target_dir.path().join("v3.duckdb");

    let summary = migrate::run(&Source::Local(src_dir.path().into()), &target).unwrap();

    assert_eq!(summary.records_read, 2, "summary={summary}");
    assert_eq!(summary.query_inserted, 1, "summary={summary}");
    assert_eq!(summary.deduped, 1, "summary={summary}");

    let conn = Connection::open(&target).unwrap();
    let n: i64 = conn
        .query_row("SELECT COUNT(*) FROM query_measurements", [], |r| r.get(0))
        .unwrap();
    assert_eq!(n, 1);
}

#[test]
fn dedup_with_conflicting_value_ns_is_counted() {
    // Same dim columns, different `value`s. Dedup keeps the first
    // and bumps `deduped_with_conflict` because the dropped row's
    // value_ns differed from the kept row's. This is the signal we
    // care about when watching for silent value-corruption across
    // duplicated v2 emissions.
    const DATA: &str = r#"{"name":"clickbench_q07/datafusion:parquet","commit_id":"deadbeef","unit":"ns","value":111}
{"name":"clickbench_q07/datafusion:parquet","commit_id":"deadbeef","unit":"ns","value":222}
"#;

    let src_dir = build_fixture(COMMITS_JSONL, DATA, &[]);
    let target_dir = TempDir::new().unwrap();
    let target = target_dir.path().join("v3.duckdb");

    let summary = migrate::run(&Source::Local(src_dir.path().into()), &target).unwrap();

    assert_eq!(summary.deduped, 1, "summary={summary}");
    assert_eq!(summary.deduped_with_conflict, 1, "summary={summary}");
}

#[test]
fn dedup_with_matching_value_ns_does_not_count_conflict() {
    // Same dim columns AND identical `value`s. Dedup still drops the
    // duplicate, but `deduped_with_conflict` stays 0.
    const DATA: &str = r#"{"name":"clickbench_q07/datafusion:parquet","commit_id":"deadbeef","unit":"ns","value":111}
{"name":"clickbench_q07/datafusion:parquet","commit_id":"deadbeef","unit":"ns","value":111}
"#;

    let src_dir = build_fixture(COMMITS_JSONL, DATA, &[]);
    let target_dir = TempDir::new().unwrap();
    let target = target_dir.path().join("v3.duckdb");

    let summary = migrate::run(&Source::Local(src_dir.path().into()), &target).unwrap();

    assert_eq!(summary.deduped, 1, "summary={summary}");
    assert_eq!(summary.deduped_with_conflict, 0, "summary={summary}");
}

#[test]
fn compression_size_data_and_file_sizes_merge() {
    // A `vortex size/tpch` record from data.json.gz and a
    // file-sizes-tpch-nvme.json.gz row covering the same (commit,
    // dataset, format, SF) tuple should produce the *same*
    // measurement_id so the in-memory accumulator merges them into
    // one row instead of two.
    //
    // Both sources use scale_factor "1.0", which both code paths
    // filter out → dataset_variant: None on both sides → matching mid.
    const DATA: &str = r#"{"name":"vortex size/tpch","commit_id":"deadbeef","unit":"bytes","value":200,"dataset":{"tpch":{"scale_factor":"1.0"}}}
"#;
    const FILE_SIZES: &str = r#"{"commit_id":"deadbeef","benchmark":"tpch","scale_factor":"1.0","format":"vortex-file-compressed","file":"part-0.vortex","size_bytes":100}
"#;

    let src_dir = build_fixture(
        COMMITS_JSONL,
        DATA,
        &[("file-sizes-tpch-nvme.json.gz", FILE_SIZES)],
    );
    let target_dir = TempDir::new().unwrap();
    let target = target_dir.path().join("v3.duckdb");

    let summary = migrate::run(&Source::Local(src_dir.path().into()), &target).unwrap();

    assert_eq!(summary.compression_size_inserted, 1, "summary={summary}");

    let conn = Connection::open(&target).unwrap();
    let (n, value_bytes): (i64, i64) = conn
        .query_row(
            "SELECT COUNT(*), SUM(value_bytes) FROM compression_sizes",
            [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .unwrap();
    assert_eq!(n, 1);
    // data.json.gz seeds value_bytes=200, file-sizes adds 100.
    assert_eq!(value_bytes, 300);
}

#[test]
fn empty_author_email_stored_as_null() {
    // v2 sometimes wrote `""` for blank author/email/message. The
    // migrator normalizes those to None so DuckDB stores SQL NULL,
    // letting the UI distinguish "missing metadata" from "empty
    // string". Here author.email is "" — verify the column is NULL,
    // not the empty string.
    const COMMITS: &str = r#"{"id":"deadbeef","timestamp":"2026-04-25T00:00:00Z","message":"fixture","author":{"name":"A","email":""},"committer":{"name":"C","email":"c@example.com"},"tree_id":"abcd0001","url":"https://example.com/commit/deadbeef"}
"#;

    let src_dir = build_fixture(COMMITS, "", &[]);
    let target_dir = TempDir::new().unwrap();
    let target = target_dir.path().join("v3.duckdb");

    migrate::run(&Source::Local(src_dir.path().into()), &target).unwrap();

    let conn = Connection::open(&target).unwrap();
    let is_null: bool = conn
        .query_row(
            "SELECT author_email IS NULL FROM commits WHERE commit_sha = 'deadbeef'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert!(is_null, "empty author.email must store as SQL NULL");

    // Non-empty fields still round-trip as strings.
    let committer_email: String = conn
        .query_row(
            "SELECT committer_email FROM commits WHERE commit_sha = 'deadbeef'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(committer_email, "c@example.com");
}

#[test]
fn open_target_db_removes_orphan_wal() {
    // A `.wal` left from a previous crash with no main file present
    // must still be removed so the next run starts from a known-empty
    // state. Otherwise DuckDB can replay stale WAL into the fresh DB
    // and corrupt subsequent inserts.
    let target_dir = TempDir::new().unwrap();
    let target = target_dir.path().join("v3.duckdb");
    let wal = target_dir.path().join("v3.duckdb.wal");
    std::fs::write(&wal, b"orphan-wal-bytes").unwrap();
    assert!(wal.exists(), "precondition: orphan wal staged");
    assert!(!target.exists(), "precondition: no main db file");

    let _conn = migrate::open_target_db(&target).unwrap();

    // The migrator opens the DB after sweeping the WAL; DuckDB may
    // recreate its own wal under load, but our pre-existing orphan
    // bytes must not survive the sweep. We assert by content: either
    // the path is missing, or its contents differ from the orphan we
    // staged.
    if wal.exists() {
        let now = std::fs::read(&wal).unwrap();
        assert_ne!(
            now, b"orphan-wal-bytes",
            "orphan wal bytes must not survive open_target_db"
        );
    }
}

#[test]
fn file_sizes_unknown_id_falls_back_to_unknown_prefix() {
    // A file-sizes-*.json.gz whose id isn't in
    // `KNOWN_FILE_SIZES_SUITES`, with an empty `benchmark` field, used
    // to surface as a bare id like `mystery-suite` and render as a
    // dataset name. The migrator now prefixes those with `unknown:`
    // so the UI can flag them.
    const FILE_SIZES: &str = r#"{"commit_id":"deadbeef","benchmark":"","scale_factor":"","format":"vortex-file-compressed","file":"part-0.vortex","size_bytes":1000}
"#;

    let src_dir = build_fixture(
        COMMITS_JSONL,
        "",
        &[("file-sizes-mystery-suite.json.gz", FILE_SIZES)],
    );
    let target_dir = TempDir::new().unwrap();
    let target = target_dir.path().join("v3.duckdb");

    migrate::run(&Source::Local(src_dir.path().into()), &target).unwrap();

    let conn = Connection::open(&target).unwrap();
    let dataset: String = conn
        .query_row("SELECT dataset FROM compression_sizes", [], |r| r.get(0))
        .unwrap();
    assert_eq!(dataset, "unknown:mystery-suite");
}

#[test]
fn file_sizes_known_id_uses_id_directly() {
    // For a KNOWN_FILE_SIZES_SUITES id, the fallback path keeps the
    // raw id (no `unknown:` prefix). `clickbench-nvme` is on the list.
    const FILE_SIZES: &str = r#"{"commit_id":"deadbeef","benchmark":"","scale_factor":"","format":"vortex-file-compressed","file":"part-0.vortex","size_bytes":1000}
"#;

    let src_dir = build_fixture(
        COMMITS_JSONL,
        "",
        &[("file-sizes-clickbench-nvme.json.gz", FILE_SIZES)],
    );
    let target_dir = TempDir::new().unwrap();
    let target = target_dir.path().join("v3.duckdb");

    migrate::run(&Source::Local(src_dir.path().into()), &target).unwrap();

    let conn = Connection::open(&target).unwrap();
    let dataset: String = conn
        .query_row("SELECT dataset FROM compression_sizes", [], |r| r.get(0))
        .unwrap();
    assert_eq!(dataset, "clickbench-nvme");
}

#[test]
fn compression_size_data_and_file_sizes_merge_with_canonical_sf() {
    // Same logical SF written as `"10"` on the data.json.gz side and
    // `"10.0"` on the file-sizes side. Both paths must canonicalize
    // to `"10"` so the rows share a `measurement_id` and merge into
    // one compression_sizes row.
    const DATA: &str = r#"{"name":"vortex size/tpch","commit_id":"deadbeef","unit":"bytes","value":200,"dataset":{"tpch":{"scale_factor":"10"}}}
"#;
    const FILE_SIZES: &str = r#"{"commit_id":"deadbeef","benchmark":"tpch","scale_factor":"10.0","format":"vortex-file-compressed","file":"part-0.vortex","size_bytes":100}
"#;

    let src_dir = build_fixture(
        COMMITS_JSONL,
        DATA,
        &[("file-sizes-tpch-nvme-10.json.gz", FILE_SIZES)],
    );
    let target_dir = TempDir::new().unwrap();
    let target = target_dir.path().join("v3.duckdb");

    let summary = migrate::run(&Source::Local(src_dir.path().into()), &target).unwrap();

    assert_eq!(summary.compression_size_inserted, 1, "summary={summary}");
    let conn = Connection::open(&target).unwrap();
    let (n, value_bytes, dataset_variant): (i64, i64, String) = conn
        .query_row(
            "SELECT COUNT(*), SUM(value_bytes), MAX(dataset_variant) FROM compression_sizes",
            [],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )
        .unwrap();
    assert_eq!(n, 1);
    // data.json.gz seeds 200, file-sizes adds 100.
    assert_eq!(value_bytes, 300);
    assert_eq!(dataset_variant, "10");
}

#[test]
fn summary_counts_match_actual_rows_on_success() {
    // Sister test to migrate::tests::flush_all_does_not_overcount_on_failure.
    // On a fully successful run, the post-flush summary counters must
    // equal `SELECT COUNT(*)` from each fact table. This is the
    // invariant the flush-after-count refactor preserves.
    let src_dir = build_fixture(COMMITS_JSONL, DATA_JSONL, &[]);
    let target_dir = TempDir::new().unwrap();
    let target = target_dir.path().join("v3.duckdb");

    let summary = migrate::run(&Source::Local(src_dir.path().into()), &target).unwrap();

    let conn = Connection::open(&target).unwrap();
    let actual = |table: &str| -> u64 {
        let n: i64 = conn
            .query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |r| r.get(0))
            .unwrap();
        n as u64
    };
    assert_eq!(summary.query_inserted, actual("query_measurements"));
    assert_eq!(
        summary.compression_time_inserted,
        actual("compression_times")
    );
    assert_eq!(
        summary.compression_size_inserted,
        actual("compression_sizes")
    );
    assert_eq!(
        summary.random_access_inserted,
        actual("random_access_times")
    );
}

#[test]
fn file_sizes_sum_into_one_row() {
    // Two file-sizes rows sharing (commit, benchmark, format,
    // scale_factor) and value_bytes 100 + 200 must collapse to a
    // single compression_sizes row with 300.
    const FILE_SIZES: &str = r#"{"commit_id":"deadbeef","benchmark":"clickbench","scale_factor":"1.0","format":"vortex-file-compressed","file":"part-0.vortex","size_bytes":100}
{"commit_id":"deadbeef","benchmark":"clickbench","scale_factor":"1.0","format":"vortex-file-compressed","file":"part-1.vortex","size_bytes":200}
"#;

    let src_dir = build_fixture(
        COMMITS_JSONL,
        "",
        &[("file-sizes-clickbench.json.gz", FILE_SIZES)],
    );
    let target_dir = TempDir::new().unwrap();
    let target = target_dir.path().join("v3.duckdb");

    let summary = migrate::run(&Source::Local(src_dir.path().into()), &target).unwrap();

    assert_eq!(summary.file_size_inserted, 2, "summary={summary}");
    assert_eq!(summary.compression_size_inserted, 1, "summary={summary}");

    let conn = Connection::open(&target).unwrap();
    let n: i64 = conn
        .query_row("SELECT COUNT(*) FROM compression_sizes", [], |r| r.get(0))
        .unwrap();
    assert_eq!(n, 1);
    let value_bytes: i64 = conn
        .query_row("SELECT value_bytes FROM compression_sizes", [], |r| {
            r.get(0)
        })
        .unwrap();
    assert_eq!(value_bytes, 300);
}
