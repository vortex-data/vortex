// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Inline JSONL fixture exercising 1 record per kind through the full
//! migration into a tempdir DuckDB. No live S3.

use std::fs::File;
use std::io::Write;

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

fn write_local_dir() -> TempDir {
    let dir = TempDir::new().expect("tempdir");
    {
        let mut f = File::create(dir.path().join("commits.json")).unwrap();
        f.write_all(COMMITS_JSONL.as_bytes()).unwrap();
    }
    {
        let f = File::create(dir.path().join("data.json.gz")).unwrap();
        let mut gz = GzEncoder::new(f, Compression::default());
        gz.write_all(DATA_JSONL.as_bytes()).unwrap();
        gz.finish().unwrap();
    }
    // No file-sizes-*.json.gz to keep the fixture minimal.
    dir
}

#[test]
fn migrate_inline_fixture_populates_each_table() {
    let src_dir = write_local_dir();
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
