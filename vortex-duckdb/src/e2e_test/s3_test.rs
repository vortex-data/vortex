// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::duckdb::Connection;
use crate::duckdb::Database;

fn s3_uri() -> String {
    std::env::var("VORTEX_DUCKDB_S3_URI")
        .expect("VORTEX_DUCKDB_S3_URI must be set")
        .trim_end_matches('/')
        .to_string()
}

fn database_connection() -> Connection {
    let db = Database::open_in_memory().unwrap();
    db.register_vortex_scan_replacement().unwrap();
    crate::initialize(&db).unwrap();
    db.connect().unwrap()
}

fn row_count(conn: &Connection, query: &str) -> u64 {
    conn.query(query)
        .unwrap_or_else(|err| panic!("query failed: {query}\n{err}"))
        .into_iter()
        .map(|chunk| chunk.len())
        .sum()
}

#[test]
#[ignore = "requires an S3 endpoint"]
fn s3_roundtrip() {
    let uri = format!("{}/write_then_read.vortex", s3_uri());

    let writer = database_connection();
    writer
        .query(&format!(
            "COPY (SELECT i AS id, i * 2 AS doubled FROM range(1000) t(i)) \
             TO '{uri}' (FORMAT VORTEX)"
        ))
        .expect("COPY TO s3 should succeed");

    let reader = database_connection();
    let total = row_count(&reader, &format!("SELECT id, doubled FROM '{uri}'"));
    assert_eq!(total, 1000);
}

#[test]
#[ignore = "requires an S3 endpoint"]
fn s3_read_with_filter() {
    let uri = format!("{}/read_with_filter.vortex", s3_uri());

    let conn = database_connection();
    conn.query(&format!(
        "COPY (SELECT i AS id, i * 2 AS doubled FROM range(1000) t(i)) \
         TO '{uri}' (FORMAT VORTEX)"
    ))
    .expect("COPY TO s3 should succeed");

    let matching = row_count(
        &conn,
        &format!("SELECT id FROM '{uri}' WHERE id >= 250 AND doubled = id * 2"),
    );
    assert_eq!(matching, 750);
}
