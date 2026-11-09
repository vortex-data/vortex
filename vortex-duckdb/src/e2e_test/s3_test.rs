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
    let mut actual = Vec::new();
    for chunk in reader
        .query(&format!("SELECT id, doubled FROM '{uri}'"))
        .unwrap()
    {
        let len = usize::try_from(chunk.len()).unwrap();
        let ids = chunk.get_vector(0);
        let doubled = chunk.get_vector(1);
        assert!((0..chunk.len()).all(|i| !ids.row_is_null(i) && !doubled.row_is_null(i)));
        actual.extend(
            ids.as_slice_with_len::<i64>(len)
                .iter()
                .copied()
                .zip(doubled.as_slice_with_len::<i64>(len).iter().copied()),
        );
    }
    actual.sort_unstable();
    assert_eq!(actual, (0..1000i64).map(|i| (i, i * 2)).collect::<Vec<_>>());
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

    let mut actual = Vec::new();
    for chunk in conn
        .query(&format!(
            "SELECT id FROM '{uri}' WHERE id >= 250 AND doubled = id * 2"
        ))
        .unwrap()
    {
        let values = chunk.get_vector(0);
        assert!((0..chunk.len()).all(|i| !values.row_is_null(i)));
        actual.extend_from_slice(
            values.as_slice_with_len::<i64>(usize::try_from(chunk.len()).unwrap()),
        );
    }
    actual.sort_unstable();
    assert_eq!(actual, (250..1000i64).collect::<Vec<_>>());
}
