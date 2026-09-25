// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Result-level tests for `DATE` columns compared against `TIMESTAMP` bounds.
//!
//! DuckDB widens a `DATE` column to `TIMESTAMP` whenever the other side is one, which is what
//! `date '1993-07-01' + interval '3' month` produces. `convert::expr` folds that cast into the
//! literal so the bound can reach the scan as a `DATE` comparison.
//!
//! These tests pin the *semantics* of that rewrite: every bound is evaluated against both a
//! Vortex file and a native DuckDB table built from the same rows, so DuckDB is the oracle.
//! They cover each operator at midnight, strictly inside a day, reversed operand order, and
//! `TIMESTAMP WITH TIME ZONE` across four session timezones.
//!
//! Where the bound runs is asserted elsewhere, against plans rather than results:
//! `slt/duckdb/cast_pushdown.slt` checks that the `date + interval` bound leaves no `FILTER`
//! above the scan while the timezone-dependent one still does, and the TPC-H plans
//! (`slt/tpch/duckdb/plans/q4.slt.no` and its q15 and q20 siblings) show the folded
//! `($.o_orderdate < 1993-10-01)` in the scan's own filter list.

use num_traits::AsPrimitive;
use rstest::rstest;
use tempfile::NamedTempFile;

use crate::duckdb::Connection;
use crate::duckdb::Database;

/// Two years of consecutive dates, spanning the bounds used below.
const ROWS: &str = "SELECT DATE '1993-01-01' + INTERVAL (i) DAY AS d FROM range(0, 730) t(i)";

fn database_connection() -> Connection {
    let db = Database::open_in_memory().unwrap();
    db.register_vortex_scan_replacement().unwrap();
    crate::initialize(&db).unwrap();
    db.connect().unwrap()
}

/// A connection holding [`ROWS`] both as a vortex file and as a native `dates` table, so the
/// same predicate can be run against each.
fn date_fixture() -> (Connection, NamedTempFile) {
    let conn = database_connection();
    let file = NamedTempFile::with_suffix(".vortex").unwrap();
    let path = file.path().to_string_lossy().to_string();

    conn.query(&format!("COPY ({ROWS}) TO '{path}' (FORMAT VORTEX);"))
        .unwrap();
    conn.query(&format!("CREATE TABLE dates AS {ROWS};"))
        .unwrap();
    (conn, file)
}

/// Read back the single `i64` of a one-row, one-column query.
fn query_i64(conn: &Connection, query: &str) -> i64 {
    let result = conn.query(query).unwrap();
    let chunk = result.into_iter().next().unwrap();
    chunk
        .get_vector(0)
        .as_slice_with_len::<i64>(chunk.len().as_())[0]
}

/// Assert the vortex file and the native table agree on how many rows match `filter`.
fn assert_matches_duckdb(conn: &Connection, path: &str, filter: &str) -> i64 {
    let vortex = query_i64(
        conn,
        &format!("SELECT count(*) FROM '{path}' WHERE {filter}"),
    );
    let native = query_i64(conn, &format!("SELECT count(*) FROM dates WHERE {filter}"));
    assert_eq!(vortex, native, "`{filter}` disagrees with DuckDB");
    vortex
}

/// Every bound TPC-H states as `date + interval`, and every operator against a midnight
/// timestamp, keeps DuckDB's own answer. Q4, q15 and q20 are the first three shapes.
#[rstest]
#[case::q4_range("d >= DATE '1993-07-01' AND d < DATE '1993-07-01' + INTERVAL '3' MONTH")]
#[case::q15_range("d >= DATE '1993-01-01' AND d < DATE '1993-01-01' + INTERVAL '3' MONTH")]
#[case::q20_range("d >= DATE '1993-01-01' AND d < DATE '1993-01-01' + INTERVAL '1' YEAR")]
#[case::upper_only("d < DATE '1993-07-01' + INTERVAL '3' MONTH")]
#[case::lower_only("d >= DATE '1993-07-01' + INTERVAL '3' MONTH")]
#[case::midnight_lt("d < TIMESTAMP '1993-07-01 00:00:00'")]
#[case::midnight_lte("d <= TIMESTAMP '1993-07-01 00:00:00'")]
#[case::midnight_gt("d > TIMESTAMP '1993-07-01 00:00:00'")]
#[case::midnight_gte("d >= TIMESTAMP '1993-07-01 00:00:00'")]
#[case::midnight_eq("d = TIMESTAMP '1993-07-01 00:00:00'")]
#[case::reversed("TIMESTAMP '1993-07-01 00:00:00' > d")]
fn date_timestamp_bound_keeps_duckdbs_answer(#[case] filter: &str) {
    let (conn, file) = date_fixture();
    let path = file.path().to_string_lossy().to_string();

    let matched = assert_matches_duckdb(&conn, &path, filter);
    assert!(
        matched > 0,
        "`{filter}` matches nothing, so it proves little"
    );
}

/// A bound strictly inside a day has no exact `DATE` equivalent for `=` and `<>`, and rounds to
/// the day for the inequalities. Either way the count must not move.
#[rstest]
#[case::lt("d < TIMESTAMP '1993-07-01 12:00:00'")]
#[case::lte("d <= TIMESTAMP '1993-07-01 12:00:00'")]
#[case::gt("d > TIMESTAMP '1993-07-01 12:00:00'")]
#[case::gte("d >= TIMESTAMP '1993-07-01 12:00:00'")]
#[case::eq("d = TIMESTAMP '1993-07-01 12:00:00'")]
#[case::not_eq("d <> TIMESTAMP '1993-07-01 12:00:00'")]
fn bound_inside_a_day_keeps_its_meaning(#[case] filter: &str) {
    let (conn, file) = date_fixture();
    let path = file.path().to_string_lossy().to_string();

    assert_matches_duckdb(&conn, &path, filter);
}

/// `TIMESTAMP WITH TIME ZONE` bounds depend on the session timezone, so the fold declines them.
/// The answer must track DuckDB's as the timezone moves the bound across midnight.
#[rstest]
#[case("UTC")]
#[case("Europe/London")]
#[case("America/New_York")]
#[case("Asia/Tokyo")]
fn timestamptz_bound_follows_the_session_timezone(#[case] timezone: &str) {
    let (conn, file) = date_fixture();
    let path = file.path().to_string_lossy().to_string();
    conn.query(&format!("SET TimeZone = '{timezone}';"))
        .unwrap();

    assert_matches_duckdb(&conn, &path, "d < TIMESTAMPTZ '1993-07-01 00:00:00'");
}
