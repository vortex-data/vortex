// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Pushdown tests for the geo scalar functions: every lowered filter must reach the Vortex
//! scan on a direct file scan and through a view.

use num_traits::AsPrimitive;
use rstest::rstest;
use tempfile::NamedTempFile;
use vortex::array::IntoArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::StructArray;
use vortex::file::WriteOptionsSessionExt;
use vortex::io::runtime::BlockingRuntime;
use vortex_array::arrays::ExtensionArray;
use vortex_array::dtype::extension::ExtDType;
use vortex_geo::extension::GeoMetadata;
use vortex_geo::extension::Point;

use crate::RUNTIME;
use crate::SESSION;
use crate::cpp::duckdb_string_t;
use crate::duckdb::Connection;
use crate::duckdb::Database;

/// The test points; the filters below state how many of them they match.
const POINTS: [(f64, f64); 5] = [
    (1.0, 1.0),
    (4.0, 4.0),
    (10.0, 10.0),
    (-1.0, 5.0),
    (2.0, 3.0),
];

/// Matches three of [`POINTS`]: two inside the polygon and one on its boundary.
const ST_INTERSECTS_FILTER: &str =
    "ST_Intersects(geometry, ST_GeomFromText('POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))'))";

/// Matches two of [`POINTS`]: those closer than `3` to `(1, 1)`.
const ST_DISTANCE_FILTER: &str = "ST_Distance(geometry, ST_GeomFromText('POINT (1 1)')) < 3.0";

/// Matches the same two points as [`ST_DISTANCE_FILTER`], phrased as `ST_DWithin`.
const ST_DWITHIN_FILTER: &str = "ST_DWithin(geometry, ST_GeomFromText('POINT (1 1)'), 3.0)";

/// A vortex file whose single column `geometry` holds [`POINTS`] as native `Point`s.
fn native_point_file() -> NamedTempFile {
    RUNTIME.block_on(async {
        let xs = PrimitiveArray::from_iter(POINTS.map(|(x, _)| x)).into_array();
        let ys = PrimitiveArray::from_iter(POINTS.map(|(_, y)| y)).into_array();
        let storage = StructArray::from_fields(&[("x", xs), ("y", ys)])
            .unwrap()
            .into_array();
        let dtype =
            ExtDType::<Point>::try_new(GeoMetadata { crs: None }, storage.dtype().clone()).unwrap();
        let points = ExtensionArray::new(dtype.erased(), storage).into_array();

        let file = NamedTempFile::with_suffix(".vortex").unwrap();
        let table = StructArray::from_fields(&[("geometry", points)]).unwrap();
        let mut writer = async_fs::File::create(&file).await.unwrap();
        SESSION
            .write_options()
            .write(&mut writer, table.into_array().to_array_stream())
            .await
            .unwrap();
        file
    })
}

/// An in-memory database with the Vortex extension initialized, `spatial` loaded, and the
/// spatial overrides registered.
fn spatial_database() -> (Database, Connection) {
    let db = Database::open_in_memory().unwrap();
    db.register_vortex_scan_replacement().unwrap();
    crate::initialize(&db).unwrap();
    let conn = db.connect().unwrap();
    conn.query("INSTALL spatial; LOAD spatial;").unwrap();
    // Must follow `LOAD spatial`: the overrides copy spatial's catalog entries.
    db.register_spatial_overrides().unwrap();
    (db, conn)
}

/// Read back the single `i64` of a one-row, one-column query.
fn query_i64(conn: &Connection, query: &str) -> i64 {
    let result = conn.query(query).unwrap();
    let chunk = result.into_iter().next().unwrap();
    chunk
        .get_vector(0)
        .as_slice_with_len::<i64>(chunk.len().as_())[0]
}

/// The `EXPLAIN` physical plan of `query` as one string.
fn explain_plan(conn: &Connection, query: &str) -> String {
    let explain = conn.query(&format!("EXPLAIN {query}")).unwrap();
    let mut plan = String::new();
    for mut chunk in explain {
        let len = chunk.len().as_();
        let vec = chunk.get_vector_mut(1);
        for value in unsafe { vec.as_slice_mut::<duckdb_string_t>(len) } {
            let slice: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    crate::cpp::duckdb_string_t_data(&raw mut *value) as _,
                    crate::cpp::duckdb_string_t_length(*value) as usize,
                )
            };
            plan.push_str(&String::from_utf8_lossy(slice));
        }
    }
    plan
}

/// Assert that filtering `table` with `filter` counts `expected` points and leaves no `FILTER`
/// operator in the plan, i.e. the predicate ran inside the scan.
fn assert_pushed(conn: &Connection, table: &str, filter: &str, expected: i64) {
    let query = format!("SELECT count(*) FROM {table} WHERE {filter}");
    assert_eq!(query_i64(conn, &query), expected);
    let plan = explain_plan(conn, &query);
    assert!(!plan.contains("FILTER"), "filter was not pushed:\n{plan}");
}

/// Every lowered geo filter pushes on a direct file scan.
#[rstest]
#[case::st_intersects(ST_INTERSECTS_FILTER, 3)]
#[case::st_distance(ST_DISTANCE_FILTER, 2)]
#[case::st_dwithin(ST_DWITHIN_FILTER, 2)]
fn geo_filter_pushes_on_file_scan(#[case] filter: &str, #[case] expected: i64) {
    let file = native_point_file();
    let (_db, conn) = spatial_database();
    let table = format!("'{}'", file.path().to_string_lossy());
    assert_pushed(&conn, &table, filter, expected);
}

/// Every lowered geo filter pushes through a view; without the overrides, DuckDB would keep
/// `ST_Intersects` (can-throw) above the view's projection and hide `ST_DWithin`'s radius.
#[rstest]
#[case::st_intersects(ST_INTERSECTS_FILTER, 3)]
#[case::st_distance(ST_DISTANCE_FILTER, 2)]
#[case::st_dwithin(ST_DWITHIN_FILTER, 2)]
fn geo_filter_pushes_through_view(#[case] filter: &str, #[case] expected: i64) {
    let file = native_point_file();
    let (_db, conn) = spatial_database();
    conn.query(&format!(
        "CREATE VIEW points_v AS SELECT * FROM read_vortex('{}')",
        file.path().to_string_lossy()
    ))
    .unwrap();
    assert_pushed(&conn, "points_v", filter, expected);
}
