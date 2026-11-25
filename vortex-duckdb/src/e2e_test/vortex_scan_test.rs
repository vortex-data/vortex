// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! This module contains tests for the `vortex_scan` table function.

use std::slice;
use std::sync::Arc;

use anyhow::Result;
use geo_types::LineString;
use geo_types::Polygon;
use num_traits::AsPrimitive;
use rstest::rstest;
use tempfile::NamedTempFile;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::BoolArray;
use vortex::array::arrays::ConstantArray;
use vortex::array::arrays::DictArray;
use vortex::array::arrays::FixedSizeListArray;
use vortex::array::arrays::ListArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::VarBinArray;
use vortex::array::validity::Validity;
use vortex::buffer::buffer;
use vortex::dtype::Nullability;
use vortex::dtype::PType;
use vortex::encodings::fastlanes::RLEData;
use vortex::file::WriteOptionsSessionExt;
use vortex::io::runtime::BlockingRuntime;
use vortex::layout::layouts::flat::writer::FlatLayoutStrategy;
use vortex::scalar::PValue;
use vortex::scalar::Scalar;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::RunEnd;
use vortex_array::arrays::varbin::builder::VarBinBuilder;
use vortex_array::dtype::DType;
use vortex_array::dtype::extension::ExtDType;
use vortex_sequence::Sequence;
use vortex_spatial::extension::SpatialMetadata;
use vortex_spatial::extension::WellKnownBinary;
use wkb::writer::WriteOptions;

use crate::RUNTIME;
use crate::SESSION;
use crate::cpp;
use crate::cpp::duckdb_string_t;
use crate::duckdb::Connection;
use crate::duckdb::Database;

fn database_connection() -> Connection {
    let db = Database::open_in_memory().unwrap();
    db.register_vortex_scan_replacement().unwrap();
    crate::initialize(&db).unwrap();
    db.connect().unwrap()
}

fn create_temp_file() -> NamedTempFile {
    NamedTempFile::with_suffix(".vortex").unwrap()
}

async fn write_single_column_vortex_file(field_name: &str, array: impl IntoArray) -> NamedTempFile {
    write_vortex_file([(field_name, array)].into_iter()).await
}

async fn write_vortex_file(
    iter: impl Iterator<Item = (impl AsRef<str>, impl IntoArray)>,
) -> NamedTempFile {
    let temp_file_path = create_temp_file();

    let struct_array = StructArray::try_from_iter(iter).unwrap();
    let mut file = async_fs::File::create(&temp_file_path).await.unwrap();
    SESSION
        .write_options()
        .write(&mut file, struct_array.into_array().to_array_stream())
        .await
        .unwrap();

    temp_file_path
}

trait FromDuckDBValue<T> {
    fn from_duckdb_value(value: &mut T) -> Self;
}

impl FromDuckDBValue<duckdb_string_t> for String {
    fn from_duckdb_value(value: &mut duckdb_string_t) -> Self {
        let slice: &[u8] = unsafe {
            slice::from_raw_parts(
                cpp::duckdb_string_t_data(&raw mut *value) as _,
                cpp::duckdb_string_t_length(*value) as usize,
            )
        };
        String::from_utf8_lossy(slice).to_string()
    }
}

impl FromDuckDBValue<i32> for i32 {
    fn from_duckdb_value(value: &mut i32) -> Self {
        *value
    }
}

fn scan_vortex_file<D, T: FromDuckDBValue<D>>(
    tmp_file: NamedTempFile,
    query: &str,
    col_idx: usize,
) -> Result<Vec<T>> {
    let conn = database_connection();
    let file_path = tmp_file.path().to_string_lossy();
    let formatted_query = query.replace('?', &format!("'{file_path}'"));

    let result = conn.query(&formatted_query)?;

    let mut values = Vec::new();
    for mut chunk in result {
        let len = chunk.len().as_();
        let vec = chunk.get_vector_mut(col_idx);
        assert!((0..len).all(|i| !vec.row_is_null(i as u64)));
        values.extend(
            unsafe { vec.as_slice_mut::<D>(len) }
                .iter_mut()
                .map(T::from_duckdb_value),
        );
    }

    Ok(values)
}

async fn write_vortex_file_with_encodings() -> NamedTempFile {
    let temp_file_path = create_temp_file();

    // 0. Primitive
    let primitive_i32 = buffer![1i32, 2, 3, 4, 5];
    let primitive_f64 = buffer![1.1f64, 2.2, 3.3, 4.4, 5.5];

    // 1. Constant
    let constant_str = ConstantArray::new(Scalar::from("constant_value"), 5);

    // 2. Boolean
    let bool_array = BoolArray::new(
        vec![true, false, true, false, true].into(),
        Validity::NonNullable,
    );

    // 3. Dictionary
    let keys = buffer![0u32, 1, 0, 2, 1];
    let values = VarBinArray::from(vec!["apple", "banana", "cherry"]);
    let dict_array = DictArray::try_new(keys.into_array(), values.into_array()).unwrap();

    // 4. Run-End
    let run_ends = buffer![3u32, 5];
    let run_values = buffer![100i32, 200];
    let mut rle_ctx = SESSION.create_execution_ctx();
    let rle_array =
        RunEnd::try_new(run_ends.into_array(), run_values.into_array(), &mut rle_ctx).unwrap();

    // 5. Sequence array
    let sequence_array = Sequence::try_new(
        PValue::I64(0),
        PValue::I64(10),
        PType::I64,
        Nullability::NonNullable,
        5,
    )
    .unwrap()
    .into_array();

    // 6. VarBin
    let varbin_array = VarBinArray::from(vec!["hello", "world", "vortex", "test", "data"]);

    // 7. List
    let list_values = PrimitiveArray::from_iter([1i32, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    let list_offsets = buffer![0u32, 2, 5, 6, 10, 10]; // [1,2], [3,4,5], [6], [7,8,9,10], []
    let list_array = ListArray::try_new(
        list_values.into_array(),
        list_offsets.into_array(),
        Validity::NonNullable,
        &mut SESSION.create_execution_ctx(),
    )
    .unwrap();

    // 8. Fixed-size list
    let fixed_list_values = PrimitiveArray::from_iter([1i32, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    let fixed_list_array = FixedSizeListArray::try_new(
        fixed_list_values.into_array(),
        2, // 2 elements per list
        Validity::NonNullable,
        5, // 5 lists
    )
    .unwrap();

    // Struct array containing the different encodings.
    let struct_array = StructArray::try_from_iter([
        ("primitive_i32", primitive_i32.into_array()),
        ("primitive_f64", primitive_f64.into_array()),
        ("constant_str", constant_str.into_array()),
        ("bool_col", bool_array.into_array()),
        ("dict_col", dict_array.into_array()),
        ("rle_col", rle_array.into_array()),
        ("sequence_col", sequence_array),
        ("varbin_col", varbin_array.into_array()),
        ("list_col", list_array.into_array()),
        ("fixed_list_col", fixed_list_array.into_array()),
    ])
    .unwrap();

    // Write to file
    let mut file = async_fs::File::create(&temp_file_path).await.unwrap();
    SESSION
        .write_options()
        .write(&mut file, struct_array.into_array().to_array_stream())
        .await
        .unwrap();

    temp_file_path
}

#[expect(clippy::cognitive_complexity)]
#[test]
fn test_vortex_encodings_roundtrip() {
    let file = RUNTIME.block_on(write_vortex_file_with_encodings());
    let conn = database_connection();

    // Test reading back each column type
    let result = conn
        .query(&format!(
            "SELECT * FROM '{}'",
            file.path().to_string_lossy()
        ))
        .unwrap();

    let mut chunk = result.into_iter().next().unwrap();
    let len: usize = chunk.len().as_();
    assert_eq!(len, 5); // 5 rows
    assert_eq!(chunk.column_count(), 10); // 10 columns
    for col in 0..10 {
        assert!((0..len).all(|row| !chunk.get_vector(col).row_is_null(row as u64)));
    }

    // Verify primitive i32 (column 0)
    let primitive_i32_vec = chunk.get_vector(0);
    let primitive_i32_slice = primitive_i32_vec.as_slice_with_len::<i32>(len);
    assert_eq!(primitive_i32_slice, [1, 2, 3, 4, 5]);

    // Verify primitive f64 (column 1)
    let primitive_f64_vec = chunk.get_vector(1);
    let primitive_f64_slice = primitive_f64_vec.as_slice_with_len::<f64>(len);
    assert_eq!(primitive_f64_slice, [1.1, 2.2, 3.3, 4.4, 5.5]);

    // Verify constant string (column 2)
    let constant_vec = chunk.get_vector_mut(2);
    let constant_slice = unsafe { constant_vec.as_slice_mut::<duckdb_string_t>(len) };
    for idx in 0..5 {
        let string_val = String::from_duckdb_value(&mut constant_slice[idx]);
        assert_eq!(string_val, "constant_value");
    }

    // Verify boolean (column 3)
    let bool_vec = chunk.get_vector(3);
    let bool_slice = bool_vec.as_slice_with_len::<bool>(len);
    assert_eq!(bool_slice, [true, false, true, false, true]);

    // Verify dictionary (column 4)
    let dict_vec = chunk.get_vector_mut(4);
    let dict_slice = unsafe { dict_vec.as_slice_mut::<duckdb_string_t>(len) };
    // Keys were [0, 1, 0, 2, 1] and values were ["apple", "banana", "cherry"]
    let expected_dict_values = ["apple", "banana", "apple", "cherry", "banana"];
    for idx in 0..5 {
        let string_val = String::from_duckdb_value(&mut dict_slice[idx]);
        assert_eq!(string_val, expected_dict_values[idx]);
    }

    // Verify RLE (column 5)
    let rle_vec = chunk.get_vector(5);
    let rle_slice = rle_vec.as_slice_with_len::<i32>(len);
    assert_eq!(rle_slice, [100, 100, 100, 200, 200]);

    // Verify sequence (column 6)
    let seq_vec = chunk.get_vector(6);
    let seq_slice = seq_vec.as_slice_with_len::<i64>(len);
    assert_eq!(seq_slice, [0, 10, 20, 30, 40]);

    // Verify varbin (column 7)
    let varbin_vec = chunk.get_vector_mut(7);
    let varbin_slice = unsafe { varbin_vec.as_slice_mut::<duckdb_string_t>(len) };
    let expected_strings = ["hello", "world", "vortex", "test", "data"];
    for i in 0..5 {
        let string_val = String::from_duckdb_value(&mut varbin_slice[i]);
        assert_eq!(string_val, expected_strings[i]);
    }

    // Verify list (column 8)
    // Expected lists: [1,2], [3,4,5], [6], [7,8,9,10], []
    let list_vec = chunk.get_vector(8);
    let list_entries = list_vec.as_slice_with_len::<cpp::duckdb_list_entry>(len);

    // Verify list lengths
    assert_eq!(list_entries[0].length, 2); // [1,2]
    assert_eq!(list_entries[1].length, 3); // [3,4,5]
    assert_eq!(list_entries[2].length, 1); // [6]
    assert_eq!(list_entries[3].length, 4); // [7,8,9,10]
    assert_eq!(list_entries[4].length, 0); // []

    // Verify list offsets are sequential
    assert_eq!(list_entries[0].offset, 0);
    assert_eq!(list_entries[1].offset, 2);
    assert_eq!(list_entries[2].offset, 5);
    assert_eq!(list_entries[3].offset, 6);
    assert_eq!(list_entries[4].offset, 10);

    // Get child vector and verify actual values
    let list_child_len = list_vec.list_vector_get_size();
    assert_eq!(list_child_len, 10);
    let list_child = list_vec.list_vector_get_child();
    let child_values = list_child.as_slice_with_len::<i32>(list_child_len.as_());
    assert_eq!(child_values, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);

    // Verify fixed-size list column (column 9)
    // Expected fixed-size lists: [1,2], [3,4], [5,6], [7,8], [9,10]
    let fixed_list_vec = chunk.get_vector(9);
    let fixed_child = fixed_list_vec.array_vector_get_child();
    let fixed_child_values = fixed_child.as_slice_with_len::<i32>(10); // 10 total child elements
    assert_eq!(fixed_child_values, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
}

#[test]
fn test_fastlanes_rle_roundtrip() {
    let expected: Vec<i32> = (0i32..2048).map(|i| i / 256).collect();
    let file = RUNTIME.block_on(async {
        let mut ctx = SESSION.create_execution_ctx();
        let primitive = PrimitiveArray::from_iter(expected.clone());
        let rle = RLEData::encode(primitive.as_view(), &mut ctx).unwrap();
        write_single_column_vortex_file("rle_col", rle.into_array()).await
    });

    let values: Vec<i32> = scan_vortex_file::<i32, _>(file, "SELECT rle_col FROM ?", 0).unwrap();
    assert_eq!(values, expected);
}

// Spatial extension is not bundled with duckdb. If we're building from a
// commit, don't run this test, since bundling spatial requires openssl-dev
// which is an issue on macos runners.
#[cfg_attr(
    not(duckdb_release),
    ignore = "spatial extension requires a release DuckDB build"
)]
#[test]
fn test_geometry() {
    let file = RUNTIME.block_on(async {
        let rect10 = Polygon::new(
            LineString::from_iter([[0., 0.], [10., 0.], [10., 10.], [0., 10.], [0., 0.]]),
            vec![],
        );
        let mut wkb_binary: Vec<u8> = Vec::new();
        wkb::writer::write_polygon(&mut wkb_binary, &rect10, &WriteOptions::default())
            .expect("serializing WKB");
        let mut geometry = VarBinBuilder::<u32>::with_capacity_in(
            DType::Binary(Nullability::NonNullable),
            10,
            vortex::buffer::BufferAllocatorRef::static_ref(),
        );
        for _ in 0..10 {
            geometry.append_value(wkb_binary.as_slice());
        }
        let geometry = geometry.finish_into_varbin();

        let geometry = ExtensionArray::new(
            ExtDType::<WellKnownBinary>::try_new(
                SpatialMetadata::new(Some("EPSG:32600".to_string())),
                geometry.dtype().nullability(),
            )
            .expect("making extension array")
            .erased(),
            geometry.into_array(),
        )
        .into_array();

        write_single_column_vortex_file("geometry", geometry).await
    });

    let conn = database_connection();
    super::load_spatial(&conn);
    let file_path = file.path().to_string_lossy();
    let result = conn
        .query(&format!("SELECT SUM(ST_Area(geometry)) FROM '{file_path}'"))
        .unwrap();
    let chunk = result.into_iter().next().unwrap();
    let vec = chunk.get_vector(0);
    let area = vec.as_slice_with_len::<f64>(chunk.len().as_())[0];
    assert_eq!(area, 1000.0);
}

/// Vortex allows duplicate struct names but duckdb doesn't. Ensure we can't
/// read a file if names are not unique
#[test]
fn test_duplicate_struct_fields() {
    let array = StructArray::try_from_iter([
        ("a", buffer![1i32, 2, 3].into_array()),
        ("a", buffer![10i64, 20, 30].into_array()),
    ])
    .unwrap();
    let array = StructArray::try_from_iter([("s", array)])
        .unwrap()
        .into_array();
    let path = create_temp_file();
    RUNTIME.block_on(async {
        let mut file = async_fs::File::create(&path).await.unwrap();
        SESSION
            .write_options()
            .with_strategy(Arc::new(FlatLayoutStrategy::default()))
            .write(&mut file, array.to_array_stream())
            .await
            .unwrap()
    });
    let conn = database_connection();
    let path = path.path().to_string_lossy();

    assert!(conn.query(&format!("SELECT s FROM '{path}'")).is_err());
    assert!(
        conn.query(&format!("SELECT string_agg(s::VARCHAR, '') FROM '{path}'"))
            .is_err()
    );
}

/// Runs `query` on the file at `path` and collects its single BIGINT output column.
fn query_i64_column(path: &str, query: &str) -> Result<Vec<i64>> {
    let conn = database_connection();
    let result = conn.query(&query.replace('?', &format!("'{path}'")))?;
    let mut values = Vec::new();
    for chunk in result {
        let vec = chunk.get_vector(0);
        values.extend_from_slice(vec.as_slice_with_len::<i64>(chunk.len().as_()));
    }
    Ok(values)
}

/// Runs `f` with `SCAN_V2` set to `enabled`, serialized against other tests toggling the global
/// switch and always restoring `false` on drop.
fn with_scan_v2<T>(enabled: bool, f: impl FnOnce() -> T) -> T {
    static SCAN_V2_LOCK: parking_lot::Mutex<()> = parking_lot::Mutex::new(());
    struct ResetOnDrop;
    impl Drop for ResetOnDrop {
        fn drop(&mut self) {
            crate::set_scan_v2(false);
        }
    }

    let _lock = SCAN_V2_LOCK.lock();
    crate::set_scan_v2(enabled);
    let _reset = ResetOnDrop;
    f()
}

#[rstest]
#[case::filtered_count("SELECT count(*) FROM ? WHERE b < 100")]
#[case::filtered_max("SELECT max(a) FROM ? WHERE b BETWEEN 300 AND 310")]
#[case::filtered_projection("SELECT a FROM ? WHERE b = 42 ORDER BY a")]
#[case::expression("SELECT a + b FROM ? WHERE a >= 50000 ORDER BY 1")]
#[case::full_scan("SELECT a FROM ? ORDER BY a")]
#[case::empty("SELECT a FROM ? WHERE b > 5000")]
fn test_vortex_scan_v2_matches_v1(#[case] query: &str) -> Result<()> {
    // Many row blocks, so the file is cut into several V2 splits.
    let rows = 100_000i64;
    let file = RUNTIME.block_on(write_vortex_file(
        [
            ("a", (0..rows).collect::<PrimitiveArray>().into_array()),
            (
                "b",
                (0..rows)
                    .map(|i| (i * 37) % 1000)
                    .collect::<PrimitiveArray>()
                    .into_array(),
            ),
        ]
        .into_iter(),
    ));
    let path = file.path().to_string_lossy();

    let v1 = with_scan_v2(false, || query_i64_column(&path, query))?;
    let v2 = with_scan_v2(true, || query_i64_column(&path, query))?;

    assert_eq!(v2, v1);
    Ok(())
}

/// Runs `EXPLAIN {query}` and returns the physical plan text.
fn explain_plan(conn: &Connection, query: &str) -> Result<String> {
    let explain = conn.query(&format!("EXPLAIN {query}"))?;
    let mut plan = String::new();
    for mut chunk in explain {
        let len: usize = chunk.len().as_();
        let vec = chunk.get_vector_mut(1);
        // SAFETY: Column 1 of `EXPLAIN` is a VARCHAR vector with `len` initialized `duckdb_string_t`
        // entries valid for the lifetime of `chunk`.
        for value in unsafe { vec.as_slice_mut::<duckdb_string_t>(len) } {
            plan.push_str(&String::from_duckdb_value(value));
        }
    }
    Ok(plan)
}

#[test]
fn test_hash_join_dynamic_table_filter_v1_and_v2() -> Result<()> {
    let rows = 50_000i64;
    let fact_file = RUNTIME.block_on(write_vortex_file(
        [
            ("id", (0..rows).collect::<PrimitiveArray>().into_array()),
            (
                "val",
                (0..rows)
                    .map(|i| i * 3)
                    .collect::<PrimitiveArray>()
                    .into_array(),
            ),
        ]
        .into_iter(),
    ));
    let dim_file = RUNTIME.block_on(write_vortex_file(
        [
            (
                "id",
                buffer![10_000i64, 20_000, 30_000, 40_000].into_array(),
            ),
            ("label", buffer![1i64, 2, 3, 4].into_array()),
        ]
        .into_iter(),
    ));

    let fact = fact_file.path().to_string_lossy();
    let dim = dim_file.path().to_string_lossy();
    // Filter the build side at runtime so DuckDB pushes an Optional(Dynamic(...)) join filter
    // into the probe-side Vortex scan rather than folding build-side file stats statically.
    let query = format!(
        "SELECT f.val + d.label FROM '{fact}' f \
         JOIN (SELECT id, label FROM '{dim}' WHERE (label * 7) % 2 = 0 OR label = 1) d \
         ON f.id = d.id \
         ORDER BY f.id"
    );

    let conn = database_connection();
    let plan = explain_plan(&conn, &query)?;
    assert!(
        plan.contains("Filters:") && plan.contains("$.id"),
        "expected pushed-down join filter on id in Read Vortex plan:\n{plan}"
    );

    let expected = vec![30_001i64, 60_002, 120_004];

    let v1 = with_scan_v2(false, || query_i64_column(&fact, &query))?;
    assert_eq!(v1, expected);

    let v2 = with_scan_v2(true, || query_i64_column(&fact, &query))?;
    assert_eq!(v2, expected);
    Ok(())
}

#[test]
fn test_top_n_late_materialization_file_row_number() -> Result<()> {
    // Wide payload columns encourage DuckDB's late materialization pass to split the scan into a
    // Top-N scan over `(sort_key, file_row_number)` followed by a row-id fetch with a
    // `file_row_number` table filter.
    let rows = 10_000i64;
    let payload: Vec<String> = (0..rows)
        .map(|i| format!("payload_{i:06}_padding_data"))
        .collect();
    let payload_refs: Vec<&str> = payload.iter().map(String::as_str).collect();
    let file = RUNTIME.block_on(write_vortex_file(
        [
            (
                "k",
                (0..rows)
                    .map(|i| (i * 7919) % rows)
                    .collect::<PrimitiveArray>()
                    .into_array(),
            ),
            ("p1", VarBinArray::from(payload_refs.clone()).into_array()),
            ("p2", VarBinArray::from(payload_refs.clone()).into_array()),
            ("p3", VarBinArray::from(payload_refs.clone()).into_array()),
            ("p4", VarBinArray::from(payload_refs).into_array()),
        ]
        .into_iter(),
    ));
    let path = file.path().to_string_lossy();
    let query = format!("SELECT k, p1, p2, p3, p4 FROM '{path}' ORDER BY k ASC LIMIT 5");

    let conn = database_connection();
    let plan = explain_plan(&conn, &query)?;
    assert!(
        plan.contains("file_row_number"),
        "expected late materialization over file_row_number:\n{plan}"
    );

    let mut actual = Vec::new();
    for mut chunk in conn.query(&query)? {
        let len: usize = chunk.len().as_();
        let ks = chunk.get_vector(0).as_slice_with_len::<i64>(len).to_vec();
        let read_strings = |col: usize, chunk: &mut crate::duckdb::DataChunk| -> Vec<String> {
            let vec = chunk.get_vector_mut(col);
            // SAFETY: Columns 1..=4 are VARCHAR vectors with `len` initialized `duckdb_string_t`
            // entries valid for the lifetime of `chunk`.
            unsafe { vec.as_slice_mut::<duckdb_string_t>(len) }
                .iter_mut()
                .map(String::from_duckdb_value)
                .collect()
        };
        let p1 = read_strings(1, &mut chunk);
        let p2 = read_strings(2, &mut chunk);
        let p3 = read_strings(3, &mut chunk);
        let p4 = read_strings(4, &mut chunk);
        for idx in 0..len {
            actual.push((
                ks[idx],
                p1[idx].clone(),
                p2[idx].clone(),
                p3[idx].clone(),
                p4[idx].clone(),
            ));
        }
    }
    let expected: Vec<_> = (0..5i64)
        .map(|k| {
            let i = (0..rows).find(|&i| (i * 7919) % rows == k).unwrap();
            let p = format!("payload_{i:06}_padding_data");
            (k, p.clone(), p.clone(), p.clone(), p)
        })
        .collect();
    assert_eq!(actual, expected);
    Ok(())
}

#[test]
fn test_in_conjunction_and_struct_extract_filters() -> Result<()> {
    let ids = buffer![1i32, 2, 3, 4, 5, 6].into_array();
    let nullable_vals =
        PrimitiveArray::from_option_iter([Some(10i32), None, Some(30), Some(40), None, Some(60)])
            .into_array();
    let nested = StructArray::try_from_iter([
        ("x", buffer![100i32, 200, 300, 400, 500, 600].into_array()),
        (
            "y",
            VarBinArray::from(vec!["a", "b", "c", "d", "e", "f"]).into_array(),
        ),
    ])?
    .into_array();

    let file = RUNTIME.block_on(write_vortex_file(
        [("id", ids), ("v", nullable_vals), ("s", nested)].into_iter(),
    ));
    let path = file.path().to_string_lossy();
    let conn = database_connection();

    let run = |sql: &str| -> Result<Vec<i32>> {
        let mut out = Vec::new();
        for chunk in conn.query(sql)? {
            let len: usize = chunk.len().as_();
            out.extend_from_slice(chunk.get_vector(0).as_slice_with_len::<i32>(len));
        }
        Ok(out)
    };

    // IN filter + conjunction AND/OR + IS NULL / IS NOT NULL + struct field filter.
    assert_eq!(
        run(&format!(
            "SELECT id FROM '{path}' WHERE id IN (1, 3, 5, 6) AND v IS NOT NULL ORDER BY id"
        ))?,
        vec![1, 3, 6]
    );
    assert_eq!(
        run(&format!(
            "SELECT id FROM '{path}' WHERE v IS NULL OR (id >= 5 AND s.x >= 600) ORDER BY id"
        ))?,
        vec![2, 5, 6]
    );
    assert_eq!(
        run(&format!(
            "SELECT id FROM '{path}' WHERE s.x IN (200, 400) AND s.y = 'd' ORDER BY id"
        ))?,
        vec![4]
    );
    Ok(())
}
