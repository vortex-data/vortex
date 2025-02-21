// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! This module contains tests for the `vortex_scan` table function.

use std::ffi::CStr;
use std::path::Path;
use std::slice;
use std::str::FromStr;

use anyhow::Result;
use jiff::tz::TimeZone;
use jiff::{Span, Timestamp, Zoned, tz};
use num_traits::AsPrimitive;
use tempfile::NamedTempFile;
use vortex::IntoArray;
use vortex::arrays::{
    BoolArray, ConstantArray, FixedSizeListArray, ListArray, PrimitiveArray, StructArray,
    VarBinArray, VarBinViewArray,
};
use vortex::buffer::buffer;
use vortex::file::VortexWriteOptions;
use vortex::scalar::Scalar;
use vortex::validity::Validity;

use crate::cpp;
use crate::cpp::{duckdb_string_t, duckdb_timestamp};
use crate::duckdb::{Connection, Database};

fn database_connection() -> Connection {
    let db = Database::open_in_memory().unwrap();
    let connection = db.connect().unwrap();
    crate::register_table_functions(&connection).unwrap();
    connection
}

fn create_temp_file() -> NamedTempFile {
    NamedTempFile::new().unwrap()
}

async fn write_single_column_vortex_file(field_name: &str, array: impl IntoArray) -> NamedTempFile {
    write_vortex_file([(field_name, array)].into_iter()).await
}

async fn write_vortex_file(
    iter: impl Iterator<Item = (impl AsRef<str>, impl IntoArray)>,
) -> NamedTempFile {
    let temp_file_path = create_temp_file();

    let struct_array = StructArray::try_from_iter(iter).unwrap();
    let mut file = tokio::fs::File::create(&temp_file_path).await.unwrap();
    VortexWriteOptions::default()
        .write(&mut file, struct_array.to_array_stream())
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

impl FromDuckDBValue<i32> for i64 {
    fn from_duckdb_value(value: &mut i32) -> Self {
        *value as i64
    }
}

impl FromDuckDBValue<i64> for i64 {
    fn from_duckdb_value(value: &mut i64) -> Self {
        *value
    }
}

fn scan_vortex_file_single_row<D, T: FromDuckDBValue<D>>(
    tmp_file: NamedTempFile,
    query: &str,
    col_idx: usize,
) -> T {
    let conn = database_connection();
    let file_path = tmp_file.path().to_string_lossy();
    let formatted_query = query.replace('?', &format!("'{file_path}'"));

    let result = conn.query(&formatted_query).unwrap();
    let chunk = result.into_iter().next().unwrap();
    let mut vec = chunk.get_vector(col_idx);
    T::from_duckdb_value(&mut unsafe { vec.as_slice_mut::<D>(chunk.len().as_()) }[0])
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
    for chunk in result {
        let mut vec = chunk.get_vector(col_idx);
        values.extend(
            unsafe { vec.as_slice_mut::<D>(chunk.len().as_()) }
                .iter_mut()
                .map(T::from_duckdb_value),
        );
    }

    Ok(values)
}

async fn write_vortex_file_to_dir(
    dir: &Path,
    field_name: &str,
    array: impl IntoArray,
) -> NamedTempFile {
    let struct_array = StructArray::from_fields(&[(field_name, array.into_array())]).unwrap();
    let temp_file_path = tempfile::Builder::new()
        .suffix(".vortex")
        .tempfile_in(dir)
        .unwrap();

    let mut file = tokio::fs::File::create(&temp_file_path).await.unwrap();
    VortexWriteOptions::default()
        .write(&mut file, struct_array.to_array_stream())
        .await
        .unwrap();

    temp_file_path
}

#[test]
fn test_scan_function_registration() {
    let conn = database_connection();
    let result = conn
        .query("SELECT function_name FROM duckdb_functions() WHERE function_name = 'vortex_scan'")
        .unwrap();
    let chunk = result.into_iter().next().unwrap();
    let vec = chunk.get_vector(0);
    let mut result = vec.as_slice_with_len::<duckdb_string_t>(chunk.len().as_())[0];
    let string =
        unsafe { CStr::from_ptr(cpp::duckdb_string_t_data(&raw mut result)).to_string_lossy() };

    assert_eq!(string, "vortex_scan");
}

#[test]
fn test_vortex_scan_strings() {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let file = runtime.block_on(async {
        let strings = VarBinArray::from(vec!["Hello", "Hi", "Hey"]);
        write_single_column_vortex_file("strings", strings).await
    });

    let result: String = scan_vortex_file_single_row(
        file,
        "SELECT string_agg(strings, ',') FROM vortex_scan(?)",
        0,
    );

    assert_eq!(result, "Hello,Hi,Hey");
}

#[test]
fn test_vortex_scan_strings_contains() {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let file = runtime.block_on(async {
        let strings = VarBinArray::from(vec!["Hello", "Hi", "Hey"]);
        write_single_column_vortex_file("strings", strings).await
    });
    let result: String = scan_vortex_file_single_row(
        file,
        "SELECT string_agg(strings, ',') FROM vortex_scan(?) WHERE strings LIKE '%He%'",
        0,
    );

    assert_eq!(result, "Hello,Hey");
}

#[test]
fn test_vortex_scan_integers() {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let file = runtime.block_on(async {
        let numbers = buffer![1i32, 42, 100, -5, 0];
        write_single_column_vortex_file("number", numbers).await
    });
    let sum: i64 =
        scan_vortex_file_single_row::<i64, _>(file, "SELECT SUM(number) FROM vortex_scan(?)", 0);
    assert_eq!(sum, 138);
}

#[test]
fn test_vortex_scan_integers_in_list() {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let file = runtime.block_on(async {
        let numbers = buffer![1i32, 42, 100, -5, 0];
        write_single_column_vortex_file("number", numbers).await
    });
    let sum: i64 = scan_vortex_file_single_row::<i64, _>(
        file,
        "SELECT SUM(number) FROM vortex_scan(?) WHERE number in (1, 42, -5)",
        0,
    );
    assert_eq!(sum, 38);
}

#[test]
fn test_vortex_scan_integers_between() {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let file = runtime.block_on(async {
        let numbers = buffer![1i32, 42, 100, -5, 0];
        write_single_column_vortex_file("number", numbers).await
    });
    let sum: i64 = scan_vortex_file_single_row::<i64, _>(
        file,
        "SELECT SUM(number) FROM vortex_scan(?) WHERE number > 0 and number < 100",
        0,
    );
    assert_eq!(sum, 43);
}

#[test]
fn test_vortex_scan_floats() {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let file = runtime.block_on(async {
        let values = buffer![1.5f64, -2.5, 0.0, 42.42];
        write_single_column_vortex_file("value", values).await
    });
    let count: i64 = scan_vortex_file_single_row::<i64, _>(
        file,
        "SELECT COUNT(*) FROM vortex_scan(?) WHERE value > 0",
        0,
    );
    assert_eq!(count, 2);
}

#[test]
fn test_vortex_scan_constant() {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let file = runtime.block_on(async {
        let constant = ConstantArray::new(Scalar::from(42i32), 100);
        write_single_column_vortex_file("constant", constant).await
    });
    let value: i32 = scan_vortex_file_single_row::<i32, _>(
        file,
        "SELECT constant FROM vortex_scan(?) LIMIT 1",
        0,
    );
    assert_eq!(value, 42);
}

#[test]
fn test_vortex_scan_booleans() {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let file = runtime.block_on(async {
        let flags = vec![true, false, true, true, false];
        let flags_array = BoolArray::from_bit_buffer(flags.into(), Validity::NonNullable);
        write_single_column_vortex_file("flag", flags_array).await
    });
    let true_count: i64 = scan_vortex_file_single_row::<i64, _>(
        file,
        "SELECT COUNT(*) FROM vortex_scan(?) WHERE flag = true",
        0,
    );
    assert_eq!(true_count, 3);
}

#[test]
fn test_vortex_multi_column() {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let file = runtime.block_on(async {
        let f1 = BoolArray::from_bit_buffer(
            vec![true, false, true, true, false].into(),
            Validity::NonNullable,
        )
        .to_array();
        let f2 = (0..5).collect::<PrimitiveArray>().to_array();
        let f3 = (100..105).collect::<PrimitiveArray>().to_array();
        write_vortex_file([("f1", f1), ("f2", f2), ("f3", f3)].into_iter()).await
    });

    let result: Vec<i32> = scan_vortex_file::<i32, _>(
        file,
        "SELECT f2 FROM vortex_scan(?) WHERE f1 = true and f2 >= 2",
        0,
    )
    .unwrap();

    assert_eq!(result, vec![2, 3]);
}

#[test]
fn test_vortex_scan_multiple_files() {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let (tempdir, _file1, _file2) = runtime.block_on(async {
        let tempdir = tempfile::tempdir().unwrap();

        let file1 = write_vortex_file_to_dir(tempdir.path(), "numbers", buffer![1i32, 2, 3]).await;

        let file2 = write_vortex_file_to_dir(tempdir.path(), "numbers", buffer![4i32, 5, 6]).await;

        (tempdir, file1, file2)
    });

    // Create glob pattern to match all .vortex files in the temp directory.
    let glob_pattern = format!("{}/*.vortex", tempdir.path().display());

    // Scan both Vortex files.
    let conn = database_connection();
    let result = conn
        .query(&format!(
            "SELECT SUM(numbers) FROM vortex_scan('{glob_pattern}')",
        ))
        .unwrap();
    let chunk = result.into_iter().next().unwrap();
    let vec = chunk.get_vector(0);
    let total_sum = vec.as_slice_with_len::<i64>(chunk.len().as_())[0];

    assert_eq!(total_sum, 21);
}

#[test]
fn test_write_file() {
    let conn = database_connection();
    let tempdir = tempfile::tempdir().unwrap();
    let file_path = format!("{}/test.vortex", tempdir.path().to_string_lossy());

    conn.query(&format!(
        "copy (select * as number from generate_series(10)) to '{file_path}' (FORMAT VORTEX);",
    ))
    .unwrap();

    let result = conn
        .query(&format!(
            "SELECT SUM(number) FROM vortex_scan('{file_path}')",
        ))
        .unwrap();
    let chunk = result.into_iter().next().unwrap();
    let vec = chunk.get_vector(0);
    let total_sum = vec.as_slice_with_len::<i64>(chunk.len().as_())[0];

    assert_eq!(total_sum, 55);
}

#[test]
fn test_write_timestamps() {
    let conn = database_connection();
    let tempdir = tempfile::tempdir().unwrap();
    let file_path = format!("{}/test.vortex", tempdir.path().to_string_lossy());

    conn.query(&format!(
        "COPY (SELECT '2025-05-03 16:19:14.338895-07'::timestamptz as TSTZ) TO '{file_path}' (FORMAT VORTEX);",
    ))
        .unwrap();

    let result = conn
        .query(&format!("SELECT TSTZ FROM vortex_scan('{file_path}')",))
        .unwrap();
    let chunk = result.into_iter().next().unwrap();
    let vec = chunk.get_vector(0);
    let timestamp = vec.as_slice_with_len::<duckdb_timestamp>(chunk.len().as_())[0];

    assert_eq!(
        Timestamp::UNIX_EPOCH
            .checked_add(Span::new().try_microseconds(timestamp.micros).unwrap())
            .unwrap()
            .to_zoned(TimeZone::fixed(tz::offset(-7))),
        Zoned::from_str("2025-05-03 16:19:14.338895-07[-07]").unwrap()
    );
}

#[test]
fn test_vortex_scan_fixed_size_list_utf8() {
    // Test a simple FixedSizeList of Utf8 strings to ensure proper materialization.

    let runtime = tokio::runtime::Runtime::new().unwrap();
    let file = runtime.block_on(async {
        // Create a large number of strings to stress test.
        let strings: Vec<&str> = (0..24)
            .map(|i| match i % 6 {
                0 => "first",
                1 => "second",
                2 => "third",
                3 => "fourth",
                4 => "fifth",
                _ => "sixth",
            })
            .collect();

        let strings_array = VarBinViewArray::from_iter_str(strings);

        // Create fixed-size lists of strings.
        let fsl = FixedSizeListArray::new(
            strings_array.into_array(),
            4, // 4 strings per list
            Validity::AllValid,
            6, // 6 lists total
        );

        write_single_column_vortex_file("string_lists", fsl).await
    });

    let conn = database_connection();
    let file_path = file.path().to_string_lossy();

    // Query the structure.
    let result = conn
        .query(&format!(
            "SELECT string_lists FROM vortex_scan('{file_path}')"
        ))
        .unwrap();

    let mut row_count = 0;
    for chunk in result {
        row_count += chunk.len();
        // Accessing the structure should not cause a segfault.
        let _vec = chunk.get_vector(0);
    }
    assert_eq!(row_count, 6, "Should have retrieved 6 lists");
}

#[test]
fn test_vortex_scan_nested_fixed_size_list_utf8() {
    // Regression test for a segfault that occurs inside query 7 and 8 of the `statpopgen` benchmark
    // when running with `FixedSizeList` instead of `List`.

    // Test FixedSizeList of FixedSizeList of Utf8 to ensure proper materialization.

    let runtime = tokio::runtime::Runtime::new().unwrap();
    let file = runtime.block_on(async {
        // Create a large number of strings to stress test.
        let strings: Vec<&str> = (0..24)
            .map(|i| match i % 6 {
                0 => "first",
                1 => "second",
                2 => "third",
                3 => "fourth",
                4 => "fifth",
                _ => "sixth",
            })
            .collect();

        let strings_array = VarBinViewArray::from_iter_str(strings);

        // Create inner fixed-size lists.
        let inner_fsl = FixedSizeListArray::new(
            strings_array.into_array(),
            4, // 4 strings per inner list
            Validity::AllValid,
            6, // 6 inner lists
        );

        // Create outer fixed-size list of lists.
        let outer_fsl = FixedSizeListArray::new(
            inner_fsl.into_array(),
            3, // 3 inner lists per outer list
            Validity::AllValid,
            2, // 2 outer lists
        );

        write_single_column_vortex_file("nested_string_lists", outer_fsl).await
    });

    let conn = database_connection();
    let file_path = file.path().to_string_lossy();

    // Query the nested structure.
    let result = conn
        .query(&format!(
            "SELECT nested_string_lists FROM vortex_scan('{file_path}')"
        ))
        .unwrap();

    let mut row_count = 0;
    for chunk in result {
        row_count += chunk.len();
        // Accessing the nested structure should not cause a segfault.
        let _vec = chunk.get_vector(0);
    }
    assert_eq!(row_count, 2, "Should have retrieved 2 outer lists");
}

#[test]
fn test_vortex_scan_list_of_ints() {
    // Test a simple List of integers.
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let file = runtime.block_on(async {
        // Create integers that will be grouped into lists.
        let integers = PrimitiveArray::from_iter([
            10i32, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150,
        ]);

        // Create variable-length lists using offsets.
        // List 0: [10, 20, 30] (indices 0-2)
        // List 1: [40, 50, 60, 70] (indices 3-6)
        // List 2: [80] (indices 7-7)
        // List 3: [90, 100, 110, 120, 130] (indices 8-12)
        // List 4: [140, 150] (indices 13-14)
        let offsets = buffer![0i32, 3, 7, 8, 13, 15];
        let list_array = ListArray::try_new(
            integers.into_array(),
            offsets.into_array(),
            Validity::AllValid,
        )
        .unwrap();

        write_single_column_vortex_file("int_list", list_array).await
    });

    let conn = database_connection();
    let file_path = file.path().to_string_lossy();

    // Query the list structure to verify row count.
    let result = conn
        .query(&format!("SELECT COUNT(*) FROM vortex_scan('{file_path}')"))
        .unwrap();
    let chunk = result.into_iter().next().unwrap();
    let vec = chunk.get_vector(0);
    let count = vec.as_slice_with_len::<i64>(chunk.len().as_())[0];
    assert_eq!(count, 5, "Should have 5 lists");

    // Try to access the data - this tests for segfaults.
    let result = conn
        .query(&format!("SELECT int_list FROM vortex_scan('{file_path}')"))
        .unwrap();

    let mut row_count = 0;
    for chunk in result {
        row_count += chunk.len();
        let _vec = chunk.get_vector(0);
    }
    assert_eq!(row_count, 5, "Should have retrieved 5 rows");
}

#[test]
fn test_vortex_scan_list_of_utf8() {
    // Test a simple List of UTF8 strings.
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let file = runtime.block_on(async {
        // Create UTF8 strings that will be grouped into lists.
        let strings = VarBinViewArray::from_iter_str(vec![
            "apple",
            "banana",
            "cherry",
            "date",
            "elderberry",
            "fig",
            "grape",
            "honeydew",
            "kiwi",
            "lemon",
            "mango",
            "nectarine",
        ]);

        // Create variable-length lists using offsets.
        // List 0: [apple, banana, cherry] (indices 0-2)
        // List 1: [date, elderberry] (indices 3-4)
        // List 2: [fig, grape, honeydew, kiwi] (indices 5-8)
        // List 3: [lemon, mango, nectarine] (indices 9-11)
        let offsets = buffer![0i32, 3, 5, 9, 12];
        let list_array = ListArray::try_new(
            strings.into_array(),
            offsets.into_array(),
            Validity::AllValid,
        )
        .unwrap();

        write_single_column_vortex_file("string_list", list_array).await
    });

    let conn = database_connection();
    let file_path = file.path().to_string_lossy();

    // Query the list structure to verify row count.
    let result = conn
        .query(&format!("SELECT COUNT(*) FROM vortex_scan('{file_path}')"))
        .unwrap();
    let chunk = result.into_iter().next().unwrap();
    let vec = chunk.get_vector(0);
    let count = vec.as_slice_with_len::<i64>(chunk.len().as_())[0];
    assert_eq!(count, 4, "Should have 4 lists");

    // Try to access the data - this tests for segfaults.
    let result = conn
        .query(&format!(
            "SELECT string_list FROM vortex_scan('{file_path}')"
        ))
        .unwrap();

    let mut row_count = 0;
    for chunk in result {
        row_count += chunk.len();
        let _vec = chunk.get_vector(0);
    }
    assert_eq!(row_count, 4, "Should have retrieved 4 rows");
}

#[test]
fn test_vortex_scan_ultra_deep_nesting() {
    // Test ultra-deep nesting: Multiple levels of FSL and List combinations with UTF8.
    // FSL[List[FSL[List[FSL[UTF8]]]]]

    let runtime = tokio::runtime::Runtime::new().unwrap();
    let file = runtime.block_on(async {
        // Level 1: Create base UTF8 strings - need a lot for deep nesting.
        let strings = VarBinViewArray::from_iter_str(
            (0..360)
                .map(|i| match i % 10 {
                    0 => "zero",
                    1 => "one",
                    2 => "two",
                    3 => "three",
                    4 => "four",
                    5 => "five",
                    6 => "six",
                    7 => "seven",
                    8 => "eight",
                    _ => "nine",
                })
                .collect::<Vec<_>>(),
        );

        // Level 2: Inner-most FixedSizeList of strings.
        let level2_fsl = FixedSizeListArray::new(
            strings.into_array(),
            5, // 5 strings per list
            Validity::AllValid,
            72, // 72 lists at this level
        );

        // Level 3: Variable-length lists of level 2 FSLs.
        let level3_offsets = buffer![0i32, 3, 6, 8, 12, 15, 18, 20, 24, 27, 30, 32, 36];
        let level3_list = ListArray::try_new(
            level2_fsl.into_array(),
            level3_offsets.into_array(),
            Validity::AllValid,
        )
        .unwrap();

        // Level 4: FixedSizeList of level 3 lists.
        let level4_fsl = FixedSizeListArray::new(
            level3_list.into_array(),
            3, // 3 variable lists per FSL
            Validity::AllValid,
            4, // 4 FSLs at this level
        );

        // Level 5: Variable-length lists of level 4 FSLs.
        let level5_offsets = buffer![0i32, 2, 4];
        let level5_list = ListArray::try_new(
            level4_fsl.into_array(),
            level5_offsets.into_array(),
            Validity::AllValid,
        )
        .unwrap();

        // Level 6: Outermost FixedSizeList.
        let outermost_fsl = FixedSizeListArray::new(
            level5_list.into_array(),
            2, // 2 lists per outermost FSL
            Validity::AllValid,
            1, // 1 outermost FSL
        );

        write_single_column_vortex_file("ultra_deep", outermost_fsl).await
    });

    let conn = database_connection();
    let file_path = file.path().to_string_lossy();

    // Query the ultra-deep nested structure.
    let result = conn
        .query(&format!("SELECT COUNT(*) FROM vortex_scan('{file_path}')"))
        .unwrap();
    let chunk = result.into_iter().next().unwrap();
    let vec = chunk.get_vector(0);
    let count = vec.as_slice_with_len::<i64>(chunk.len().as_())[0];
    assert_eq!(count, 1, "Should have 1 outermost list");

    // Try to access the data - this is the critical test for segfaults.
    let result = conn
        .query(&format!(
            "SELECT ultra_deep FROM vortex_scan('{file_path}')"
        ))
        .unwrap();

    let mut row_count = 0;
    for chunk in result {
        row_count += chunk.len();
        let _vec = chunk.get_vector(0);
    }
    assert_eq!(row_count, 1, "Should have retrieved 1 row");
}
