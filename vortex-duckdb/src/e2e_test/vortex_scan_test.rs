// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! This module contains tests for the `vortex_scan` table function.

use std::ffi::CStr;
use std::io::Write;
use std::net::TcpListener;
use std::path::Path;
use std::slice;
use std::str::FromStr;

use anyhow::Result;
use jiff::Span;
use jiff::Timestamp;
use jiff::Zoned;
use jiff::tz;
use jiff::tz::TimeZone;
use num_traits::AsPrimitive;
use tempfile::NamedTempFile;
use vortex::array::IntoArray;
use vortex::array::arrays::BoolArray;
use vortex::array::arrays::ConstantArray;
use vortex::array::arrays::DictArray;
use vortex::array::arrays::FixedSizeListArray;
use vortex::array::arrays::ListArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::VarBinArray;
use vortex::array::arrays::VarBinViewArray;
use vortex::array::validity::Validity;
use vortex::buffer::buffer;
use vortex::dtype::Nullability;
use vortex::dtype::PType;
use vortex::file::WriteOptionsSessionExt;
use vortex::io::runtime::BlockingRuntime;
use vortex::scalar::PValue;
use vortex::scalar::Scalar;
use vortex_runend::RunEnd;
use vortex_sequence::Sequence;

use crate::RUNTIME;
use crate::SESSION;
use crate::cpp;
use crate::cpp::duckdb_string_t;
use crate::cpp::duckdb_timestamp;
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
    let mut chunk = result.into_iter().next().unwrap();
    let len = chunk.len().as_();
    let vec = chunk.get_vector_mut(col_idx);
    T::from_duckdb_value(&mut unsafe { vec.as_slice_mut::<D>(len) }[0])
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
        values.extend(
            unsafe { vec.as_slice_mut::<D>(len) }
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

    let mut file = async_fs::File::create(&temp_file_path).await.unwrap();
    SESSION
        .write_options()
        .write(&mut file, struct_array.into_array().to_array_stream())
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
    let file = RUNTIME.block_on(async {
        let strings = VarBinArray::from(vec!["Hello", "Hi", "Hey"]);
        write_single_column_vortex_file("strings", strings).await
    });

    let result: String =
        scan_vortex_file_single_row(file, "SELECT string_agg(strings, ',') FROM ?", 0);

    assert_eq!(result, "Hello,Hi,Hey");
}

#[test]
fn test_vortex_scan_strings_contains() {
    let file = RUNTIME.block_on(async {
        let strings = VarBinArray::from(vec!["Hello", "Hi", "Hey"]);
        write_single_column_vortex_file("strings", strings).await
    });
    let result: String = scan_vortex_file_single_row(
        file,
        "SELECT string_agg(strings, ',') FROM ? WHERE strings LIKE '%He%'",
        0,
    );

    assert_eq!(result, "Hello,Hey");
}

#[test]
fn test_vortex_scan_integers() {
    let file = RUNTIME.block_on(async {
        let numbers = buffer![1i32, 42, 100, -5, 0];
        write_single_column_vortex_file("number", numbers).await
    });
    let sum: i64 = scan_vortex_file_single_row::<i64, _>(file, "SELECT SUM(number) FROM ?", 0);
    assert_eq!(sum, 138);
}

#[test]
fn test_vortex_scan_integers_in_list() {
    let file = RUNTIME.block_on(async {
        let numbers = buffer![1i32, 42, 100, -5, 0];
        write_single_column_vortex_file("number", numbers).await
    });
    let sum: i64 = scan_vortex_file_single_row::<i64, _>(
        file,
        "SELECT SUM(number) FROM ? WHERE number in (1, 42, -5)",
        0,
    );
    assert_eq!(sum, 38);
}

#[test]
fn test_vortex_scan_integers_between() {
    let file = RUNTIME.block_on(async {
        let numbers = buffer![1i32, 42, 100, -5, 0];
        write_single_column_vortex_file("number", numbers).await
    });
    let sum: i64 = scan_vortex_file_single_row::<i64, _>(
        file,
        "SELECT SUM(number) FROM ? WHERE number > 0 and number < 100",
        0,
    );
    assert_eq!(sum, 43);
}

#[test]
fn test_issue_5927_not_in_does_not_panic() {
    let file = RUNTIME.block_on(async {
        let numbers = buffer![1i32, 42, 100, -5, 0];
        write_single_column_vortex_file("number", numbers).await
    });
    let sum: i64 = scan_vortex_file_single_row::<i64, _>(
        file,
        "SELECT SUM(number) FROM ? WHERE number NOT IN (42, 100)",
        0,
    );
    assert_eq!(sum, -4);
}

#[test]
fn test_vortex_scan_floats() {
    let file = RUNTIME.block_on(async {
        let values = buffer![1.5f64, -2.5, 0.0, 42.42];
        write_single_column_vortex_file("value", values).await
    });
    let count: i64 =
        scan_vortex_file_single_row::<i64, _>(file, "SELECT COUNT(*) FROM ? WHERE value > 0", 0);
    assert_eq!(count, 2);
}

#[test]
fn test_vortex_scan_constant() {
    let file = RUNTIME.block_on(async {
        let constant = ConstantArray::new(Scalar::from(42i32), 100);
        write_single_column_vortex_file("constant", constant).await
    });
    let value: i32 =
        scan_vortex_file_single_row::<i32, _>(file, "SELECT constant FROM ? LIMIT 1", 0);
    assert_eq!(value, 42);
}

#[test]
fn test_vortex_scan_booleans() {
    let file = RUNTIME.block_on(async {
        let flags = vec![true, false, true, true, false];
        let flags_array = BoolArray::new(flags.into(), Validity::NonNullable);
        write_single_column_vortex_file("flag", flags_array).await
    });
    let true_count: i64 =
        scan_vortex_file_single_row::<i64, _>(file, "SELECT COUNT(*) FROM ? WHERE flag = true", 0);
    assert_eq!(true_count, 3);
}

#[test]
fn test_vortex_multi_column() {
    let file = RUNTIME.block_on(async {
        let f1 = BoolArray::new(
            vec![true, false, true, true, false].into(),
            Validity::NonNullable,
        )
        .into_array();
        let f2 = (0..5).collect::<PrimitiveArray>().into_array();
        let f3 = (100..105).collect::<PrimitiveArray>().into_array();
        write_vortex_file([("f1", f1), ("f2", f2), ("f3", f3)].into_iter()).await
    });

    let result: Vec<i32> =
        scan_vortex_file::<i32, _>(file, "SELECT f2 FROM ? WHERE f1 = true and f2 >= 2", 0)
            .unwrap();

    assert_eq!(result, vec![2, 3]);
}

#[test]
fn test_vortex_scan_multiple_files() {
    let (tempdir, _file1, _file2) = RUNTIME.block_on(async {
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
        .query(&format!("SELECT SUM(numbers) FROM '{glob_pattern}'",))
        .unwrap();
    let chunk = result.into_iter().next().unwrap();
    let vec = chunk.get_vector(0);
    let total_sum = vec.as_slice_with_len::<i64>(chunk.len().as_())[0];

    assert_eq!(total_sum, 21);
}

#[test]
fn test_vortex_scan_over_http() {
    let file = RUNTIME.block_on(async {
        let strings = VarBinArray::from(vec!["a", "b", "c"]);
        write_single_column_vortex_file("strings", strings).await
    });

    let file_bytes = std::fs::read(file.path()).unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();

    // Spawn 10 threads because DuckDB does HEAD and GET requests with retries,
    // thus 2 threads, one for each implementation, aren't enough
    std::thread::spawn(move || {
        for _ in 0..10 {
            if let Ok((mut stream, _)) = listener.accept() {
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n",
                    file_bytes.len()
                );
                stream.write_all(response.as_bytes()).unwrap();
                stream.write_all(&file_bytes).unwrap();
            }
        }
    });

    let conn = database_connection();
    conn.query("SET vortex_filesystem = 'duckdb';").unwrap();
    for httpfs_impl in ["httplib", "curl"] {
        println!("Testing httpfs client implementation: {httpfs_impl}");
        conn.query(&format!(
            "SET httpfs_client_implementation = '{httpfs_impl}';"
        ))
        .unwrap();

        let url = format!(
            "http://{}/{}",
            addr,
            file.path().file_name().unwrap().to_string_lossy()
        );
        println!("url={url}, file={}", file.path().display());

        let result = conn
            .query(&format!("SELECT COUNT(*) FROM read_vortex('{url}')"))
            .unwrap();
        let chunk = result.into_iter().next().unwrap();
        let count = chunk
            .get_vector(0)
            .as_slice_with_len::<i64>(chunk.len().as_())[0];

        assert_eq!(count, 3);
    }
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
        .query(&format!("SELECT SUM(number) FROM '{file_path}'",))
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
        .query(&format!("SELECT TSTZ FROM '{file_path}'",))
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

    let file = RUNTIME.block_on(async {
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
        .query(&format!("SELECT string_lists FROM '{file_path}'"))
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

    let file = RUNTIME.block_on(async {
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
        .query(&format!("SELECT nested_string_lists FROM '{file_path}'"))
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

    let file = RUNTIME.block_on(async {
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
        .query(&format!("SELECT COUNT(*) FROM '{file_path}'"))
        .unwrap();
    let chunk = result.into_iter().next().unwrap();
    let vec = chunk.get_vector(0);
    let count = vec.as_slice_with_len::<i64>(chunk.len().as_())[0];
    assert_eq!(count, 5, "Should have 5 lists");

    // Try to access the data - this tests for segfaults.
    let result = conn
        .query(&format!("SELECT int_list FROM '{file_path}'"))
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

    let file = RUNTIME.block_on(async {
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
        .query(&format!("SELECT COUNT(*) FROM '{file_path}'"))
        .unwrap();
    let chunk = result.into_iter().next().unwrap();
    let vec = chunk.get_vector(0);
    let count = vec.as_slice_with_len::<i64>(chunk.len().as_())[0];
    assert_eq!(count, 4, "Should have 4 lists");

    // Try to access the data - this tests for segfaults.
    let result = conn
        .query(&format!("SELECT string_list FROM '{file_path}'"))
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

    let file = RUNTIME.block_on(async {
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
        .query(&format!("SELECT COUNT(*) FROM '{file_path}'"))
        .unwrap();
    let chunk = result.into_iter().next().unwrap();
    let vec = chunk.get_vector(0);
    let count = vec.as_slice_with_len::<i64>(chunk.len().as_())[0];
    assert_eq!(count, 1, "Should have 1 outermost list");

    // Try to access the data - this is the critical test for segfaults.
    let result = conn
        .query(&format!("SELECT ultra_deep FROM '{file_path}'"))
        .unwrap();

    let mut row_count = 0;
    for chunk in result {
        row_count += chunk.len();
        let _vec = chunk.get_vector(0);
    }
    assert_eq!(row_count, 1, "Should have retrieved 1 row");
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
    let rle_array = RunEnd::try_new(run_ends.into_array(), run_values.into_array()).unwrap();

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

    // Verify primitive i32 (column 0)
    let primitive_i32_vec = chunk.get_vector(0);
    let primitive_i32_slice = primitive_i32_vec.as_slice_with_len::<i32>(len);
    assert_eq!(primitive_i32_slice, [1, 2, 3, 4, 5]);

    // Verify primitive f64 (column 1)
    let primitive_f64_vec = chunk.get_vector(1);
    let primitive_f64_slice = primitive_f64_vec.as_slice_with_len::<f64>(len);
    assert!((primitive_f64_slice[0] - 1.1).abs() < f64::EPSILON);
    assert!((primitive_f64_slice[1] - 2.2).abs() < f64::EPSILON);
    assert!((primitive_f64_slice[2] - 3.3).abs() < f64::EPSILON);

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
    let list_child = list_vec.list_vector_get_child();
    let child_values = list_child.as_slice_with_len::<i32>(10); // 10 total child elements
    assert_eq!(child_values, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);

    // Verify fixed-size list column (column 9)
    // Expected fixed-size lists: [1,2], [3,4], [5,6], [7,8], [9,10]
    let fixed_list_vec = chunk.get_vector(9);
    let fixed_child = fixed_list_vec.array_vector_get_child();
    let fixed_child_values = fixed_child.as_slice_with_len::<i32>(10); // 10 total child elements
    assert_eq!(fixed_child_values, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
}
