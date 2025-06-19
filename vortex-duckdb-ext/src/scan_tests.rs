use std::path::Path;

use duckdb::Connection;
use tempfile::NamedTempFile;
use vortex::IntoArray;
use vortex::arrays::{BoolArray, ConstantArray, PrimitiveArray, StructArray, VarBinArray};
use vortex::file::VortexWriteOptions;
use vortex::scalar::Scalar;
use vortex::validity::Validity;

use crate::duckdb::Database;
use crate::scan::VortexTableFunction;

fn database_connection() -> Connection {
    let db = Database::open_in_memory().unwrap();
    let connection = db.connect().unwrap();
    connection
        .register_table_function::<VortexTableFunction>(c"vortex_scan")
        .unwrap();
    connection
        .register_table_function::<VortexTableFunction>(c"read_vortex")
        .unwrap();
    unsafe { Connection::open_from_raw(db.as_ptr().cast()) }.unwrap()
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
    let file = tokio::fs::File::create(&temp_file_path).await.unwrap();
    VortexWriteOptions::default()
        .write(file, struct_array.to_array_stream())
        .await
        .unwrap();

    temp_file_path
}

fn scan_vortex_file_single_row<T>(tmp_file: NamedTempFile, query: &str, col_idx: usize) -> T
where
    T: duckdb::types::FromSql,
{
    let conn = database_connection();
    conn.prepare(query)
        .unwrap()
        .query_row([tmp_file.path().to_string_lossy()], |row| row.get(col_idx))
        .unwrap()
}

fn scan_vortex_file<T>(
    tmp_file: NamedTempFile,
    query: &str,
    col_idx: usize,
) -> Result<Vec<T>, String>
where
    T: duckdb::types::FromSql,
{
    let conn = database_connection();
    conn.prepare(query)
        .unwrap()
        .query_and_then([tmp_file.path().to_string_lossy()], |row| row.get(col_idx))
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())
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

    let file = tokio::fs::File::create(&temp_file_path).await.unwrap();
    VortexWriteOptions::default()
        .write(file, struct_array.to_array_stream())
        .await
        .unwrap();

    temp_file_path
}

#[test]
fn test_scan_function_registration() {
    let conn = database_connection();
    let result: String = conn
        .prepare("SELECT function_name FROM duckdb_functions() WHERE function_name = 'vortex_scan'")
        .unwrap()
        .query_row([], |row| row.get(0))
        .unwrap();
    assert_eq!(&result, "vortex_scan");
}

#[tokio::test]
async fn test_vortex_scan_strings() {
    let strings = VarBinArray::from(vec!["Hello", "Hi", "Hey"]);
    let file = write_single_column_vortex_file("strings", strings).await;
    let result: String = scan_vortex_file_single_row(
        file,
        "SELECT string_agg(strings, ',') FROM vortex_scan(?)",
        0,
    );
    assert_eq!(result, "Hello,Hi,Hey");
}

#[tokio::test]
async fn test_vortex_scan_strings_contains() {
    let strings = VarBinArray::from(vec!["Hello", "Hi", "Hey"]);
    let file = write_single_column_vortex_file("strings", strings).await;
    let result: String = scan_vortex_file_single_row(
        file,
        "SELECT string_agg(strings, ',') FROM vortex_scan(?) WHERE strings LIKE '%He%'",
        0,
    );
    assert_eq!(result, "Hello,Hey");
}

#[tokio::test]
async fn test_vortex_scan_integers() {
    let numbers = PrimitiveArray::from_iter([1i32, 42, 100, -5, 0]);
    let file = write_single_column_vortex_file("number", numbers).await;
    let sum: i64 = scan_vortex_file_single_row(file, "SELECT SUM(number) FROM vortex_scan(?)", 0);
    assert_eq!(sum, 138);
}

#[tokio::test]
async fn test_vortex_scan_integers_in_list() {
    let numbers = PrimitiveArray::from_iter([1i32, 42, 100, -5, 0]);
    let file = write_single_column_vortex_file("number", numbers).await;
    let sum: i64 = scan_vortex_file_single_row(
        file,
        "SELECT SUM(number) FROM vortex_scan(?) WHERE number in (1, 42, -5)",
        0,
    );
    assert_eq!(sum, 38);
}

#[tokio::test]
async fn test_vortex_scan_integers_between() {
    let numbers = PrimitiveArray::from_iter([1i32, 42, 100, -5, 0]);
    let file = write_single_column_vortex_file("number", numbers).await;
    let sum: i64 = scan_vortex_file_single_row(
        file,
        "SELECT SUM(number) FROM vortex_scan(?) WHERE number > 0 and number < 100",
        0,
    );
    assert_eq!(sum, 43);
}

#[tokio::test]
async fn test_vortex_scan_floats() {
    let values = PrimitiveArray::from_iter([1.5f64, -2.5, 0.0, 42.42]);
    let file = write_single_column_vortex_file("value", values).await;
    let count: i64 = scan_vortex_file_single_row(
        file,
        "SELECT COUNT(*) FROM vortex_scan(?) WHERE value > 0",
        0,
    );
    assert_eq!(count, 2);
}

#[tokio::test]
async fn test_vortex_scan_constant() {
    let constant = ConstantArray::new(Scalar::from(42i32), 100);
    let file = write_single_column_vortex_file("constant", constant).await;
    let value: i32 =
        scan_vortex_file_single_row(file, "SELECT constant FROM vortex_scan(?) LIMIT 1", 0);
    assert_eq!(value, 42);
}

#[tokio::test]
async fn test_vortex_scan_booleans() {
    let flags = vec![true, false, true, true, false];
    let flags_array = BoolArray::new(flags.into(), Validity::NonNullable);
    let file = write_single_column_vortex_file("flag", flags_array).await;
    let true_count: i64 = scan_vortex_file_single_row(
        file,
        "SELECT COUNT(*) FROM vortex_scan(?) WHERE flag = true",
        0,
    );
    assert_eq!(true_count, 3);
}

#[tokio::test]
async fn test_vortex_multi_column() {
    let f1 = BoolArray::new(
        vec![true, false, true, true, false].into(),
        Validity::NonNullable,
    )
    .to_array();
    let f2 = (0..5).collect::<PrimitiveArray>().to_array();
    let f3 = (100..105).collect::<PrimitiveArray>().to_array();
    let file = write_vortex_file([("f1", f1), ("f2", f2), ("f3", f3)].into_iter()).await;

    let result: Vec<i32> = scan_vortex_file(
        file,
        "SELECT f2 FROM vortex_scan(?) WHERE f1 = true and f2 >= 2",
        0,
    )
    .unwrap();

    assert_eq!(result, vec![2, 3]);
}

#[tokio::test]
async fn test_vortex_scan_multiple_files() {
    let tempdir = tempfile::tempdir().unwrap();

    let _file1 = write_vortex_file_to_dir(
        tempdir.path(),
        "numbers",
        PrimitiveArray::from_iter([1i32, 2, 3]),
    )
    .await;

    let _file2 = write_vortex_file_to_dir(
        tempdir.path(),
        "numbers",
        PrimitiveArray::from_iter([4i32, 5, 6]),
    )
    .await;

    // Create glob pattern to match all .vortex files in the temp directory.
    let glob_pattern = format!("{}/*.vortex", tempdir.path().display());

    // Scan both Vortex files.
    let conn = database_connection();
    let total_sum: i64 = conn
        .prepare("SELECT SUM(numbers) FROM vortex_scan(?)")
        .unwrap()
        .query_row([&glob_pattern], |row| row.get(0))
        .unwrap();

    assert_eq!(total_sum, 21);
}
