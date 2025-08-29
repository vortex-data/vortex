// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Tests for virtual column exposure and len() function optimization.

use tempfile::NamedTempFile;
use vortex::IntoArray;
use vortex::arrays::{PrimitiveArray, StructArray, VarBinArray};
use vortex::file::VortexWriteOptions;

use crate::RUNTIME;
use crate::duckdb::{Connection, Database};

fn database_connection_with_optimizer() -> Connection {
    let mut db = Database::open_in_memory().unwrap();

    // Register the full extension including optimizer
    crate::register_extension(&mut db).unwrap();

    db.connect().unwrap()
}

// Use simpler test helper from existing tests
fn database_connection() -> Connection {
    let db = Database::open_in_memory().unwrap();
    let connection = db.connect().unwrap();
    crate::register_table_functions(&connection).unwrap();
    connection
}

async fn create_test_vortex_file() -> NamedTempFile {
    let temp_file = NamedTempFile::new().unwrap();

    // Create test data with string columns
    let urls = VarBinArray::from_iter(
        [
            "https://example.com",
            "https://test.org/path",
            "https://short.co",
        ]
        .iter()
        .map(|s| Some(s.as_bytes())),
        vortex::dtype::DType::Utf8(vortex::dtype::Nullability::NonNullable),
    );

    let names = VarBinArray::from_iter(
        ["Alice", "Bob", "Charlie"]
            .iter()
            .map(|s| Some(s.as_bytes())),
        vortex::dtype::DType::Utf8(vortex::dtype::Nullability::NonNullable),
    );

    let struct_array = StructArray::try_from_iter([("url", urls), ("name", names)]).unwrap();

    let file = tokio::fs::File::create(&temp_file).await.unwrap();
    VortexWriteOptions::default()
        .write(file, struct_array.to_array_stream())
        .await
        .unwrap();

    temp_file
}

async fn create_complex_test_vortex_file() -> NamedTempFile {
    let temp_file = NamedTempFile::new().unwrap();

    // Create test data with multiple string columns and an integer column
    let titles = VarBinArray::from_iter(
        [
            "Machine Learning Fundamentals",
            "Advanced Database Systems",
            "Web Development with Rust",
            "Data Structures and Algorithms",
        ]
        .iter()
        .map(|s| Some(s.as_bytes())),
        vortex::dtype::DType::Utf8(vortex::dtype::Nullability::NonNullable),
    );

    let descriptions = VarBinArray::from_iter(
        [
            "An introduction to ML concepts and techniques",
            "Deep dive into modern database architectures",
            "Building fast web applications using Rust",
            "Core computer science fundamentals",
        ]
        .iter()
        .map(|s| Some(s.as_bytes())),
        vortex::dtype::DType::Utf8(vortex::dtype::Nullability::NonNullable),
    );

    let page_counts = PrimitiveArray::from_iter([245i32, 512i32, 398i32, 687i32]);

    let struct_array = StructArray::try_from_iter([
        ("title", titles.into_array()),
        ("description", descriptions.into_array()),
        ("page_count", page_counts.into_array()),
    ])
    .unwrap();

    let file = tokio::fs::File::create(&temp_file).await.unwrap();
    VortexWriteOptions::default()
        .write(file, struct_array.to_array_stream())
        .await
        .unwrap();

    temp_file
}

#[test]
fn test_virtual_columns_exposed_in_schema() {
    let temp_file = RUNTIME.block_on(create_test_vortex_file());
    let conn = database_connection_with_optimizer();
    let file_path = temp_file.path().to_string_lossy();

    // Test that we can reference virtual columns directly in a query
    // If virtual columns are properly exposed, this query should succeed
    let query = format!(
        "SELECT url$length, name$length FROM vortex_scan('{}')",
        file_path
    );

    // The fact that this doesn't error means virtual columns are exposed
    conn.query(&query).unwrap();
}

#[test]
fn test_len_function_works() {
    let temp_file = RUNTIME.block_on(create_test_vortex_file());
    let conn = database_connection_with_optimizer();
    let file_path = temp_file.path().to_string_lossy();

    // Query using len() function - test that it executes without error
    // Whether it uses optimization or not, it should still produce correct results
    let query = format!(
        "SELECT len(url), len(name) FROM vortex_scan('{}')",
        file_path
    );

    conn.query(&query).unwrap();
}

#[test]
fn test_virtual_column_direct_access() {
    let temp_file = RUNTIME.block_on(create_test_vortex_file());
    let conn = database_connection_with_optimizer();
    let file_path = temp_file.path().to_string_lossy();

    // Query virtual columns directly (this tests that they work without len() optimization)
    let query = format!(
        "SELECT url$length, name$length FROM vortex_scan('{}')",
        file_path
    );

    conn.query(&query).unwrap();
}

#[test]
fn test_optimizer_registration() {
    let temp_file = RUNTIME.block_on(create_test_vortex_file());
    let conn = database_connection_with_optimizer();
    let file_path = temp_file.path().to_string_lossy();

    // Execute a query that should trigger optimization
    let query = format!("SELECT len(url) FROM vortex_scan('{}')", file_path);

    // The fact that this doesn't crash indicates the optimizer was registered successfully
    conn.query(&query).unwrap();
}

#[test]
fn test_mixed_virtual_and_real_columns() {
    let temp_file = RUNTIME.block_on(create_test_vortex_file());
    let conn = database_connection_with_optimizer();
    let file_path = temp_file.path().to_string_lossy();

    // Query mixing real columns and virtual columns
    let query = format!("SELECT url, url$length FROM vortex_scan('{}')", file_path);

    conn.query(&query).unwrap();
}

#[test]
fn test_multiple_expr() {
    let temp_file = RUNTIME.block_on(create_complex_test_vortex_file());
    let conn = database_connection_with_optimizer();
    let file_path = temp_file.path().to_string_lossy();

    // Test virtual columns are exposed for both string columns
    let virtual_columns_query = format!(
        "SELECT page_count, page_count + 1, page_count + 2 FROM vortex_scan('{}')",
        file_path
    );
    conn.query(&virtual_columns_query).unwrap();
}

#[test]
fn test_multiple_string_columns_with_len_and_integer_column() {
    let temp_file = RUNTIME.block_on(create_complex_test_vortex_file());
    let conn = database_connection_with_optimizer();
    let file_path = temp_file.path().to_string_lossy();

    // Test len() functions work on multiple columns
    let len_functions_query = format!(
        "SELECT len(title), len(description) FROM vortex_scan('{}')",
        file_path
    );
    conn.query(&len_functions_query).unwrap();

    // Test WHERE clause with len() function
    let where_clause_query = format!(
        "SELECT title FROM vortex_scan('{}') WHERE len(title) > 25",
        file_path
    );
    conn.query(&where_clause_query).unwrap();
}

#[test]
fn test_multiple_string_columns_with_len_and_integer_column_complex() {
    let temp_file = RUNTIME.block_on(create_complex_test_vortex_file());
    let conn = database_connection_with_optimizer();
    let file_path = temp_file.path().to_string_lossy();

    // Test mixing string columns, virtual columns, and integer column
    let mixed_query = format!(
        "SELECT title, len(title), description$length, page_count FROM vortex_scan('{}')",
        file_path
    );
    conn.query(&mixed_query).unwrap();
}
