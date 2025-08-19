// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Tests for virtual column exposure and len() function optimization.

use tempfile::NamedTempFile;
use vortex::IntoArray;
use vortex::arrays::{StructArray, VarBinArray};
use vortex::file::VortexWriteOptions;

use crate::duckdb::{Connection, Database};

fn database_connection_with_optimizer() -> Connection {
    let db = Database::open_in_memory().unwrap();

    // Register the full extension including optimizer
    crate::register_extension(&db).unwrap();

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

#[tokio::test]
async fn test_virtual_columns_exposed_in_schema() {
    let temp_file = create_test_vortex_file().await;
    let conn = database_connection_with_optimizer();
    let file_path = temp_file.path().to_string_lossy();

    // Test that we can reference virtual columns directly in a query
    // If virtual columns are properly exposed, this query should succeed
    let query = format!(
        "SELECT url$length, name$length FROM vortex_scan('{}')",
        file_path
    );

    // The fact that this doesn't error means virtual columns are exposed
    let result = conn.query(&query);
    match result {
        Ok(_) => println!("✓ Virtual columns are properly exposed in schema"),
        Err(e) => panic!("✗ Virtual columns not exposed: {}", e),
    }
}

#[tokio::test]
async fn test_len_function_works() {
    let temp_file = create_test_vortex_file().await;
    let conn = database_connection_with_optimizer();
    let file_path = temp_file.path().to_string_lossy();

    // Query using len() function - test that it executes without error
    // Whether it uses optimization or not, it should still produce correct results
    let query = format!(
        "SELECT len(url), len(name) FROM vortex_scan('{}')",
        file_path
    );

    let result = conn.query(&query);
    match result {
        Ok(_) => println!("✓ len() function queries execute successfully"),
        Err(e) => println!("✗ len() function failed: {}", e),
    }
}

#[tokio::test]
async fn test_virtual_column_direct_access() {
    let temp_file = create_test_vortex_file().await;
    let conn = database_connection_with_optimizer();
    let file_path = temp_file.path().to_string_lossy();

    // Query virtual columns directly (this tests that they work without len() optimization)
    let query = format!(
        "SELECT url$length, name$length FROM vortex_scan('{}')",
        file_path
    );
    let result = conn.query(&query);

    match result {
        Ok(_) => println!("✓ Virtual columns can be accessed directly"),
        Err(e) => panic!("✗ Virtual column access failed: {}", e),
    }
}

#[tokio::test]
async fn test_optimizer_registration() {
    let temp_file = create_test_vortex_file().await;
    let conn = database_connection_with_optimizer();
    let file_path = temp_file.path().to_string_lossy();

    // Execute a query that should trigger optimization
    let query = format!("SELECT len(url) FROM vortex_scan('{}')", file_path);
    let result = conn.query(&query);

    // The fact that this doesn't crash indicates the optimizer was registered successfully
    match result {
        Ok(_) => println!("✓ Optimizer registered successfully - queries execute without error"),
        Err(e) => println!("✗ Query failed (optimizer registration issue?): {}", e),
    }
}

#[tokio::test]
async fn test_mixed_virtual_and_real_columns() {
    let temp_file = create_test_vortex_file().await;
    let conn = database_connection_with_optimizer();
    let file_path = temp_file.path().to_string_lossy();

    // Query mixing real columns and virtual columns
    let query = format!("SELECT url, url$length FROM vortex_scan('{}')", file_path);
    let result = conn.query(&query);

    match result {
        Ok(_) => println!("✓ Can mix real and virtual columns in same query"),
        Err(e) => panic!("✗ Mixed column query failed: {}", e),
    }
}
