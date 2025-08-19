// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Debug tests to catch and analyze foreign exceptions

use std::panic;

use tempfile::NamedTempFile;
use vortex::arrays::{StructArray, VarBinArray};
use vortex::file::VortexWriteOptions;

use crate::RUNTIME;
use crate::duckdb::{Connection, Database};

fn database_connection_with_optimizer() -> Connection {
    let db = Database::open_in_memory().unwrap();

    // Register the full extension including optimizer
    crate::register_extension(&db).unwrap();

    db.connect().unwrap()
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

#[test]
fn test_debug_virtual_column_exception() {
    println!("\n🔍 DEBUG: Testing virtual column exception");

    let temp_file = RUNTIME.block_on(create_test_vortex_file());
    let conn = database_connection_with_optimizer();
    let file_path = temp_file.path().to_string_lossy();

    // Set up panic hook to catch details
    let original_hook = panic::take_hook();
    panic::set_hook(Box::new(|panic_info| {
        eprintln!("🚨 PANIC CAUGHT: {}", panic_info);
        if let Some(location) = panic_info.location() {
            eprintln!(
                "📍 Location: {}:{}:{}",
                location.file(),
                location.line(),
                location.column()
            );
        }
        if let Some(s) = panic_info.payload().downcast_ref::<&str>() {
            eprintln!("💬 Message: {}", s);
        } else if let Some(s) = panic_info.payload().downcast_ref::<String>() {
            eprintln!("💬 Message: {}", s);
        }
    }));

    // Try to query virtual columns with detailed error handling
    let query = format!(
        "SELECT url$length, name$length FROM vortex_scan('{}')",
        file_path
    );
    println!("🔍 Query: {}", query);

    let result = panic::catch_unwind(|| conn.query(&query));

    // Restore original panic hook
    panic::set_hook(original_hook);

    match result {
        Ok(query_result) => match query_result {
            Ok(_) => println!("✅ Query succeeded unexpectedly!"),
            Err(e) => println!("❌ Query failed with DuckDB error: {}", e),
        },
        Err(panic_payload) => {
            println!("🚨 Caught panic!");
            if let Some(s) = panic_payload.downcast_ref::<&str>() {
                println!("💬 Panic message: {}", s);
            } else if let Some(s) = panic_payload.downcast_ref::<String>() {
                println!("💬 Panic message: {}", s);
            } else {
                println!("💬 Unknown panic payload");
            }
        }
    }
}

#[tokio::test]
async fn test_debug_real_columns_only() {
    println!("\n🔍 DEBUG: Testing real columns only (should work)");

    let temp_file = create_test_vortex_file().await;
    let conn = database_connection_with_optimizer();
    let file_path = temp_file.path().to_string_lossy();

    // Query only real columns - this should work
    let query = format!("SELECT url, name FROM vortex_scan('{}')", file_path);
    println!("🔍 Query: {}", query);

    let result = conn.query(&query);
    match result {
        Ok(_) => println!("✅ Real columns query succeeded!"),
        Err(e) => println!("❌ Real columns query failed: {}", e),
    }
}
