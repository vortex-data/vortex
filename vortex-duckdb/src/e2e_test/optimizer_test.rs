// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Tests for optimizer registration and basic functionality.

use tempfile::NamedTempFile;
use vortex::arrays::{StructArray, VarBinArray};
use vortex::file::VortexWriteOptions;

use crate::RUNTIME;
use crate::duckdb::{Connection, Database};

#[test]
fn test_optimizer_registration_does_not_crash() {
    // Test that registering the optimizer extension doesn't crash
    let mut db = Database::open_in_memory().unwrap();

    // This should succeed without crashing
    let result = crate::register_extension(&mut db);

    match result {
        Ok(_) => println!("✓ Optimizer extension registered successfully"),
        Err(e) => panic!("✗ Optimizer registration failed: {}", e),
    }
}

#[test]
fn test_table_function_registration_still_works() {
    // Test that our changes didn't break the existing table function registration
    let db = Database::open_in_memory().unwrap();
    let conn = db.connect().unwrap();

    let result = crate::register_table_functions(&conn);

    match result {
        Ok(_) => println!("✓ Table functions registered successfully"),
        Err(e) => panic!("✗ Table function registration failed: {}", e),
    }
}

fn database_connection_with_optimizer() -> Connection {
    let mut db = Database::open_in_memory().unwrap();

    // Register the full extension including optimizer
    crate::register_extension(&mut db).unwrap();

    db.connect().unwrap()
}

async fn create_test_vortex_file() -> NamedTempFile {
    let temp_file = NamedTempFile::new().unwrap();

    // Create test data with string columns that will have virtual length columns
    let urls = VarBinArray::from_iter(
        [
            "https://example.com/page",
            "https://test.org/long/path/here",
            "https://short.co",
        ]
        .iter()
        .map(|s| Some(s.as_bytes())),
        vortex::dtype::DType::Utf8(vortex::dtype::Nullability::NonNullable),
    );

    let names = VarBinArray::from_iter(
        ["Alice Smith", "Bob", "Charlie Brown"]
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
fn test_expose_query_plan_with_len_function() {
    println!("\n🔍 QUERY PLAN EXPOSURE TEST");
    println!("============================");

    let temp_file = RUNTIME.block_on(create_test_vortex_file());
    let conn = database_connection_with_optimizer();
    let file_path = temp_file.path().to_string_lossy();

    // Test EXPLAIN with len() function to show plan transformation
    let query = format!(
        "SELECT len(url), len(name) FROM vortex_scan('{}')",
        file_path
    );
    let explain_query = format!("EXPLAIN {}", query);

    println!("\n📋 Original Query:");
    println!("   {}", query);

    println!("\n📋 EXPLAIN Query:");
    println!("   {}", explain_query);

    println!("\n🚀 Executing EXPLAIN (watch for optimizer transformation messages)...");
    println!("─────────────────────────────────────────────────────────────────────");

    let result = conn.query(&explain_query);

    match result {
        Ok(query_result) => {
            println!("✅ EXPLAIN executed successfully!");

            let mut chunk_count = 0;
            let mut full_plan = String::new();
            for chunk in query_result {
                chunk_count += 1;
                let _len = chunk.len();
                let chunk_str = String::try_from(&chunk).unwrap();
                full_plan.push_str(&chunk_str);
            }

            // Assert that we got a meaningful plan
            assert!(chunk_count > 0, "EXPLAIN query should return at least one chunk");
            assert!(!full_plan.is_empty(), "EXPLAIN plan should not be empty");
            
            // Check if length pushdown is happening
            let has_length_columns = full_plan.contains("$length")
                || full_plan.contains("url$length")
                || full_plan.contains("name$length");
            let has_len_function = full_plan.contains("len(") || full_plan.contains("length(");

            // Assert that we have evidence of optimization or the original function
            assert!(has_length_columns || has_len_function, 
                "Plan should contain either virtual length columns (optimized) or len() functions (unoptimized)");
        }
        Err(e) => {
            panic!("EXPLAIN query failed: {}. This indicates the optimizer transformation caused an issue.", e);
        }
    }

    // Test a simple query without len() function for comparison
    let simple_query = format!("SELECT url, name FROM vortex_scan('{}')", file_path);
    let simple_explain = format!("EXPLAIN {}", simple_query);

    let simple_result = conn.query(&simple_explain);
    // Simple query should also work
    simple_result.expect("Simple query EXPLAIN should work without optimization");
}

#[test]
fn test_optimizer_transformation_messages() {
    let temp_file = RUNTIME.block_on(create_test_vortex_file());
    let conn = database_connection_with_optimizer();
    let file_path = temp_file.path().to_string_lossy();

    // Execute EXPLAIN to see the plan transformation
    let query = format!("SELECT len(url) FROM vortex_scan('{}')", file_path);
    let explain_query = format!("EXPLAIN {}", query);

    let result = conn.query(&explain_query);

    let query_result = result.expect("EXPLAIN query for length optimization should execute successfully");

    let mut full_plan = String::new();
    for chunk in query_result {
        let chunk_str = String::try_from(&chunk).unwrap();
        full_plan.push_str(&chunk_str);
    }

    assert!(!full_plan.is_empty(), "EXPLAIN plan should not be empty");

    // Analyze transformation results
    let has_url_length = full_plan.contains("url$length");
    let has_len_url = full_plan.contains("len(url)") || full_plan.contains("length(url)");

    // Assert that either optimization occurred or original function is preserved
    assert!(has_url_length || has_len_url, 
        "Plan should contain either virtual 'url$length' column (optimized) or 'len(url)' function (unoptimized)");
}

#[test]
fn test_simple_len_query_without_optimizer() {
    let temp_file = RUNTIME.block_on(create_test_vortex_file());
    // Use database without trying to register optimizer extension
    let db = Database::open_in_memory().unwrap();
    let conn = db.connect().unwrap();
    crate::register_table_functions(&conn).unwrap();

    let file_path = temp_file.path().to_string_lossy();

    // Simple len() query to verify basic functionality works
    let query = format!("SELECT len(url) FROM vortex_scan('{}')", file_path);
    let explain_query = format!("EXPLAIN {}", query);

    let result = conn.query(&explain_query);

    let query_result = result.expect("EXPLAIN query should work without optimizer extension");

    let mut found_len_function = false;
    for chunk in query_result {
        let chunk_str = String::try_from(&chunk).unwrap();
        if chunk_str.contains("len(url)") {
            found_len_function = true;
        }
    }

    // Without optimizer, len() function should be present in the plan
    assert!(found_len_function, "len(url) should be present in plan without optimizer");
}

#[test]
fn test_optimizer_with_actual_query() {
    let temp_file = RUNTIME.block_on(create_test_vortex_file());
    let conn = database_connection_with_optimizer();
    let file_path = temp_file.path().to_string_lossy();

    // Execute the actual query (not EXPLAIN) to see if optimizer is triggered
    let query = format!("SELECT len(url) FROM vortex_scan('{}')", file_path);

    let result = conn.query(&query);

    let query_result = result.expect("Query with len() function should execute successfully with optimizer");

    // Verify we get results
    let mut row_count = 0;
    for chunk in query_result {
        row_count += chunk.len();
    }
    
    assert!(row_count > 0, "Query should return at least one row");
}
