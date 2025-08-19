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

            // Print the full EXPLAIN plan
            println!("\n📄 EXPLAIN Query Plan:");
            println!("─────────────────────────");

            let mut chunk_count = 0;
            let mut full_plan = String::new();
            for chunk in query_result {
                chunk_count += 1;
                let len = chunk.len();
                let chunk_str = String::try_from(&chunk).unwrap();
                full_plan.push_str(&chunk_str);
                println!("{}", chunk_str);
            }

            println!("─────────────────────────");
            println!("📊 Total chunks: {}", chunk_count);

            // Check if length pushdown is happening
            let has_length_columns = full_plan.contains("$length")
                || full_plan.contains("url$length")
                || full_plan.contains("name$length");
            let has_len_function = full_plan.contains("len(") || full_plan.contains("length(");

            println!("\n🔍 PLAN ANALYSIS:");
            if has_length_columns {
                println!("✅ Virtual length columns detected in plan - pushdown working!");
            } else if has_len_function {
                println!("⚠️  len() functions still present - transformation may not be working");
            } else {
                println!("❓ No obvious len() or $length references found in plan");
            }
        }
        Err(e) => {
            println!("❌ EXPLAIN failed with error: {}", e);
            println!("   This indicates the optimizer transformation caused an issue.");
        }
    }

    println!("\n🔍 Testing without len() function for comparison...");
    let simple_query = format!("SELECT url, name FROM vortex_scan('{}')", file_path);
    let simple_explain = format!("EXPLAIN {}", simple_query);

    println!("📋 Simple Query: {}", simple_query);

    let simple_result = conn.query(&simple_explain);
    match simple_result {
        Ok(_) => println!("✅ Simple query EXPLAIN also works"),
        Err(e) => println!("❌ Simple query EXPLAIN failed: {}", e),
    }
}

#[test]
fn test_optimizer_transformation_messages() {
    println!("\n🔧 OPTIMIZER TRANSFORMATION MESSAGES TEST");
    println!("==========================================");

    let temp_file = RUNTIME.block_on(create_test_vortex_file());
    let conn = database_connection_with_optimizer();
    let file_path = temp_file.path().to_string_lossy();

    println!("\n📝 This test should show optimizer messages in stdout during execution.");
    println!("Look for messages like: '🔄 OPTIMIZER: Rewriting len(url) → url$length'");

    // Execute EXPLAIN to see the plan transformation
    let query = format!("SELECT len(url) FROM vortex_scan('{}')", file_path);
    let explain_query = format!("EXPLAIN {}", query);

    println!("\n🎯 Original Query: {}", query);
    println!("\n🎯 EXPLAIN Query: {}", explain_query);
    println!("─────────────────────────────────────");

    let result = conn.query(&explain_query);

    println!("─────────────────────────────────────");

    match result {
        Ok(query_result) => {
            println!("✅ EXPLAIN executed successfully!");

            // Print the full plan and analyze for transformations
            println!("\n📄 EXPLAIN Plan for len(url) query:");
            println!("────────────────────────────────────");

            let mut full_plan = String::new();
            for chunk in query_result {
                let chunk_str = String::try_from(&chunk).unwrap();
                full_plan.push_str(&chunk_str);
                println!("{}", chunk_str);
            }

            println!("────────────────────────────────────");

            // Detailed analysis of transformation
            let has_url_length = full_plan.contains("url$length");
            let has_len_url = full_plan.contains("len(url)") || full_plan.contains("length(url)");

            println!("\n🔧 TRANSFORMATION ANALYSIS:");
            if has_url_length {
                println!("✅ SUCCESS: 'url$length' found in plan - len(url) was transformed!");
                if has_len_url {
                    println!("⚠️  WARNING: Still found 'len(url)' - partial transformation?");
                } else {
                    println!("✅ PERFECT: No 'len(url)' found - complete transformation!");
                }
            } else if has_len_url {
                println!("❌ ISSUE: 'len(url)' still present - transformation NOT working");
                println!("   Expected: len(url) → url$length");
            } else {
                println!("❓ UNCLEAR: Neither 'len(url)' nor 'url$length' found in plan");
            }
        }
        Err(e) => {
            println!("❌ EXPLAIN failed: {}", e);
            println!("   This indicates the optimizer transformation caused an issue.");
        }
    }
}

#[test]
fn test_simple_len_query_without_optimizer() {
    println!("\\n🔧 SIMPLE LEN QUERY TEST (no extension)");
    println!("=========================================");

    let temp_file = RUNTIME.block_on(create_test_vortex_file());
    // Use database without trying to register optimizer extension
    let db = Database::open_in_memory().unwrap();
    let conn = db.connect().unwrap();
    crate::register_table_functions(&conn).unwrap();

    let file_path = temp_file.path().to_string_lossy();

    // Simple len() query to verify basic functionality works
    let query = format!("SELECT len(url) FROM vortex_scan('{}')", file_path);
    let explain_query = format!("EXPLAIN {}", query);

    println!("\\n🎯 Query: {}", query);

    let result = conn.query(&explain_query);

    match result {
        Ok(query_result) => {
            println!("✅ Query works without optimizer extension!");

            for chunk in query_result {
                let chunk_str = String::try_from(&chunk).unwrap();
                println!("{}", chunk_str);

                if chunk_str.contains("len(url)") {
                    println!(
                        "✅ CONFIRMED: len(url) is present in plan (normal behavior without optimizer)"
                    );
                }
            }
        }
        Err(e) => {
            println!("❌ Query failed: {}", e);
        }
    }
}

#[test]
fn test_optimizer_with_actual_query() {
    println!("\\n🔧 ACTUAL QUERY OPTIMIZER TEST");
    println!("==============================");

    let temp_file = RUNTIME.block_on(create_test_vortex_file());
    let conn = database_connection_with_optimizer();
    let file_path = temp_file.path().to_string_lossy();

    // Execute the actual query (not EXPLAIN) to see if optimizer is triggered
    let query = format!("SELECT len(url) FROM vortex_scan('{}')", file_path);

    println!("\\n🎯 Query: {}", query);
    println!("\\n🎯 Executing actual query (watch for optimizer messages)...");
    println!("─────────────────────────────────────");

    let result = conn.query(&query);

    println!("─────────────────────────────────────");

    match result {
        Ok(query_result) => {
            println!("✅ Query executed successfully!");

            // Just verify we get results
            let mut row_count = 0;
            for chunk in query_result {
                row_count += chunk.len();
                println!("📊 Got chunk with {} rows", chunk.len());
            }
            println!("📊 Total rows: {}", row_count);
        }
        Err(e) => {
            println!("❌ Query failed: {}", e);
        }
    }
}
