// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Tests for optimizer registration and basic functionality.

use crate::duckdb::Database;

#[test]
fn test_optimizer_registration_does_not_crash() {
    // Test that registering the optimizer extension doesn't crash
    let db = Database::open_in_memory().unwrap();

    // This should succeed without crashing
    let result = crate::register_extension(&db);

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
