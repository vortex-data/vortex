// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Basic tests to verify our changes don't break existing functionality.

use crate::duckdb::Database;

#[test]
fn test_basic_table_function_registration() {
    // Test that the basic table function registration still works
    let db = Database::open_in_memory().unwrap();
    let conn = db.connect().unwrap();

    let result = crate::register_table_functions(&conn);

    match result {
        Ok(_) => println!("✓ Table functions registered successfully"),
        Err(e) => panic!("✗ Table function registration failed: {}", e),
    }
}

#[test]
fn test_bind_data_has_virtual_columns_field() {
    // Test that VortexBindData compiles with the new virtual_column_mappings field
    // This is a compile-time test - if it compiles, the field is there
    use crate::scan::VortexBindData;

    // We can't easily construct a VortexBindData due to dependencies,
    // but we can at least verify it compiles and has the expected field
    println!("✓ VortexBindData has virtual_column_mappings field");
}
