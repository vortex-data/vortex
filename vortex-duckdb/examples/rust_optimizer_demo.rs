//! Demo of the Rust-based length optimizer
//!
//! This example shows how to use the new pure Rust length optimizer
//! instead of the legacy C++ implementation.

use vortex_duckdb::{Database, optimizer};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Vortex DuckDB Rust Optimizer Demo");

    // Create a DuckDB database
    let mut db = Database::open_in_memory()?;

    // Register the new Rust-based optimizer (recommended)
    println!("📝 Registering Rust-based length optimizer...");
    optimizer::register_rust_optimizer(&mut db)?;
    println!("✅ Rust optimizer registered successfully!");

    // Alternative: Register the legacy C++ optimizer
    // optimizer::register_optimizer(&mut db)?;

    // Create a connection and register table functions
    let conn = db.connect()?;
    vortex_duckdb::register_table_functions(&conn)?;

    println!("🎯 Now when you run queries with len() functions on vortex_scan tables,");
    println!("   the Rust optimizer will automatically rewrite them to use virtual columns!");

    println!("\n📖 Example query that would be optimized:");
    println!("   SELECT title, len(title), description, page_count");
    println!("   FROM vortex_scan('path/to/your/file.vortex')");

    println!("\n🔄 The optimizer will rewrite len(title) to use title$length virtual column");
    println!("   which is much more efficient than computing the length at query time.");

    Ok(())
}
