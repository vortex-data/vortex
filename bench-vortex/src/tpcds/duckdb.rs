// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::path::PathBuf;

use anyhow::Result;
use vortex_duckdb::duckdb::Database;

use crate::Format;

/// Generate TPC-DS data using a DuckDB connection
pub fn generate_tpcds(base_dir: PathBuf, scale_factor: String, format: Format) -> Result<PathBuf> {
    // Create output directory based on format
    let output_dir = base_dir.join(format.to_string());
    std::fs::create_dir_all(&output_dir)?;

    // Check if already generated
    let success_file = output_dir.join(".success");
    if success_file.exists() {
        return Ok(output_dir);
    }

    // Create in-memory database
    let db = Database::open_in_memory()?;
    let conn = db.connect()?;

    // Register vortex extension if needed
    if matches!(format, Format::OnDiskVortex | Format::VortexCompact) {
        vortex_duckdb::register_table_functions(&conn)?;
    }

    // Install and load required extensions
    conn.query("SET autoinstall_known_extensions=1;")?;
    conn.query("SET autoload_known_extensions=1;")?;

    // Install TPC-DS extension
    conn.query("INSTALL tpcds")?;
    conn.query("LOAD tpcds")?;

    // Generate TPC-DS data
    let query = format!("CALL dsdgen(sf={scale_factor})");
    conn.query(&query)?;

    // Export to the desired format
    match format {
        Format::Csv => {
            let query = format!(
                "EXPORT DATABASE '{}' (FORMAT CSV, delimiter '|', header false, FILE_EXTENSION tbl);",
                output_dir.to_string_lossy()
            );
            conn.query(&query)?;
        }
        Format::Parquet => {
            let query = format!(
                "EXPORT DATABASE '{}' (FORMAT PARQUET);",
                output_dir.to_string_lossy()
            );
            conn.query(&query)?;
        }
        Format::OnDiskVortex | Format::VortexCompact => {
            let query = format!(
                "EXPORT DATABASE '{}' (FORMAT VORTEX);",
                output_dir.to_string_lossy()
            );
            conn.query(&query)?;
        }
        Format::OnDiskDuckDB | Format::Arrow => {
            // These formats don't need export
        }
        Format::Lance => unimplemented!(),
    }

    // Write success marker
    std::fs::write(success_file, vec![])?;

    Ok(output_dir)
}
