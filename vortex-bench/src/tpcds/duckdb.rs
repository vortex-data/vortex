// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::path::PathBuf;
use std::process::Command;

use anyhow::Result;
use tracing::info;

use crate::Format;

pub fn generate_tpcds(base_dir: PathBuf, scale_factor: String) -> Result<PathBuf> {
    // Create output directory based on format
    let output_dir = base_dir.join(Format::Parquet.name());
    std::fs::create_dir_all(&output_dir)?;

    // Check if already generated
    let success_file = output_dir.join(".success");
    if success_file.exists() {
        return Ok(output_dir);
    }

    info!(
        "Generating TPC-DS data with scale factor {} @ {}",
        scale_factor,
        output_dir.display()
    );

    let sql_script = format!(
        "SET autoinstall_known_extensions=1;\
        SET autoload_known_extensions=1;\
        INSTALL tpcds;\
        LOAD tpcds;\
        CALL dsdgen(sf={scale_factor});\
        EXPORT DATABASE '{output_dir}' (FORMAT PARQUET);",
        scale_factor = scale_factor,
        output_dir = output_dir.to_string_lossy()
    );

    let result = Command::new("duckdb")
        .arg("-c")
        .arg(&sql_script)
        .spawn()?
        .wait()?;

    if !result.success() {
        anyhow::bail!("DuckDB CLI failed to generate TPC-DS data");
    }

    // Write success marker
    std::fs::write(success_file, vec![])?;

    Ok(output_dir)
}
