// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::Result;
use vortex_duckdb::duckdb::Database;
use xshell::Shell;

use crate::Format;

pub enum TpcDataset {
    TpcH,
    TpcDs,
}

pub struct DuckdbTpcOptions {
    /// Scale factor of the data in GB.
    pub scale_factor: String,

    /// Location on-disk to store generated files.
    pub base_dir: PathBuf,

    pub dataset: TpcDataset,

    pub format: Format,
}

impl DuckdbTpcOptions {
    // TODO(joe); this is not unused fix this in tpch
    pub fn csvs_dir(&self) -> PathBuf {
        self.base_dir.join(&self.scale_factor)
    }

    pub fn output_dir(&self) -> PathBuf {
        self.base_dir.join(self.format.to_string())
    }
}

impl DuckdbTpcOptions {
    pub fn new(base_dir: PathBuf, dataset: TpcDataset, format: Format) -> Self {
        Self {
            scale_factor: "1".to_string(),
            base_dir,
            dataset,
            format,
        }
    }
}

impl DuckdbTpcOptions {
    pub fn with_base_dir<P: AsRef<Path>>(mut self, dir: P) -> Self {
        self.base_dir = dir.as_ref().to_path_buf();
        self
    }

    pub fn with_scale_factor(mut self, scale_factor: String) -> Self {
        self.scale_factor = scale_factor;
        self
    }

    pub fn with_format(mut self, format: Format) -> Self {
        self.format = format;
        self
    }

    pub fn with_dataset(mut self, dataset: TpcDataset) -> Self {
        self.dataset = dataset;
        self
    }
}

pub fn generate_tpc(opts: DuckdbTpcOptions) -> Result<PathBuf> {
    let sh = Shell::new()?;

    let scale_factor = &opts.scale_factor;

    // mkdir -p the output directory
    let output_dir = opts.output_dir();
    sh.create_dir(&output_dir)?;

    // See if the success file has been written. If so, do not run expensive generator
    // process again.
    let success_file = output_dir.join(".success");
    if sh.path_exists(&success_file) {
        return Ok(output_dir);
    }

    let mut command = Command::new("duckdb");
    command
        .arg("-c")
        .arg("install vortex from community; load vortex;");

    command
        .arg("-c")
        .arg("SET autoinstall_known_extensions=1;")
        .arg("-c")
        .arg("SET autoload_known_extensions=1;");

    match opts.dataset {
        TpcDataset::TpcH => command
            .arg("-c")
            .arg(format!("call dbgen(sf={scale_factor})")),
        TpcDataset::TpcDs => command
            .arg("-c")
            .arg(format!("call dsdgen(sf={scale_factor})")),
    };

    match opts.format {
        Format::Csv => {
            command.arg("-c").arg(format!(
                "EXPORT DATABASE '{}' (FORMAT CSV, delimiter '|', header false, FILE_EXTENSION tbl);",
                output_dir.to_string_lossy(),
            ));
        }
        Format::Parquet => {
            command.arg("-c").arg(format!(
                "EXPORT DATABASE '{}' (format PARQUET);",
                output_dir.to_string_lossy(),
            ));
        }
        Format::OnDiskVortex => {
            command.arg("-c").arg(format!(
                "EXPORT DATABASE '{}' (format VORTEX);",
                output_dir.to_string_lossy(),
            ));
        }
        Format::VortexCompact => {
            command.arg("-c").arg(format!(
                "EXPORT DATABASE '{}' (format VORTEX);",
                output_dir.to_string_lossy(),
            ));
        }
        Format::OnDiskDuckDB | Format::Arrow => { /* Do nothing */ }
    };

    command.envs(std::env::vars_os());
    let output = command.output()?;

    if !output.status.success() || !output.stderr.is_empty() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("duckdb failed, generating tpc*: stdout=\"{stdout}\", stderr=\"{stderr}\"");
    }

    // Write a success file to indicate this scale-factor is created.
    sh.write_file(success_file, vec![])?;

    Ok(output_dir)
}

/// Generate TPC-DS data using a DuckDB connection
pub fn generate_tpcds_with_connection(
    base_dir: PathBuf,
    scale_factor: String,
    format: Format,
) -> Result<PathBuf> {
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
    }

    // Write success marker
    std::fs::write(success_file, vec![])?;

    Ok(output_dir)
}
