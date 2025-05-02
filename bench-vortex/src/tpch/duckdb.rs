use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use anyhow::Result;
use xshell::Shell;

use crate::Format;
use crate::ddb::DuckDBExecutor;

pub enum TpcDataset {
    TpcH,
    TpcDs,
}

pub struct DuckdbTpcOptions {
    /// Scale factor of the data in GB.
    pub scale_factor: u8,

    /// Location on-disk to store generated files.
    pub base_dir: PathBuf,

    pub dataset: TpcDataset,

    pub format: Format,

    pub duckdb_path: Option<PathBuf>,
}

impl DuckdbTpcOptions {
    // TODO(joe); this is not unused fix this in tpch
    pub fn csvs_dir(&self) -> PathBuf {
        self.base_dir.join(self.scale_factor.to_string())
    }

    pub fn output_dir(&self) -> PathBuf {
        self.base_dir
            .join(self.scale_factor.to_string())
            .join(self.format.to_string())
    }
}

impl DuckdbTpcOptions {
    pub fn new(base_dir: PathBuf, dataset: TpcDataset, format: Format) -> Self {
        Self {
            scale_factor: 1,
            base_dir,
            dataset,
            format,
            duckdb_path: None,
        }
    }
}

impl DuckdbTpcOptions {
    pub fn with_base_dir<P: AsRef<Path>>(mut self, dir: P) -> Self {
        self.base_dir = dir.as_ref().to_path_buf();
        self
    }

    pub fn with_scale_factor(mut self, scale_factor: u8) -> Self {
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
    pub fn with_duckdb_path(mut self, path: PathBuf) -> Self {
        self.duckdb_path = Some(path);
        self
    }
}

pub fn generate_tpc(opts: DuckdbTpcOptions) -> Result<PathBuf> {
    let sh = Shell::new()?;

    let scale_factor = opts.scale_factor;

    // mkdir -p the output directory
    let output_dir = opts.output_dir();
    sh.create_dir(&output_dir)?;

    // See if the success file has been written. If so, do not run expensive generator
    // process again.
    let success_file = output_dir.join(".success");
    if sh.path_exists(&success_file) {
        return Ok(output_dir);
    }

    let is_local_duckdb = opts.duckdb_path.is_some();

    let mut command = Command::new(opts.duckdb_path.unwrap_or_else(|| PathBuf::from("duckdb")));

    match opts.dataset {
        TpcDataset::TpcH => command
            .arg("-c")
            .arg("load tpch;")
            .arg("-c")
            .arg(format!("call dbgen(sf={scale_factor})")),
        TpcDataset::TpcDs => command
            .arg("-c")
            .arg("load tpcds;")
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
        Format::OnDiskVortex | Format::InMemoryVortex => {
            if !is_local_duckdb {
                command
                    .arg("-c")
                    .arg("install vortex from community; load vortex;");
            }

            command.arg("-c").arg(format!(
                "EXPORT DATABASE '{}' (format VORTEX);",
                output_dir.to_string_lossy(),
            ));
        }
        Format::OnDiskDuckDB | Format::Arrow => { /* Do nothing */ }
    };

    let output = command.output()?;

    if !output.status.success() || !output.stderr.is_empty() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("duckdb failed: stdout=\"{stdout}\", stderr=\"{stderr}\"");
    }

    // Write a success file to indicate this scale-factor is created.
    sh.write_file(success_file, vec![])?;

    Ok(output_dir)
}

/// Convenience wrapper for TPC-H benchmarks
pub fn execute_duckdb_tpch_query(
    queries: &[String],
    duckdb_executor: &DuckDBExecutor,
) -> Result<Duration> {
    crate::engines::ddb::execute_query(queries, duckdb_executor)
}
