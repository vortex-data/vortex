use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::Result;
use xshell::Shell;

use crate::IdempotentPath;

pub struct DuckdbTpchOptions {
    /// Scale factor of the data in GB.
    pub scale_factor: u8,

    /// Location on-disk to store generated files.
    pub base_dir: PathBuf,
}

impl DuckdbTpchOptions {
    pub fn csvs_dir(&self) -> PathBuf {
        self.base_dir.join(self.scale_factor.to_string())
    }
}

impl Default for DuckdbTpchOptions {
    fn default() -> Self {
        Self {
            scale_factor: 1,
            base_dir: "tpch-duckdb".to_data_path(),
        }
    }
}

impl DuckdbTpchOptions {
    pub fn with_base_dir<P: AsRef<Path>>(self, dir: P) -> Self {
        Self {
            base_dir: dir.as_ref().to_owned(),
            scale_factor: self.scale_factor,
        }
    }

    pub fn with_scale_factor(self, scale_factor: u8) -> Self {
        Self {
            base_dir: self.base_dir,
            scale_factor,
        }
    }
}

pub fn generate_tpch(opts: DuckdbTpchOptions) -> Result<PathBuf> {
    let sh = Shell::new()?;

    let scale_factor = opts.scale_factor;

    // mkdir -p the output directory
    let output_dir = opts.csvs_dir();
    sh.create_dir(&output_dir)?;

    // See if the success file has been written. If so, do not run expensive generator
    // process again.
    let success_file = output_dir.join(".success");
    if sh.path_exists(&success_file) {
        return Ok(output_dir);
    }

    let output = Command::new("duckdb")
        .current_dir(&output_dir)
        .arg("-c")
        .arg(format!("install tpch; load tpch; call dbgen(sf={scale_factor}); export database '.' (format csv, delimiter '|', header false);"))
        .output()?;

    if !output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("duckdb failed: stdout=\"{stdout}\", stderr=\"{stderr}\"");
    }

    // rename .csv files into the expected .tbl extension
    sh.read_dir(&output_dir)?
        .into_iter()
        .filter(|p| p.extension().is_some_and(|ext| ext == "csv"))
        .try_for_each(|p| fs::rename(p.clone(), p.with_extension("tbl")))?;

    // Write a success file to indicate this scale-factor is created.
    sh.write_file(success_file, vec![])?;

    Ok(output_dir)
}
