// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmark output utilities for writing results to the target directory.
//!
//! This module provides a standard location for benchmark results, similar to
//! how criterion stores its data in `target/criterion/`.
//!
//! Results are stored in `target/vortex-bench/<benchmark-id>/results.json`.

use std::fs::File;
use std::fs::{self};
use std::io::Write;
use std::io::{self};
use std::path::PathBuf;

use crate::display::DisplayFormat;
use crate::workspace_root;

/// The default output filename for benchmark results.
const DEFAULT_FILENAME: &str = "results.json";

/// Helper for managing benchmark output paths and files.
///
/// By default, outputs to `target/vortex-bench/<benchmark-id>/results.json`.
/// Can be overridden with a custom path.
pub struct BenchmarkOutput {
    path: PathBuf,
}

impl BenchmarkOutput {
    /// Create a new benchmark output with the default path.
    ///
    /// Results will be written to `target/vortex-bench/<benchmark_id>/results.json`.
    pub fn new(benchmark_id: &str) -> Self {
        let path = default_output_path(benchmark_id);
        Self { path }
    }

    /// Create a benchmark output with a custom path, or use the default if None.
    pub fn with_path(benchmark_id: &str, custom_path: Option<PathBuf>) -> Self {
        match custom_path {
            Some(path) => Self { path },
            None => Self::new(benchmark_id),
        }
    }

    /// Get the output path.
    pub fn path(&self) -> &PathBuf {
        &self.path
    }

    /// Create a writer for the output file.
    ///
    /// This will create parent directories if they don't exist.
    pub fn create_writer(&self) -> io::Result<Box<dyn Write>> {
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent)?;
        }
        let file = File::create(&self.path)?;
        Ok(Box::new(file))
    }

    /// Create a writer, or return stdout if creation fails.
    ///
    /// Useful for CLI tools that want to fall back to stdout.
    pub fn create_writer_or_stdout(&self) -> Box<dyn Write> {
        match self.create_writer() {
            Ok(writer) => writer,
            Err(e) => {
                eprintln!(
                    "Warning: could not create output file {}: {e}. Writing to stdout.",
                    self.path.display()
                );
                Box::new(io::stdout().lock())
            }
        }
    }
}

/// Get the default output path for a benchmark.
///
/// Returns `target/vortex-bench/<benchmark_id>/results.json`.
pub fn default_output_path(benchmark_id: &str) -> PathBuf {
    vortex_bench_dir().join(benchmark_id).join(DEFAULT_FILENAME)
}

/// Create an appropriate writer based on display format and output path.
///
/// - If an explicit output path is provided, always write to that file.
/// - Otherwise, write to stdout (for both Table and GhJson formats).
pub fn create_output_writer(
    display_format: &DisplayFormat,
    output_path: Option<PathBuf>,
    _benchmark_id: &str,
) -> io::Result<Box<dyn Write>> {
    match (display_format, output_path) {
        (_, Some(path)) => {
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)?;
            }

            Ok(Box::new(File::create(path)?))
        }
        (_, None) => Ok(Box::new(io::stdout())),
    }
}

/// Get the base directory for vortex benchmark results.
///
/// Returns `target/vortex-bench/`.
pub fn vortex_bench_dir() -> PathBuf {
    workspace_root().join("target").join("vortex-bench")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_output_path() {
        let path = default_output_path("compress");

        assert_eq!(
            path,
            workspace_root().join("target/vortex-bench/compress/results.json")
        );
    }

    #[test]
    fn test_benchmark_output_new() {
        let output = BenchmarkOutput::new("random-access");
        assert_eq!(
            output.path(),
            &workspace_root().join("target/vortex-bench/random-access/results.json")
        );
    }

    #[test]
    fn test_benchmark_output_with_custom_path() {
        let custom = PathBuf::from("/tmp/my-results.json");
        let output = BenchmarkOutput::with_path("test", Some(custom.clone()));
        assert_eq!(output.path(), &custom);
    }

    #[test]
    fn test_benchmark_output_with_none_uses_default() {
        let output = BenchmarkOutput::with_path("test", None);
        assert_eq!(
            output.path(),
            &workspace_root().join("target/vortex-bench/test/results.json")
        );
    }
}
