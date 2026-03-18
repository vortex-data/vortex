// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::path::Path;

use clap::ValueEnum;
use serde::Serialize;
use vortex_array::assert_arrays_eq;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;

use crate::adapter;
use crate::fixtures::all_fixtures;

/// How to handle mismatches between directory and known fixtures.
#[derive(Clone, ValueEnum)]
pub enum Mode {
    /// Directory must match fixtures exactly.
    Exact,
    /// Directory may have extra files (skip), but all known must be present.
    Subset,
    /// Directory may be missing files (skip), but no unknown files allowed.
    Superset,
}

#[derive(Serialize)]
struct CheckResult {
    passed: Vec<String>,
    failed: Vec<FailedFixture>,
    skipped: Vec<String>,
}

#[derive(Serialize)]
struct FailedFixture {
    name: String,
    error: String,
}

/// Check `.vortex` files in `dir` against in-memory fixtures.
///
/// For each known fixture, generates fresh files in a temp directory via
/// `fixture.write(tmp_dir)`, then reads both the stored file and fresh file,
/// decodes them, and compares the arrays.
///
/// Prints JSON result to stdout, human-readable progress to stderr.
/// Returns error if any fixture failed or if mode constraints are violated.
pub fn check(dir: &Path, mode: Mode, exclude: &[String]) -> VortexResult<()> {
    let fixtures = all_fixtures();
    let fixtures: Vec<_> = fixtures
        .into_iter()
        .filter(|f| {
            let name = f.name();
            !exclude.iter().any(|pat| name.contains(pat.as_str()))
        })
        .collect();

    if !exclude.is_empty() {
        eprintln!("excluding: {}", exclude.join(", "));
    }

    // Generate fresh fixtures into a temp directory.
    let tmp_dir = tempfile::tempdir().map_err(|e| vortex_err!("failed to create temp dir: {e}"))?;

    eprintln!("generating fresh fixtures for comparison...");
    for fixture in &fixtures {
        fixture.write(tmp_dir.path())?;
    }

    // Collect .vortex files in the check directory.
    let dir_files: Vec<String> = std::fs::read_dir(dir)
        .map_err(|e| vortex_err!("failed to read dir {}: {e}", dir.display()))?
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let name = entry.file_name().to_string_lossy().to_string();
            name.ends_with(".vortex").then_some(name)
        })
        .collect();

    // Collect all fixture names (each fixture may produce multiple files).
    let fresh_files: Vec<String> = std::fs::read_dir(tmp_dir.path())
        .map_err(|e| vortex_err!("failed to read tmp dir: {e}"))?
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let name = entry.file_name().to_string_lossy().to_string();
            name.ends_with(".vortex").then_some(name)
        })
        .collect();

    let mut result = CheckResult {
        passed: Vec::new(),
        failed: Vec::new(),
        skipped: Vec::new(),
    };

    // Check for unknown files in the directory.
    for file_name in &dir_files {
        if !fresh_files.contains(file_name) {
            match mode {
                Mode::Exact | Mode::Superset => {
                    result.failed.push(FailedFixture {
                        name: file_name.clone(),
                        error: "unknown fixture (not in current fixture set)".to_string(),
                    });
                }
                Mode::Subset => {
                    eprintln!("  skip {file_name} (unknown)");
                    result.skipped.push(file_name.clone());
                }
            }
        }
    }

    // Check each known fixture file.
    for fresh_name in &fresh_files {
        let stored_path = dir.join(fresh_name);
        if !stored_path.exists() {
            match mode {
                Mode::Exact | Mode::Subset => {
                    result.failed.push(FailedFixture {
                        name: fresh_name.clone(),
                        error: "file missing from directory".to_string(),
                    });
                }
                Mode::Superset => {
                    eprintln!("  skip {fresh_name} (missing)");
                    result.skipped.push(fresh_name.clone());
                }
            }
            continue;
        }

        eprintln!("  checking {fresh_name}...");

        // Read the stored file.
        let stored_bytes = match std::fs::read(&stored_path) {
            Ok(b) => b,
            Err(e) => {
                result.failed.push(FailedFixture {
                    name: fresh_name.clone(),
                    error: format!("failed to read stored file: {e}"),
                });
                continue;
            }
        };
        let stored_array = match adapter::read_file(ByteBuffer::from(stored_bytes)) {
            Ok(a) => a,
            Err(e) => {
                result.failed.push(FailedFixture {
                    name: fresh_name.clone(),
                    error: format!("failed to decode stored vortex file: {e}"),
                });
                continue;
            }
        };

        // Read the fresh file.
        let fresh_path = tmp_dir.path().join(fresh_name);
        let fresh_bytes = match std::fs::read(&fresh_path) {
            Ok(b) => b,
            Err(e) => {
                result.failed.push(FailedFixture {
                    name: fresh_name.clone(),
                    error: format!("failed to read fresh file: {e}"),
                });
                continue;
            }
        };
        let fresh_array = match adapter::read_file(ByteBuffer::from(fresh_bytes)) {
            Ok(a) => a,
            Err(e) => {
                result.failed.push(FailedFixture {
                    name: fresh_name.clone(),
                    error: format!("failed to decode fresh vortex file: {e}"),
                });
                continue;
            }
        };

        // Compare arrays.
        assert_arrays_eq!(stored_array, fresh_array);
        eprintln!("  pass {fresh_name}");
        result.passed.push(fresh_name.clone());
    }

    // Print JSON result to stdout.
    let json = serde_json::to_string_pretty(&result)
        .map_err(|e| vortex_err!("failed to serialize result: {e}"))?;
    println!("{json}");

    // Summary to stderr.
    eprintln!(
        "\nresult: {} passed, {} failed, {} skipped",
        result.passed.len(),
        result.failed.len(),
        result.skipped.len()
    );

    if !result.failed.is_empty() {
        for f in &result.failed {
            eprintln!("  FAIL {}: {}", f.name, f.error);
        }
        vortex_bail!("{} fixture(s) failed", result.failed.len());
    }

    Ok(())
}
