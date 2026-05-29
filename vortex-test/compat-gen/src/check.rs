// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::path::Path;
use std::time::Instant;

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

    let generation_start = Instant::now();
    eprintln!(
        "generating {} fresh fixtures for comparison in {}...",
        fixtures.len(),
        tmp_dir.path().display()
    );
    for (idx, fixture) in fixtures.iter().enumerate() {
        let fixture_start = Instant::now();
        eprintln!(
            "  generating {}/{} {}...",
            idx + 1,
            fixtures.len(),
            fixture.name()
        );
        let entries = fixture.write(tmp_dir.path())?;
        let written = entries
            .iter()
            .map(|entry| entry.name.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        eprintln!(
            "  generated {}/{} {} in {:.3}s: {}",
            idx + 1,
            fixtures.len(),
            fixture.name(),
            fixture_start.elapsed().as_secs_f64(),
            written
        );
    }
    eprintln!(
        "generated fresh fixtures in {:.3}s",
        generation_start.elapsed().as_secs_f64()
    );

    // Collect .vortex files in the check directory.
    let mut dir_files: Vec<String> = std::fs::read_dir(dir)
        .map_err(|e| vortex_err!("failed to read dir {}: {e}", dir.display()))?
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let name = entry.file_name().to_string_lossy().to_string();
            name.ends_with(".vortex").then_some(name)
        })
        .collect();
    dir_files.sort();

    // Collect all fixture names (each fixture may produce multiple files).
    let mut fresh_files: Vec<String> = std::fs::read_dir(tmp_dir.path())
        .map_err(|e| vortex_err!("failed to read tmp dir: {e}"))?
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let name = entry.file_name().to_string_lossy().to_string();
            name.ends_with(".vortex").then_some(name)
        })
        .collect();
    fresh_files.sort();

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

        let check_start = Instant::now();
        eprintln!("  checking {fresh_name}...");

        // Read the stored file.
        let read_stored_start = Instant::now();
        eprintln!("    reading stored file...");
        let stored_bytes = match std::fs::read(&stored_path) {
            Ok(b) => ByteBuffer::from(b),
            Err(e) => {
                result.failed.push(FailedFixture {
                    name: fresh_name.clone(),
                    error: format!("failed to read stored file: {e}"),
                });
                continue;
            }
        };
        eprintln!(
            "    read stored file in {:.3}s ({} bytes)",
            read_stored_start.elapsed().as_secs_f64(),
            stored_bytes.len()
        );

        // Read the fresh file.
        let fresh_path = tmp_dir.path().join(fresh_name);
        let read_fresh_start = Instant::now();
        eprintln!("    reading fresh file...");
        let fresh_bytes = match std::fs::read(&fresh_path) {
            Ok(b) => ByteBuffer::from(b),
            Err(e) => {
                result.failed.push(FailedFixture {
                    name: fresh_name.clone(),
                    error: format!("failed to read fresh file: {e}"),
                });
                continue;
            }
        };
        eprintln!(
            "    read fresh file in {:.3}s ({} bytes)",
            read_fresh_start.elapsed().as_secs_f64(),
            fresh_bytes.len()
        );

        // Validate the full layout tree of the stored file (reads every segment
        // including zone maps, dictionaries, etc.).
        let layout_start = Instant::now();
        eprintln!("    validating stored layout tree...");
        if let Err(e) = adapter::read_layout_tree(stored_bytes.clone()) {
            result.failed.push(FailedFixture {
                name: fresh_name.clone(),
                error: format!("stored file layout tree invalid: {e}"),
            });
            continue;
        }
        eprintln!(
            "    validated stored layout tree in {:.3}s",
            layout_start.elapsed().as_secs_f64()
        );

        // Scan data arrays from both files and compare.
        let decode_stored_start = Instant::now();
        eprintln!("    decoding stored file...");
        let stored_array = match adapter::read_file(stored_bytes) {
            Ok(a) => a,
            Err(e) => {
                result.failed.push(FailedFixture {
                    name: fresh_name.clone(),
                    error: format!("failed to decode stored vortex file: {e}"),
                });
                continue;
            }
        };
        eprintln!(
            "    decoded stored file in {:.3}s",
            decode_stored_start.elapsed().as_secs_f64()
        );

        let decode_fresh_start = Instant::now();
        eprintln!("    decoding fresh file...");
        let fresh_array = match adapter::read_file(fresh_bytes) {
            Ok(a) => a,
            Err(e) => {
                result.failed.push(FailedFixture {
                    name: fresh_name.clone(),
                    error: format!("failed to decode fresh vortex file: {e}"),
                });
                continue;
            }
        };
        eprintln!(
            "    decoded fresh file in {:.3}s",
            decode_fresh_start.elapsed().as_secs_f64()
        );

        let compare_start = Instant::now();
        eprintln!("    comparing arrays...");
        assert_arrays_eq!(stored_array, fresh_array);
        eprintln!(
            "    compared arrays in {:.3}s",
            compare_start.elapsed().as_secs_f64()
        );
        eprintln!(
            "  pass {fresh_name} in {:.3}s",
            check_start.elapsed().as_secs_f64()
        );
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
