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
/// Prints JSON result to stdout, human-readable progress to stderr.
/// Returns error if any fixture failed or if mode constraints are violated.
pub fn check(dir: &Path, mode: Mode, exclude: &[String]) -> VortexResult<()> {
    let fixtures: Vec<_> = all_fixtures()
        .into_iter()
        .filter(|f| {
            let name = f.name();
            !exclude.iter().any(|pat| name.contains(pat.as_str()))
        })
        .collect();

    if !exclude.is_empty() {
        eprintln!("excluding: {}", exclude.join(", "));
    }
    let tmp_dir = dir.join(".tmp");
    std::fs::create_dir_all(&tmp_dir).map_err(|e| vortex_err!("failed to create tmp dir: {e}"))?;

    // Collect .vortex files in the directory.
    let dir_files: Vec<String> = std::fs::read_dir(dir)
        .map_err(|e| vortex_err!("failed to read dir {}: {e}", dir.display()))?
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let name = entry.file_name().to_string_lossy().to_string();
            name.ends_with(".vortex").then_some(name)
        })
        .collect();

    let fixture_names: Vec<&str> = fixtures.iter().map(|f| f.name()).collect();

    let mut result = CheckResult {
        passed: Vec::new(),
        failed: Vec::new(),
        skipped: Vec::new(),
    };

    // Check for unknown files in the directory.
    for file_name in &dir_files {
        if !fixture_names.contains(&file_name.as_str()) {
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

    // Check each known fixture.
    for fixture in &fixtures {
        let file_path = dir.join(fixture.name());
        if !file_path.exists() {
            match mode {
                Mode::Exact | Mode::Subset => {
                    result.failed.push(FailedFixture {
                        name: fixture.name().to_string(),
                        error: "file missing from directory".to_string(),
                    });
                }
                Mode::Superset => {
                    eprintln!("  skip {} (missing)", fixture.name());
                    result.skipped.push(fixture.name().to_string());
                }
            }
            continue;
        }

        eprintln!("  checking {}...", fixture.name());

        // Setup + build expected arrays.
        if let Err(e) = fixture.setup(&tmp_dir) {
            result.failed.push(FailedFixture {
                name: fixture.name().to_string(),
                error: format!("setup failed: {e}"),
            });
            continue;
        }
        let expected = match fixture.build(&tmp_dir) {
            Ok(arr) => arr,
            Err(e) => {
                result.failed.push(FailedFixture {
                    name: fixture.name().to_string(),
                    error: format!("build failed: {e}"),
                });
                continue;
            }
        };

        // Read actual file.
        let file_bytes = match std::fs::read(&file_path) {
            Ok(b) => b,
            Err(e) => {
                result.failed.push(FailedFixture {
                    name: fixture.name().to_string(),
                    error: format!("failed to read file: {e}"),
                });
                continue;
            }
        };
        let actual = match adapter::read_file(ByteBuffer::from(file_bytes)) {
            Ok(a) => a,
            Err(e) => {
                result.failed.push(FailedFixture {
                    name: fixture.name().to_string(),
                    error: format!("failed to decode vortex file: {e}"),
                });
                continue;
            }
        };

        // Compare.
        assert_arrays_eq!(actual, expected);
        eprintln!("  pass {}", fixture.name());
        result.passed.push(fixture.name().to_string());
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
