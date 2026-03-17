// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::path::Path;
use std::thread;

use serde::Serialize;
use sha2::Digest;
use sha2::Sha256;
use vortex_array::ArrayRef;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;

use crate::adapter;
use crate::fixtures::ArrayFixture;
use crate::fixtures::all_fixtures;
use crate::fixtures::check_expected_encodings;

#[derive(Serialize)]
struct FixturesJson {
    fixtures: Vec<FixtureInfo>,
}

#[derive(Serialize)]
struct FixtureInfo {
    name: String,
    description: String,
    sha256: String,
}

/// Generate all fixtures into `output_dir`.
///
/// Three phases:
/// 1. **Setup** — run each fixture's `setup()` in parallel threads (I/O).
/// 2. **Build** — construct arrays in parallel threads (CPU).
/// 3. **Write** — serialize `.vortex` files and `fixtures.json` to disk.
///
/// All fixtures must build successfully before any are written.
pub fn generate(output_dir: &Path, exclude: &[String]) -> VortexResult<()> {
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

    let tmp_dir = output_dir.join(".tmp");
    std::fs::create_dir_all(&tmp_dir).map_err(|e| vortex_err!("failed to create tmp dir: {e}"))?;

    // Phase 1: Setup (parallel I/O).
    eprintln!("[1/3] Setting up {} fixtures...", fixtures.len());
    run_setup_parallel(&fixtures, &tmp_dir)?;

    // Phase 2: Build (parallel CPU).
    eprintln!("[2/3] Building {} fixtures...", fixtures.len());
    let built = run_build_parallel(&fixtures, &tmp_dir)?;

    // Phase 3: Write to disk.
    eprintln!("[3/3] Writing to {}...", output_dir.display());
    std::fs::create_dir_all(output_dir)
        .map_err(|e| vortex_err!("failed to create output dir: {e}"))?;

    let mut infos = Vec::with_capacity(built.len());
    for (fixture, array) in &built {
        check_expected_encodings(array, *fixture)?;
        let path = output_dir.join(fixture.name());
        adapter::write_file(&path, array.clone())?;
        let file_bytes = std::fs::read(&path)
            .map_err(|e| vortex_err!("failed to read back {}: {e}", path.display()))?;
        let sha256 = format!("{:x}", Sha256::digest(&file_bytes));
        infos.push(FixtureInfo {
            name: fixture.name().to_string(),
            description: fixture.description().to_string(),
            sha256,
        });
        eprintln!("  wrote {}", fixture.name());
    }

    let fixtures_json = FixturesJson { fixtures: infos };
    let json = serde_json::to_string_pretty(&fixtures_json)
        .map_err(|e| vortex_err!("failed to serialize fixtures.json: {e}"))?;
    std::fs::write(output_dir.join("fixtures.json"), format!("{json}\n"))
        .map_err(|e| vortex_err!("failed to write fixtures.json: {e}"))?;
    eprintln!("  wrote fixtures.json");

    eprintln!(
        "\ndone: {} fixtures in {}",
        fixtures_json.fixtures.len(),
        output_dir.display()
    );
    Ok(())
}

/// Run `setup()` for all fixtures in parallel via `std::thread::scope`.
fn run_setup_parallel(fixtures: &[Box<dyn ArrayFixture>], tmp_dir: &Path) -> VortexResult<()> {
    let errors: Vec<String> = thread::scope(|s| {
        let handles: Vec<_> = fixtures
            .iter()
            .map(|fixture| {
                let name = fixture.name().to_string();
                s.spawn(move || match fixture.setup(tmp_dir) {
                    Ok(()) => {
                        eprintln!("  setup {name}");
                        None
                    }
                    Err(e) => Some(format!("{name}: {e}")),
                })
            })
            .collect();

        handles
            .into_iter()
            .filter_map(|h| {
                h.join()
                    .unwrap_or_else(|_| Some("task panicked".to_string()))
            })
            .collect()
    });

    if !errors.is_empty() {
        vortex_bail!(
            "{} fixture(s) failed setup:\n  {}",
            errors.len(),
            errors.join("\n  ")
        );
    }
    Ok(())
}

/// Build all fixtures in parallel via `std::thread::scope`.
fn run_build_parallel<'a>(
    fixtures: &'a [Box<dyn ArrayFixture>],
    tmp_dir: &Path,
) -> VortexResult<Vec<(&'a dyn ArrayFixture, ArrayRef)>> {
    let build_results: Vec<VortexResult<(&dyn ArrayFixture, ArrayRef)>> = thread::scope(|s| {
        let handles: Vec<_> = fixtures
            .iter()
            .map(|fixture| {
                let tmp = tmp_dir;
                s.spawn(move || {
                    let array = fixture.build(tmp)?;
                    Ok((fixture.as_ref(), array))
                })
            })
            .collect();

        handles
            .into_iter()
            .map(|h| {
                h.join()
                    .unwrap_or_else(|_| Err(vortex_err!("fixture thread panicked")))
            })
            .collect()
    });

    let mut built = Vec::with_capacity(fixtures.len());
    let mut errors = Vec::new();
    for result in build_results {
        match result {
            Ok(pair) => {
                eprintln!("  built {}", pair.0.name());
                built.push(pair);
            }
            Err(e) => {
                errors.push(e.to_string());
            }
        }
    }
    if !errors.is_empty() {
        vortex_bail!(
            "{} fixture(s) failed to build:\n  {}",
            errors.len(),
            errors.join("\n  ")
        );
    }
    Ok(built)
}
