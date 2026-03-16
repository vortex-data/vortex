// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::path::Path;
use std::thread;

use serde::Serialize;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;

use crate::adapter;
use crate::fixtures::Fixture;
use crate::fixtures::all_fixtures;

#[derive(Serialize)]
struct FixturesJson {
    fixtures: Vec<FixtureInfo>,
}

#[derive(Serialize)]
struct FixtureInfo {
    name: String,
    description: String,
    expected_encodings: Vec<String>,
}

/// Generate all fixtures into `output_dir`.
///
/// Three phases:
/// 1. **Setup** — run each fixture's `setup()` concurrently (async I/O).
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

    // Phase 1: Setup (concurrent I/O).
    eprintln!("[1/3] Setting up {} fixtures...", fixtures.len());
    run_setup_async(&fixtures, &tmp_dir)?;

    // Phase 2: Build (parallel CPU).
    eprintln!("[2/3] Building {} fixtures...", fixtures.len());
    let built = run_build_parallel(&fixtures, &tmp_dir)?;

    // Phase 3: Write to disk.
    eprintln!("[3/3] Writing to {}...", output_dir.display());
    std::fs::create_dir_all(output_dir)
        .map_err(|e| vortex_err!("failed to create output dir: {e}"))?;

    let mut infos = Vec::with_capacity(built.len());
    for (fixture, chunks) in &built {
        let path = output_dir.join(fixture.name());
        adapter::write_file(&path, chunks.clone())?;
        infos.push(FixtureInfo {
            name: fixture.name().to_string(),
            description: fixture.description().to_string(),
            expected_encodings: fixture
                .expected_encodings()
                .iter()
                .map(|e| e.to_string())
                .collect(),
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

/// Run `setup()` for all fixtures concurrently via tokio.
fn run_setup_async(fixtures: &[Box<dyn Fixture>], tmp_dir: &Path) -> VortexResult<()> {
    let rt = tokio::runtime::Runtime::new()
        .map_err(|e| vortex_err!("failed to create tokio runtime: {e}"))?;

    let errors: Vec<String> = rt.block_on(async {
        let mut handles = Vec::with_capacity(fixtures.len());
        for fixture in fixtures {
            let name = fixture.name().to_string();
            // Safety: fixtures are Send + Sync and live for the duration of
            // this block_on call. We transmute the lifetime to 'static so we
            // can spawn, but the block_on ensures the borrow is valid.
            let fixture: &'static dyn Fixture = unsafe { std::mem::transmute(fixture.as_ref()) };
            let tmp = tmp_dir.to_path_buf();
            handles.push(tokio::spawn(async move {
                let result = tokio::task::spawn_blocking(move || fixture.setup(&tmp)).await;
                match result {
                    Ok(Ok(())) => {
                        eprintln!("  setup {name}");
                        None
                    }
                    Ok(Err(e)) => Some(format!("{name}: {e}")),
                    Err(e) => Some(format!("{name}: task panicked: {e}")),
                }
            }));
        }

        let mut errors = Vec::new();
        for handle in handles {
            if let Ok(Some(err)) = handle.await {
                errors.push(err);
            }
        }
        errors
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
    fixtures: &'a [Box<dyn Fixture>],
    tmp_dir: &Path,
) -> VortexResult<Vec<(&'a dyn Fixture, Vec<vortex_array::ArrayRef>)>> {
    let build_results: Vec<VortexResult<(&dyn Fixture, Vec<vortex_array::ArrayRef>)>> =
        thread::scope(|s| {
            let handles: Vec<_> = fixtures
                .iter()
                .map(|fixture| {
                    let tmp = tmp_dir;
                    s.spawn(move || {
                        let chunks = fixture.build(tmp)?;
                        Ok((fixture.as_ref(), chunks))
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
