// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::path::Path;
use std::thread;

use chrono::Utc;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;

use crate::adapter;
use crate::fixtures::Fixture;
use crate::fixtures::all_fixtures;
use crate::manifest::FixtureEntry;
use crate::manifest::Manifest;

/// Generate all fixtures into a local directory.
///
/// Three-phase pipeline:
/// 1. **Setup** (async I/O) — run each fixture's `setup()` concurrently via
///    `tokio::spawn_blocking`. Downloads external data, prepares files in
///    `tmp_dir`.
/// 2. **Build** (CPU, parallel) — construct arrays in a thread pool via
///    `std::thread::scope`.
/// 3. **Write** — serialize `.vortex` files to disk.
///
/// All fixtures are built before any are written, so a failure in one fixture
/// does not leave a partial directory.
pub fn generate(output_dir: &Path, version: &str) -> VortexResult<()> {
    let fixtures = all_fixtures();

    // Create a shared tmp_dir for setup / scratch space.
    let tmp_dir = output_dir.join(".tmp");
    std::fs::create_dir_all(&tmp_dir).map_err(|e| vortex_err!("failed to create tmp dir: {e}"))?;

    // Phase 1: Run setup for all fixtures concurrently.
    eprintln!("[1/3] Setting up {} fixtures...", fixtures.len());
    run_setup_async(&fixtures, &tmp_dir)?;

    // Phase 2: Build all fixtures in parallel.
    eprintln!("[2/3] Building {} fixtures...", fixtures.len());
    let built = run_build_parallel(&fixtures, &tmp_dir)?;

    // Phase 3: Write to disk.
    eprintln!("[3/3] Writing to {}...", output_dir.display());
    std::fs::create_dir_all(output_dir)
        .map_err(|e| vortex_err!("failed to create output dir: {e}"))?;

    let mut entries = Vec::with_capacity(built.len());
    for (name, chunks) in built {
        let path = output_dir.join(name);
        adapter::write_file(&path, chunks)?;
        entries.push(FixtureEntry {
            name: name.to_string(),
            since: version.to_string(),
        });
        eprintln!("  wrote {name}");
    }

    let manifest = Manifest {
        version: version.to_string(),
        generated_at: Utc::now(),
        fixtures: entries,
    };
    let manifest_json = serde_json::to_string_pretty(&manifest)
        .map_err(|e| vortex_err!("failed to serialize manifest: {e}"))?;
    std::fs::write(
        output_dir.join("manifest.json"),
        format!("{manifest_json}\n"),
    )
    .map_err(|e| vortex_err!("failed to write manifest: {e}"))?;
    eprintln!("  wrote manifest.json");

    eprintln!(
        "\ndone: {} fixtures for v{version} in {}",
        manifest.fixtures.len(),
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
) -> VortexResult<Vec<(&'a str, Vec<vortex_array::ArrayRef>)>> {
    let build_results: Vec<VortexResult<(&str, Vec<vortex_array::ArrayRef>)>> =
        thread::scope(|s| {
            let handles: Vec<_> = fixtures
                .iter()
                .map(|fixture| {
                    let tmp = tmp_dir;
                    s.spawn(move || {
                        let chunks = fixture.build(tmp)?;
                        Ok((fixture.name(), chunks))
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
            Ok((name, chunks)) => {
                eprintln!("  built {name}");
                built.push((name, chunks));
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
