// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::path::Path;
use std::thread;

use chrono::Utc;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;

use crate::adapter;
use crate::fixtures::all_fixtures;
use crate::manifest::FixtureEntry;
use crate::manifest::Manifest;

/// Generate all fixtures into a local directory.
///
/// Two-phase pipeline:
/// 1. **Build** — construct arrays from scratch or from downloaded data
///    (parallel thread pool). Each fixture receives a `tmp_dir` it can use
///    as scratch space for downloads or intermediate files.
/// 2. **Write** — serialize `.vortex` files to disk.
///
/// All fixtures are built before any are written, so a failure in one fixture
/// does not leave a partial directory.
pub fn generate(output_dir: &Path, version: &str) -> VortexResult<()> {
    let fixtures = all_fixtures();

    // Create a shared tmp_dir for downloads / scratch space.
    let tmp_dir = output_dir.join(".tmp");
    std::fs::create_dir_all(&tmp_dir).map_err(|e| vortex_err!("failed to create tmp dir: {e}"))?;

    // Phase 1: Build all fixtures in parallel.
    eprintln!("[1/2] Building {} fixtures...", fixtures.len());
    let build_results: Vec<VortexResult<(&str, Vec<vortex_array::ArrayRef>)>> =
        thread::scope(|s| {
            let handles: Vec<_> = fixtures
                .iter()
                .map(|fixture| {
                    let tmp = &tmp_dir;
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

    // Collect results, fail if any build failed.
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

    // Phase 2: Write to disk.
    eprintln!("[2/2] Writing to {}...", output_dir.display());
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
