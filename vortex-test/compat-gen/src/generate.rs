// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use chrono::Utc;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;

use crate::adapter;
use crate::fixtures::all_fixtures;
use crate::manifest::FixtureEntry;
use crate::manifest::Manifest;
use crate::store::FixtureStore;

/// Generate fixtures for a version and write them into a store.
///
/// If `dry_run` is true, generates fixtures and merges the manifest but does not
/// write anything to the store.
pub fn generate(
    store: &dyn FixtureStore,
    version: &str,
    dry_run: bool,
    skip_build: bool,
) -> VortexResult<()> {
    let tmp_dir = tempfile::tempdir().map_err(|e| vortex_err!("failed to create temp dir: {e}"))?;
    let output_dir = tmp_dir.path().join(format!("v{version}"));
    std::fs::create_dir_all(&output_dir)
        .map_err(|e| vortex_err!("failed to create output dir: {e}"))?;

    // Step 1: Generate fixtures into temp directory.
    let fixtures = all_fixtures();
    let mut entries = Vec::with_capacity(fixtures.len());

    if !skip_build {
        eprintln!("[1/3] Generating fixtures for v{version}...");
        for fixture in &fixtures {
            let chunks = fixture.build()?;
            let path = output_dir.join(fixture.name());
            adapter::write_file(&path, chunks)?;
            entries.push(FixtureEntry {
                name: fixture.name().to_string(),
                since: version.to_string(),
            });
            eprintln!("  wrote {}", fixture.name());
        }
    } else {
        eprintln!("[1/3] Skipping build (--skip-build)");
        // Build entries from fixture registry (we still need the manifest).
        for fixture in &fixtures {
            entries.push(FixtureEntry {
                name: fixture.name().to_string(),
                since: version.to_string(),
            });
        }
    }

    let manifest = Manifest {
        version: version.to_string(),
        generated_at: Utc::now(),
        fixtures: entries,
    };

    // Step 2: Merge manifest with previous version's manifest from the store.
    eprintln!("[2/3] Merging manifest...");
    let merged = merge_manifest(store, manifest, version)?;

    let manifest_json = serde_json::to_string_pretty(&merged)
        .map_err(|e| vortex_err!("failed to serialize manifest: {e}"))?;

    if dry_run {
        eprintln!("[3/3] Dry run — not writing to store.");
        eprintln!("\nMerged manifest:\n{manifest_json}");
        return Ok(());
    }

    // Step 3: Write to store.
    eprintln!("[3/3] Writing to store ({})...", store.display_name());

    // Write manifest.
    let manifest_key = format!("v{version}/manifest.json");
    store.write(&manifest_key, format!("{manifest_json}\n").as_bytes())?;
    eprintln!("  wrote manifest.json");

    // Write fixture files.
    if !skip_build {
        for fixture in &fixtures {
            let local_path = output_dir.join(fixture.name());
            let key = format!("v{version}/{}", fixture.name());
            store.write_from_path(&key, &local_path)?;
            eprintln!("  wrote {}", fixture.name());
        }
    }

    // Update versions.json.
    let mut versions = store.read_versions_json()?;
    if !versions.contains(&version.to_string()) {
        versions.push(version.to_string());
        versions.sort_by_key(|a| version_sort_key(a));
    }
    store.write_versions_json(&versions)?;
    eprintln!("  updated versions.json");

    eprintln!(
        "\ndone: {} fixtures for v{version} written to {}",
        merged.fixtures.len(),
        store.display_name()
    );
    Ok(())
}

/// Merge `since` values from the previous version's manifest in the store.
fn merge_manifest(
    store: &dyn FixtureStore,
    mut manifest: Manifest,
    current_version: &str,
) -> VortexResult<Manifest> {
    let versions = store.list_versions()?;
    let candidates: Vec<&String> = versions
        .iter()
        .filter(|v| v.as_str() != current_version)
        .collect();

    if candidates.is_empty() {
        eprintln!("  no previous version found, skipping merge");
        return Ok(manifest);
    }

    // Take the highest version before current.
    let Some(latest) = candidates.last() else {
        eprintln!("  no previous version found, skipping merge");
        return Ok(manifest);
    };
    eprintln!("  previous version: {latest}");

    let prev_manifest = match store.fetch_manifest(latest) {
        Ok(m) => m,
        Err(_) => {
            eprintln!("  warning: could not fetch previous manifest, skipping merge");
            return Ok(manifest);
        }
    };

    // Build a map of previous fixture names -> since values.
    let prev_since: vortex_utils::aliases::hash_map::HashMap<String, String> = prev_manifest
        .fixtures
        .iter()
        .map(|e| (e.name.clone(), e.since.clone()))
        .collect();

    let gen_names: vortex_utils::aliases::hash_set::HashSet<String> =
        manifest.fixtures.iter().map(|e| e.name.clone()).collect();

    // Additive-only check.
    let missing: Vec<&String> = prev_since
        .keys()
        .filter(|name| !gen_names.contains(name.as_str()))
        .collect();
    if !missing.is_empty() {
        vortex_bail!(
            "fixtures removed since previous version: {:?}. Fixtures must never be removed.",
            missing
        );
    }

    // Merge since values.
    for entry in &mut manifest.fixtures {
        if let Some(since) = prev_since.get(&entry.name) {
            entry.since.clone_from(since);
        }
        // New fixtures keep since = current_version (already set).
    }

    let new_count = manifest.fixtures.len() - prev_since.len();
    eprintln!(
        "  merged: {} existing, {} new fixtures",
        prev_since.len(),
        new_count
    );
    Ok(manifest)
}

fn version_sort_key(v: &str) -> Vec<u64> {
    v.split('.').filter_map(|s| s.parse().ok()).collect()
}
