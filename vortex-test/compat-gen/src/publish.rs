// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::path::Path;
use std::thread;

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;

use crate::generate;
use crate::manifest::Manifest;
use crate::store::FixtureStore;

/// Publish fixtures for a version to a store.
///
/// Stages:
/// 1. Generate all fixtures locally (fail before touching the store).
/// 2. Merge manifest with the previous version in the store.
/// 3. Upload all fixture files in parallel.
/// 4. Write the merged manifest.
/// 5. Update `versions.json` (with ETag locking for S3).
///
/// If `dry_run` is true, only stages 1-2 run and the merged manifest is printed.
pub fn publish(
    store: &dyn FixtureStore,
    version: &str,
    tmp_dir: Option<&Path>,
    dry_run: bool,
) -> VortexResult<()> {
    // Stage 1: Generate fixtures locally.
    let owned_tmp;
    let output_dir = if let Some(dir) = tmp_dir {
        std::fs::create_dir_all(dir).map_err(|e| vortex_err!("failed to create tmp dir: {e}"))?;
        dir.to_path_buf()
    } else {
        owned_tmp =
            tempfile::tempdir().map_err(|e| vortex_err!("failed to create temp dir: {e}"))?;
        owned_tmp.path().join("fixtures")
    };

    generate::generate(&output_dir, version)?;

    // Stage 2: Merge manifest with previous version in the store.
    eprintln!("merging manifest...");
    let manifest_path = output_dir.join("manifest.json");
    let manifest_bytes = std::fs::read(&manifest_path)
        .map_err(|e| vortex_err!("failed to read {}: {e}", manifest_path.display()))?;
    let manifest: Manifest = serde_json::from_slice(&manifest_bytes)
        .map_err(|e| vortex_err!("failed to parse manifest: {e}"))?;

    let merged = merge_manifest(store, manifest, version)?;
    let manifest_json = serde_json::to_string_pretty(&merged)
        .map_err(|e| vortex_err!("failed to serialize manifest: {e}"))?;

    if dry_run {
        eprintln!("dry run — not uploading to store.");
        eprintln!("\nMerged manifest:\n{manifest_json}");
        eprintln!("\nGenerated fixtures in: {}", output_dir.display());
        return Ok(());
    }

    // Stage 3: Upload all fixture files in parallel.
    eprintln!(
        "uploading {} fixtures to {}...",
        merged.fixtures.len(),
        store.display_name()
    );
    upload_fixtures_parallel(store, &output_dir, &merged, version)?;

    // Stage 4: Write the merged manifest (after all files are uploaded).
    let manifest_key = format!("v{version}/manifest.json");
    store.write(&manifest_key, format!("{manifest_json}\n").as_bytes())?;
    eprintln!("  uploaded manifest.json");

    // Stage 5: Update versions.json (with ETag locking for S3).
    eprintln!("updating versions.json...");
    let mut versions = store.read_versions_json()?;
    if !versions.contains(&version.to_string()) {
        versions.push(version.to_string());
        versions.sort_by_key(|v| version_sort_key(v));
    }
    store.write_versions_json(&versions)?;
    eprintln!("  updated versions.json");

    eprintln!(
        "\ndone: {} fixtures for v{version} published to {}",
        merged.fixtures.len(),
        store.display_name()
    );
    Ok(())
}

/// Upload fixture files in parallel using scoped threads.
fn upload_fixtures_parallel(
    store: &dyn FixtureStore,
    output_dir: &Path,
    manifest: &Manifest,
    version: &str,
) -> VortexResult<()> {
    let errors = parking_lot::Mutex::new(Vec::<String>::new());

    thread::scope(|s| {
        for entry in &manifest.fixtures {
            let local_path = output_dir.join(&entry.name);
            let key = format!("v{version}/{}", entry.name);
            let errors = &errors;

            s.spawn(move || match store.write_from_path(&key, &local_path) {
                Ok(()) => eprintln!("  uploaded {}", entry.name),
                Err(e) => {
                    eprintln!("  FAIL uploading {}: {e}", entry.name);
                    errors.lock().push(format!("{}: {e}", entry.name));
                }
            });
        }
    });

    let errors = errors.into_inner();
    if !errors.is_empty() {
        vortex_bail!(
            "{} fixture upload(s) failed:\n  {}",
            errors.len(),
            errors.join("\n  ")
        );
    }
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
        let missing_list: Vec<&str> = missing.iter().map(|s| s.as_str()).collect();
        vortex_bail!(
            "fixtures removed since previous version: {}. Fixtures must never be removed.",
            missing_list.join(", ")
        );
    }

    // Merge since values.
    for entry in &mut manifest.fixtures {
        if let Some(since) = prev_since.get(&entry.name) {
            entry.since.clone_from(since);
        }
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
