// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fs;
use std::path::Path;
use std::path::PathBuf;

use vortex_array::assert_arrays_eq;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_utils::aliases::hash_set::HashSet;

use crate::adapter;
use crate::fixtures::all_fixtures;
use crate::manifest::Manifest;

/// Result of validating one version's fixtures.
pub struct VersionResult {
    pub version: String,
    pub passed: usize,
    pub skipped: usize,
    pub failed: Vec<(String, String)>,
}

/// Validate all versions' fixtures against the current reader.
pub fn validate_all(
    source: &FixtureSource,
    versions: &[String],
) -> VortexResult<Vec<VersionResult>> {
    let fixtures = all_fixtures();

    // Generate fresh fixtures into a temp dir.
    let tmp_dir = tempfile::tempdir().map_err(|e| vortex_err!("failed to create temp dir: {e}"))?;
    let mut fresh_names: Vec<String> = Vec::new();
    for fixture in &fixtures {
        let entries = fixture.write(tmp_dir.path())?;
        for entry in entries {
            fresh_names.push(entry.name);
        }
    }

    let fresh_set: HashSet<&str> = fresh_names.iter().map(|n| n.as_str()).collect();

    let mut results = Vec::new();
    for version in versions {
        let result = validate_version(source, version, tmp_dir.path(), &fresh_set)?;
        results.push(result);
    }
    Ok(results)
}

fn validate_version(
    source: &FixtureSource,
    version: &str,
    fresh_dir: &Path,
    fresh_set: &HashSet<&str>,
) -> VortexResult<VersionResult> {
    let manifest = source.fetch_manifest(version)?;
    let mut passed = 0;
    let mut skipped = 0;
    let mut failed = Vec::new();

    for entry in &manifest.fixtures {
        if !fresh_set.contains(entry.name.as_str()) {
            eprintln!(
                "  warn: unknown fixture {} in v{version}, skipping",
                entry.name
            );
            skipped += 1;
            continue;
        }

        eprintln!("  checking {} from v{version}...", entry.name);
        let stored_bytes = source.fetch_fixture(version, &entry.name)?;
        let fresh_path = fresh_dir.join(&entry.name);
        let fresh_bytes = fs::read(&fresh_path).map_err(|e| {
            vortex_err!("failed to read fresh fixture {}: {e}", fresh_path.display())
        })?;

        match validate(stored_bytes, ByteBuffer::from(fresh_bytes)) {
            Ok(()) => passed += 1,
            Err(e) => {
                eprintln!("  FAIL: {} from v{version}: {e}", entry.name);
                failed.push((entry.name.clone(), e.to_string()));
            }
        }
    }

    Ok(VersionResult {
        version: version.to_string(),
        passed,
        skipped,
        failed,
    })
}

fn validate(stored_bytes: ByteBuffer, fresh_bytes: ByteBuffer) -> VortexResult<()> {
    let stored_array = adapter::read_file(stored_bytes)?;
    let fresh_array = adapter::read_file(fresh_bytes)?;

    assert_arrays_eq!(stored_array, fresh_array);
    Ok(())
}

/// Source for fetching fixture files -- either HTTPS or local directory.
pub enum FixtureSource {
    Url(String),
    Dir(PathBuf),
}

impl FixtureSource {
    fn fetch_manifest(&self, version: &str) -> VortexResult<Manifest> {
        let json = match self {
            FixtureSource::Url(base) => {
                let url = format!("{base}/v{version}/manifest.json");
                http_get_bytes(&url)?
            }
            FixtureSource::Dir(dir) => {
                let path = dir.join(format!("v{version}")).join("manifest.json");
                fs::read(&path)
                    .map_err(|e| vortex_err!("failed to read {}: {e}", path.display()))?
            }
        };
        serde_json::from_slice(&json)
            .map_err(|e| vortex_err!("failed to parse manifest for v{version}: {e}"))
    }

    fn fetch_fixture(&self, version: &str, name: &str) -> VortexResult<ByteBuffer> {
        let bytes = match self {
            FixtureSource::Url(base) => {
                let url = format!("{base}/v{version}/{name}");
                http_get_bytes(&url)?
            }
            FixtureSource::Dir(dir) => {
                let path = dir.join(format!("v{version}")).join(name);
                fs::read(&path)
                    .map_err(|e| vortex_err!("failed to read {}: {e}", path.display()))?
            }
        };
        Ok(ByteBuffer::from(bytes))
    }
}

/// Discover versions from a versions.json file, or from local directory listing.
pub fn discover_versions(source: &FixtureSource) -> VortexResult<Vec<String>> {
    match source {
        FixtureSource::Url(base) => {
            let url = format!("{base}/versions.json");
            let bytes = http_get_bytes(&url)?;
            let versions: Vec<String> = serde_json::from_slice(&bytes)
                .map_err(|e| vortex_err!("failed to parse versions.json: {e}"))?;
            Ok(versions)
        }
        FixtureSource::Dir(dir) => {
            let mut versions = Vec::new();
            for entry in fs::read_dir(dir)
                .map_err(|e| vortex_err!("failed to read dir {}: {e}", dir.display()))?
            {
                let entry = entry.map_err(|e| vortex_err!("failed to read dir entry: {e}"))?;
                let name = entry.file_name();
                let name = name.to_string_lossy();
                if let Some(version) = name.strip_prefix('v')
                    && entry.path().join("manifest.json").exists()
                {
                    versions.push(version.to_string());
                }
            }
            versions.sort();
            Ok(versions)
        }
    }
}

fn http_get_bytes(url: &str) -> VortexResult<Vec<u8>> {
    let response = reqwest::blocking::get(url)
        .map_err(|e| vortex_err!("HTTP request failed for {url}: {e}"))?;
    if !response.status().is_success() {
        vortex_bail!("HTTP {} fetching {url}", response.status());
    }
    response
        .bytes()
        .map(|b| b.to_vec())
        .map_err(|e| vortex_err!("failed to read response body from {url}: {e}"))
}
