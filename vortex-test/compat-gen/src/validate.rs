// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::path::PathBuf;

use vortex_array::assert_arrays_eq;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_utils::aliases::hash_map::HashMap;

use crate::adapter;
use crate::fixtures::ArrayFixture;
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
    let fixture_map: HashMap<&str, &dyn ArrayFixture> =
        fixtures.iter().map(|f| (f.name(), f.as_ref())).collect();

    let mut results = Vec::new();
    for version in versions {
        let result = validate_version(source, version, &fixture_map)?;
        results.push(result);
    }
    Ok(results)
}

fn validate_version(
    source: &FixtureSource,
    version: &str,
    fixture_map: &HashMap<&str, &dyn ArrayFixture>,
) -> VortexResult<VersionResult> {
    let manifest = source.fetch_manifest(version)?;
    let mut passed = 0;
    let mut skipped = 0;
    let mut failed = Vec::new();

    for entry in &manifest.fixtures {
        let Some(fixture) = fixture_map.get(entry.name.as_str()) else {
            eprintln!(
                "  warn: unknown fixture {} in v{version}, skipping",
                entry.name
            );
            skipped += 1;
            continue;
        };

        eprintln!("  checking {} from v{version}...", entry.name);
        let bytes = source.fetch_fixture(version, &entry.name)?;
        match validate(bytes, *fixture) {
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

fn validate(bytes: ByteBuffer, fixture: &dyn ArrayFixture) -> VortexResult<()> {
    let actual = adapter::read_file(bytes)?;
    let expected = fixture.build()?;

    assert_arrays_eq!(actual, expected);
    Ok(())
}

/// Source for fetching fixture files — either HTTPS or local directory.
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
                std::fs::read(&path)
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
                std::fs::read(&path)
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
            for entry in std::fs::read_dir(dir)
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
