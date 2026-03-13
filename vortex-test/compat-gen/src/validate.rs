// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::IntoArray;
use vortex_array::arrays::ChunkedArray;
use vortex_array::assert_arrays_eq;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_utils::aliases::hash_map::HashMap;

use crate::adapter;
use crate::fixtures::Fixture;
use crate::fixtures::all_fixtures;
use crate::store::FixtureStore;

/// Result of validating one version's fixtures.
pub struct VersionResult {
    pub version: String,
    pub passed: usize,
    pub skipped: usize,
    pub failed: Vec<(String, String)>,
}

/// Validate all versions' fixtures against the current reader.
pub fn validate_all(
    store: &dyn FixtureStore,
    versions: &[String],
) -> VortexResult<Vec<VersionResult>> {
    let fixtures = all_fixtures();
    let fixture_map: HashMap<&str, &dyn Fixture> =
        fixtures.iter().map(|f| (f.name(), f.as_ref())).collect();

    let tmp_dir = tempfile::tempdir()
        .map_err(|e| vortex_error::vortex_err!("failed to create temp dir: {e}"))?;

    let mut results = Vec::new();
    for version in versions {
        let result = validate_version(store, version, &fixture_map, tmp_dir.path())?;
        results.push(result);
    }
    Ok(results)
}

fn validate_version(
    store: &dyn FixtureStore,
    version: &str,
    fixture_map: &HashMap<&str, &dyn Fixture>,
    tmp_dir: &std::path::Path,
) -> VortexResult<VersionResult> {
    let manifest = store.fetch_manifest(version)?;
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
        let bytes = store.fetch_fixture(version, &entry.name)?;
        match validate_one(ByteBuffer::from(bytes), *fixture, tmp_dir) {
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

fn validate_one(
    bytes: ByteBuffer,
    fixture: &dyn Fixture,
    tmp_dir: &std::path::Path,
) -> VortexResult<()> {
    let actual = adapter::read_file(bytes)?;
    fixture.setup(tmp_dir)?;
    let expected = fixture.build(tmp_dir)?;

    let actual_dtype = actual[0].dtype().clone();
    let expected_dtype = expected[0].dtype().clone();
    let actual_arr = ChunkedArray::try_new(actual, actual_dtype)?.into_array();
    let expected_arr = ChunkedArray::try_new(expected, expected_dtype)?.into_array();

    assert_arrays_eq!(actual_arr, expected_arr);
    Ok(())
}

/// Run validation and print results. Returns error if any fixture failed.
pub fn run_check(
    store: &dyn FixtureStore,
    filter_versions: Option<Vec<String>>,
) -> VortexResult<()> {
    let versions = match filter_versions {
        Some(v) => v,
        None => {
            eprintln!("discovering versions...");
            store.list_versions()?
        }
    };

    eprintln!(
        "testing {} version(s): {}",
        versions.len(),
        versions.join(", ")
    );

    let results = validate_all(store, &versions)?;

    let mut total_passed = 0;
    let mut total_failed = 0;
    let mut total_skipped = 0;

    for r in &results {
        total_passed += r.passed;
        total_failed += r.failed.len();
        total_skipped += r.skipped;
        if r.failed.is_empty() {
            eprintln!(
                "  v{}: {} passed, {} skipped",
                r.version, r.passed, r.skipped
            );
        } else {
            eprintln!(
                "  v{}: {} passed, {} FAILED, {} skipped",
                r.version,
                r.passed,
                r.failed.len(),
                r.skipped
            );
            for (name, err) in &r.failed {
                eprintln!("    FAIL {name}: {err}");
            }
        }
    }

    eprintln!("\nresult: {total_passed} passed, {total_failed} failed, {total_skipped} skipped");

    if total_failed > 0 {
        vortex_bail!("{total_failed} fixture(s) failed validation");
    }

    Ok(())
}
