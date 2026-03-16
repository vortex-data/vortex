// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod clickbench;
mod synthetic;
mod tpch;

use vortex_array::ArrayRef;
use vortex_array::ArrayVisitorExt;
use vortex_array::vtable::ArrayId;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

/// A deterministic fixture that produces the same arrays every time.
pub trait ArrayFixture: Send + Sync {
    /// The filename for this fixture, e.g. "primitives.vortex".
    fn name(&self) -> &str;

    /// A short human-readable description of what this fixture tests.
    fn description(&self) -> &str;

    /// Build the expected arrays. Must be deterministic under all version of vortex.
    fn build(&self) -> VortexResult<ArrayRef>;

    /// Encoding IDs that must appear somewhere in the array tree produced by [`Self::build`].
    ///
    /// Only include encodings that the fixture is specifically testing, not incidental ones
    /// (e.g. a primitives fixture should not list struct even if it wraps values in one).
    ///
    /// An empty slice (the default) disables the check.
    fn expected_encodings(&self) -> Vec<ArrayId> {
        vec![]
    }
}

/// Walk the array tree, collect encoding IDs, and assert that all expected encodings
/// are present. This is a no-op when [`ArrayFixture::expected_encodings`] returns an empty vec.
pub fn check_expected_encodings(array: &ArrayRef, fixture: &dyn ArrayFixture) -> VortexResult<()> {
    let expected = fixture.expected_encodings();
    if expected.is_empty() {
        return Ok(());
    }

    let mut found: Vec<ArrayId> = Vec::new();
    for node in array.depth_first_traversal() {
        let id = node.encoding_id();
        if !found.contains(&id) {
            found.push(id);
        }
    }

    let missing: Vec<&ArrayId> = expected.iter().filter(|id| !found.contains(id)).collect();

    if !missing.is_empty() {
        vortex_bail!(
            "fixture '{}' is missing expected encodings: {:?} (found: {:?})",
            fixture.name(),
            missing,
            found,
        );
    }

    Ok(())
}

/// All registered fixtures.
pub fn all_fixtures() -> Vec<Box<dyn ArrayFixture>> {
    let mut fixtures = Vec::new();
    fixtures.extend(synthetic::fixtures());
    fixtures.extend(tpch::fixtures());
    fixtures.extend(clickbench::fixtures());
    fixtures
}
