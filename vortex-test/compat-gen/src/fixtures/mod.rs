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

    /// Build the expected arrays. Must be deterministic.
    ///
    /// Returns a `Vec` to support chunked fixtures (multiple chunks).
    /// Single-array fixtures return a one-element vec.
    fn build(&self) -> VortexResult<Vec<ArrayRef>>;

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

/// Walk every array in `chunks`, collect encoding IDs, and assert that all expected encodings
/// are present. This is a no-op when [`ArrayFixture::expected_encodings`] returns an empty vec.
pub fn check_expected_encodings(
    chunks: &[ArrayRef],
    fixture: &dyn ArrayFixture,
) -> VortexResult<()> {
    let expected = fixture.expected_encodings();
    if expected.is_empty() {
        return Ok(());
    }

    let mut found: Vec<ArrayId> = Vec::new();
    for chunk in chunks {
        for array in chunk.depth_first_traversal() {
            let id = array.encoding_id();
            if !found.contains(&id) {
                found.push(id);
            }
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
    vec![
        Box::new(synthetic::PrimitivesFixture),
        Box::new(synthetic::StringsFixture),
        Box::new(synthetic::BooleansFixture),
        Box::new(synthetic::NullableFixture),
        Box::new(synthetic::StructNestedFixture),
        Box::new(synthetic::ChunkedFixture),
        Box::new(tpch::TpchLineitemFixture),
        Box::new(tpch::TpchOrdersFixture),
        Box::new(clickbench::ClickBenchHits1kFixture),
    ]
}
