// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod arrays;

use std::path::Path;
use std::sync::Arc;

use vortex::file::WriteStrategyBuilder;
use vortex_array::ArrayRef;
use vortex_array::ArrayVisitorExt;
use vortex_array::vtable::ArrayId;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::adapter;
use crate::manifest::FixtureEntry;

/// Top-level trait that the runner (compat-gen / compat-validate) interacts with.
pub trait Fixture {
    /// Filename for this fixture, e.g. "primitives.vortex" or "tpch_lineitem.regular.vortex".
    fn name(&self) -> &str;

    /// A short human-readable description of what this fixture tests.
    fn description(&self) -> &str;

    /// Generate the fixture file(s) into `dir`, returning manifest entries.
    fn write(&self, dir: &Path) -> VortexResult<Vec<FixtureEntry>>;
}

/// A deterministic fixture that produces a single array written via flat layout (no compression).
pub trait FlatLayoutFixture {
    /// The filename for this fixture, e.g. "primitives.vortex".
    fn name(&self) -> &str;

    /// A short human-readable description of what this fixture tests.
    fn description(&self) -> &str;

    /// Build the expected arrays. Must be deterministic under all versions of vortex.
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

/// A fixture backed by a real-world dataset that produces multiple chunks.
///
/// Each dataset fixture is written twice: once with the default (BtrBlocks) compressor
/// and once with compact encodings (Pco + Zstd).
pub trait DatasetFixture {
    /// Base name without strategy suffix, e.g. "tpch_lineitem".
    fn name(&self) -> &str;

    /// A short human-readable description of what this fixture tests.
    fn description(&self) -> &str;

    /// Build the dataset as a chunked array. Must be deterministic.
    fn build(&self) -> VortexResult<ArrayRef>;
}

// ---------------------------------------------------------------------------
// Adapters
// ---------------------------------------------------------------------------

/// Adapts a [`FlatLayoutFixture`] into a [`Fixture`] by writing through the flat layout strategy.
pub struct FlatLayoutAdapter(pub Box<dyn FlatLayoutFixture>);

impl Fixture for FlatLayoutAdapter {
    fn name(&self) -> &str {
        self.0.name()
    }

    fn description(&self) -> &str {
        self.0.description()
    }

    fn write(&self, dir: &Path) -> VortexResult<Vec<FixtureEntry>> {
        let array = self.0.build()?;
        check_expected_encodings(&array, self.0.as_ref())?;
        let path = dir.join(self.name());
        adapter::write_file(&path, array)?;
        Ok(vec![FixtureEntry {
            name: self.name().to_string(),
            description: self.description().to_string(),
        }])
    }
}

/// Adapts a [`DatasetFixture`] into a [`Fixture`] by writing through the compressor pipeline.
pub struct DatasetFixtureAdapter {
    inner: Arc<dyn DatasetFixture>,
    compact: bool,
    file_name: String,
}

impl DatasetFixtureAdapter {
    /// Create a pair of fixtures: one with the default strategy and one with compact encodings.
    pub fn pair(fixture: Box<dyn DatasetFixture>) -> [Box<dyn Fixture>; 2] {
        let inner: Arc<dyn DatasetFixture> = Arc::from(fixture);
        let base = inner.name().to_string();
        [
            Box::new(DatasetFixtureAdapter {
                inner: Arc::clone(&inner),
                compact: false,
                file_name: format!("{base}.regular.vortex"),
            }),
            Box::new(DatasetFixtureAdapter {
                inner,
                compact: true,
                file_name: format!("{base}.compact.vortex"),
            }),
        ]
    }
}

impl Fixture for DatasetFixtureAdapter {
    fn name(&self) -> &str {
        &self.file_name
    }

    fn description(&self) -> &str {
        self.inner.description()
    }

    fn write(&self, dir: &Path) -> VortexResult<Vec<FixtureEntry>> {
        let array = self.inner.build()?;
        let path = dir.join(self.name());
        if self.compact {
            let strategy = WriteStrategyBuilder::default()
                .with_compact_encodings()
                .build();
            adapter::write_compressed(&path, array, strategy)?;
        } else {
            let strategy = WriteStrategyBuilder::default().build();
            adapter::write_compressed(&path, array, strategy)?;
        }
        Ok(vec![FixtureEntry {
            name: self.name().to_string(),
            description: self.description().to_string(),
        }])
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Walk the array tree, collect encoding IDs, and assert that all expected encodings
/// are present. This is a no-op when [`FlatLayoutFixture::expected_encodings`] returns an empty vec.
pub fn check_expected_encodings(
    array: &ArrayRef,
    fixture: &dyn FlatLayoutFixture,
) -> VortexResult<()> {
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
pub fn all_fixtures() -> Vec<Box<dyn Fixture>> {
    let mut out: Vec<Box<dyn Fixture>> = Vec::new();
    for f in arrays::synthetic_fixtures() {
        out.push(Box::new(FlatLayoutAdapter(f)));
    }
    for f in arrays::dataset_fixtures() {
        out.extend(DatasetFixtureAdapter::pair(f));
    }
    out
}
