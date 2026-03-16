// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod clickbench;
#[allow(clippy::cast_possible_truncation)]
mod encoding_fixtures;
mod synthetic;
mod tpch;

use std::path::Path;

use vortex_array::ArrayRef;
use vortex_array::vtable::ArrayId;
use vortex_error::VortexResult;
use vortex_layout::LayoutId;

/// Declares which encoding(s) a fixture is designed to exercise.
#[derive(Debug, Clone)]
pub enum ExpectedEncoding {
    /// An array-level encoding, e.g. `"vortex.dict"`, `"vortex.fsst"`.
    Array(ArrayId),
    /// A layout-level encoding, e.g. `"vortex.chunked"`, `"vortex.flat"`.
    Layout(LayoutId),
}

/// A deterministic fixture that produces the same arrays every time.
pub trait Fixture: Send + Sync {
    /// The filename for this fixture, e.g. "primitives.vortex".
    fn name(&self) -> &str;

    /// Human-readable description of what this fixture tests.
    fn description(&self) -> &str;

    /// Encodings this fixture is designed to exercise.
    fn expected_encodings(&self) -> Vec<ExpectedEncoding>;

    /// Optional setup step (e.g. download external data).
    fn setup(&self, _tmp_dir: &Path) -> VortexResult<()> {
        Ok(())
    }

    /// Build the expected arrays. Must be deterministic.
    ///
    /// Returns a `Vec` to support chunked fixtures (multiple chunks).
    /// Single-array fixtures return a one-element vec.
    fn build(&self, tmp_dir: &Path) -> VortexResult<Vec<ArrayRef>>;
}

/// All registered fixtures.
pub fn all_fixtures() -> Vec<Box<dyn Fixture>> {
    let mut fixtures: Vec<Box<dyn Fixture>> = vec![
        // Existing synthetic fixtures
        Box::new(synthetic::PrimitivesFixture),
        Box::new(synthetic::StringsFixture),
        Box::new(synthetic::BooleansFixture),
        Box::new(synthetic::NullableFixture),
        Box::new(synthetic::StructNestedFixture),
        Box::new(synthetic::ChunkedFixture),
        Box::new(synthetic::ListFixture),
        Box::new(synthetic::FixedSizeListFixture),
        Box::new(synthetic::NullFixture),
        // Real-world fixtures
        Box::new(tpch::TpchLineitemFixture),
        Box::new(tpch::TpchOrdersFixture),
        Box::new(clickbench::ClickBenchHits1kFixture),
    ];

    // Per-encoding fixtures
    fixtures.extend(encoding_fixtures::all_encoding_fixtures());

    fixtures
}
