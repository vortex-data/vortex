// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod clickbench;
mod synthetic;
mod tpch;

use std::fmt;
use std::path::Path;

use vortex::layout::LayoutId;
use vortex_array::ArrayRef;
use vortex_array::vtable::ArrayId;
use vortex_error::VortexResult;

/// An encoding that a fixture is designed to exercise.
///
/// Used for coverage tracking and verification: at generation time, the tool
/// checks that every declared encoding actually appears in the written file.
/// At check time, it verifies the encodings are still present.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExpectedEncoding {
    /// Array-level encoding (compression layer), e.g. `vortex.primitive`.
    Array(ArrayId),
    /// Layout-level encoding (storage layer), e.g. `vortex.flat`.
    Layout(LayoutId),
}

impl fmt::Display for ExpectedEncoding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExpectedEncoding::Array(id) => write!(f, "array:{id}"),
            ExpectedEncoding::Layout(id) => write!(f, "layout:{id}"),
        }
    }
}

/// A deterministic fixture that produces the same arrays every time.
///
/// Lifecycle:
/// 1. **Setup** (optional, async I/O) — download external data or prepare
///    intermediate files in `tmp_dir`. Run concurrently across fixtures.
/// 2. **Build** (CPU, parallel) — construct arrays from scratch or from
///    data prepared during setup.
pub trait Fixture: Send + Sync {
    /// Unique name for this fixture, used as the output filename.
    fn name(&self) -> &str;

    /// Human-readable description of what this fixture tests.
    fn description(&self) -> &str;

    /// Encodings this fixture is designed to exercise.
    ///
    /// At generation time, the tool verifies that all declared encodings
    /// appear in the written file. At check time, it verifies they are
    /// still present. This is a subset check — the file may contain
    /// additional encodings not listed here.
    fn expected_encodings(&self) -> Vec<ExpectedEncoding>;

    /// Optional setup phase for downloading external data or preparing files.
    ///
    /// Called before `build()`. Runs concurrently across fixtures via
    /// `tokio::spawn_blocking`. The default implementation is a no-op.
    fn setup(&self, _tmp_dir: &Path) -> VortexResult<()> {
        Ok(())
    }

    /// Build the expected arrays, using `tmp_dir` for any data prepared
    /// during `setup()`.
    ///
    /// Must be deterministic. Returns a `Vec` to support chunked fixtures.
    fn build(&self, tmp_dir: &Path) -> VortexResult<Vec<ArrayRef>>;
}

/// All registered fixtures.
pub fn all_fixtures() -> Vec<Box<dyn Fixture>> {
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
