// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod clickbench;
mod synthetic;
mod tpch;

use std::path::Path;

use vortex_array::ArrayRef;
use vortex_error::VortexResult;

/// A deterministic fixture that produces the same arrays every time.
///
/// Each fixture is given a `tmp_dir` it can use as scratch space for
/// downloading external data or intermediate files. Fixtures that don't
/// need external data simply ignore it.
pub trait Fixture: Send + Sync {
    /// The filename for this fixture, e.g. "primitives.vortex".
    fn name(&self) -> &str;

    /// Build the expected arrays, using `tmp_dir` as scratch space for any
    /// downloads or intermediate files.
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
