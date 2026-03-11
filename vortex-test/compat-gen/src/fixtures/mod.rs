mod synthetic;
mod tpch;

use vortex_array::ArrayRef;

/// A deterministic fixture that produces the same arrays every time.
pub trait Fixture: Send + Sync {
    /// The filename for this fixture, e.g. "primitives.vortex".
    fn name(&self) -> &str;

    /// Build the expected arrays. Must be deterministic.
    ///
    /// Returns a `Vec` to support chunked fixtures (multiple chunks).
    /// Single-array fixtures return a one-element vec.
    fn build(&self) -> Vec<ArrayRef>;
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
    ]
}
