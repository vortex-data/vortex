// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Aggregate summaries used by compression schemes.
//!
//! Schemes request ordinary aggregates through [`ArrayInput`]. Distribution partials retain native
//! keys for dictionary construction, while scalar summaries remain ordinary aggregate results.

use std::sync::Arc;

use vortex_array::ArrayInput;
use vortex_array::ExecutionCtx;
use vortex_array::aggregate_fn::AggregateFn;
use vortex_array::aggregate_fn::AggregateFnVTableExt;
use vortex_array::aggregate_fn::EmptyOptions;
use vortex_array::aggregate_fn::NumericalAggregateOpts;
use vortex_array::aggregate_fn::fns::count::Count;
use vortex_array::aggregate_fn::fns::min_max::MinMax;
use vortex_array::aggregate_fn::fns::min_max::MinMaxResult;
use vortex_array::aggregate_fn::fns::null_count::NullCount;
use vortex_array::match_each_integer_ptype;
use vortex_error::VortexExpect;

use crate::scheme::CompressorContext;

mod float;
pub use float::FloatDistinct;
pub use float::FloatDistinctPartial;
pub use float::FloatDistribution;

mod histogram;
pub use histogram::BitWidthHistogram;

mod integer;
pub use integer::IntegerDistribution;
pub use integer::IntegerFrequencies;
pub use integer::IntegerFrequenciesPartial;

mod prefix;
pub(crate) use prefix::VarBinViewPrefixDistinct;

mod runs;
pub use runs::RunSummary;
pub use runs::RunSummaryPartial;

/// Count valid slots, including floating-point NaNs.
///
/// # Panics
///
/// Panics if aggregation fails or the count exceeds the compressor's `u32` limit.
pub fn valid_count(data: &ArrayInput, ctx: &mut ExecutionCtx) -> u32 {
    let result = data
        .compute_result(&Count.bind(NumericalAggregateOpts::include_nans()), ctx)
        .vortex_expect("valid count aggregation succeeds for canonical compressor inputs");
    u32::try_from(
        result
            .as_primitive()
            .typed_value::<u64>()
            .vortex_expect("count is non-null"),
    )
    .vortex_expect("compressor value counts fit in u32")
}

/// Count null slots.
///
/// # Panics
///
/// Panics if aggregation fails or the count exceeds the compressor's `u32` limit.
pub fn null_count(data: &ArrayInput, ctx: &mut ExecutionCtx) -> u32 {
    let result = data
        .compute_result(&NullCount.bind(EmptyOptions), ctx)
        .vortex_expect("null count aggregation succeeds for canonical compressor inputs");
    u32::try_from(
        result
            .as_primitive()
            .typed_value::<u64>()
            .vortex_expect("count is non-null"),
    )
    .vortex_expect("compressor null counts fit in u32")
}

/// Retain the native integer frequencies used by dictionary and sparse encoders.
///
/// # Panics
///
/// Panics if the input is not an integer array or aggregation fails.
pub fn integer_frequencies(
    data: &ArrayInput,
    ctx: &mut ExecutionCtx,
) -> Arc<IntegerFrequenciesPartial> {
    data.compute_partial(&AggregateFn::new(IntegerFrequencies, EmptyOptions), ctx)
        .vortex_expect("integer frequencies succeed for canonical integer compressor inputs")
}

/// Retain bitwise distinct float values used by dictionary encoders.
///
/// # Panics
///
/// Panics if the input is not a float array or aggregation fails.
pub fn float_distinct(data: &ArrayInput, ctx: &mut ExecutionCtx) -> Arc<FloatDistinctPartial> {
    data.compute_partial(&AggregateFn::new(FloatDistinct, EmptyOptions), ctx)
        .vortex_expect("float distinct aggregation succeeds for canonical float compressor inputs")
}

/// Return the current compressor's floored valid-value average run length.
///
/// Float summaries use numeric equality. The compatibility projection adds the legacy first-NaN
/// comparison once, after all partials have been merged.
///
/// # Panics
///
/// Panics if the input is not primitive, aggregation fails, or the result exceeds `u32`.
pub fn average_run_length(
    data: &ArrayInput,
    compress_ctx: &CompressorContext,
    ctx: &mut ExecutionCtx,
) -> u32 {
    if data.array().dtype().is_int()
        && compress_ctx.requests_aggregate(&IntegerFrequencies.bind(EmptyOptions))
    {
        return integer_frequencies(data, ctx)
            .run_summary()
            .average_run_length_for_compression()
            .vortex_expect("compressor average run lengths fit in u32");
    }
    if data.array().dtype().is_float()
        && compress_ctx.requests_aggregate(&FloatDistinct.bind(EmptyOptions))
    {
        return float_distinct(data, ctx)
            .run_summary()
            .average_run_length_for_compression()
            .vortex_expect("compressor average run lengths fit in u32");
    }
    data.compute_partial(&AggregateFn::new(RunSummary, EmptyOptions), ctx)
        .vortex_expect("run summaries succeed for canonical primitive compressor inputs")
        .average_run_length_for_compression()
        .vortex_expect("compressor average run lengths fit in u32")
}

/// Return the physical view-prefix cardinality used by string dictionary estimates.
///
/// Includes invalid slots to preserve the existing compressor estimate.
///
/// # Panics
///
/// Panics if the input is not VarBinView or aggregation fails.
pub(crate) fn view_prefix_distinct(data: &ArrayInput, ctx: &mut ExecutionCtx) -> u32 {
    data.compute_result(&VarBinViewPrefixDistinct.bind(EmptyOptions), ctx)
        .vortex_expect("prefix aggregation succeeds for canonical VarBinView compressor inputs")
        .as_primitive()
        .typed_value::<u32>()
        .vortex_expect("prefix cardinality is non-null")
}

/// Integer extrema with the width arithmetic used by compression estimates.
///
/// This is a value view over MinMax, with no additional scan or cache.
pub struct IntegerRange {
    /// Non-null typed extrema.
    bounds: MinMaxResult,
}

impl IntegerRange {
    /// Whether the minimum is zero.
    pub fn min_is_zero(&self) -> bool {
        self.values().0 == 0
    }

    /// Whether the minimum is negative.
    pub fn min_is_negative(&self) -> bool {
        self.values().0 < 0
    }

    /// The exact unsigned span, including the full signed range.
    pub fn max_minus_min(&self) -> u64 {
        let (min, max) = self.values();
        u64::try_from(max - min).vortex_expect("integer span fits in u64")
    }

    /// The maximum's unsigned bit-pattern logarithm, or None for zero.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "Use the low native-width bits of a signed maximum"
    )]
    pub fn max_ilog2(&self) -> Option<u32> {
        let width = self.bounds.max.dtype().as_ptype().bit_width();
        let bits = (self.values().1 as u64) & (u64::MAX >> (64 - width));
        bits.checked_ilog2()
    }

    /// Widen both extrema without losing unsigned values.
    fn values(&self) -> (i128, i128) {
        let bounds = &self.bounds;
        match_each_integer_ptype!(bounds.min.dtype().as_ptype(), |T| {
            (
                bounds
                    .min
                    .as_primitive()
                    .typed_value::<T>()
                    .vortex_expect("integer minimum is non-null") as i128,
                bounds
                    .max
                    .as_primitive()
                    .typed_value::<T>()
                    .vortex_expect("integer maximum is non-null") as i128,
            )
        })
    }
}

/// Read nonempty integer extrema through the aggregate cache.
///
/// # Panics
///
/// Panics if the array has no valid values, is not integer, or aggregation fails.
pub fn integer_range(data: &ArrayInput, ctx: &mut ExecutionCtx) -> IntegerRange {
    let result = data
        .compute_result(&MinMax.bind(NumericalAggregateOpts::skip_nans()), ctx)
        .vortex_expect("min/max aggregation succeeds for canonical integer compressor inputs");
    IntegerRange {
        bounds: MinMaxResult::from_scalar(result)
            .vortex_expect("MinMax result has its declared struct dtype")
            .vortex_expect("compressor handles empty and all-null arrays before integer estimates"),
    }
}

#[cfg(test)]
mod run_tests;

#[cfg(test)]
mod distribution_tests;

#[cfg(test)]
mod tests {
    use vortex_array::ArrayInput;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;

    use super::integer_range;

    #[test]
    fn full_integer_spans_and_native_signed_bits() {
        let mut ctx = vortex_array::array_session().create_execution_ctx();
        let signed = ArrayInput::new(PrimitiveArray::from_iter([i64::MIN, i64::MAX]).into_array());
        let range = integer_range(&signed, &mut ctx);
        assert_eq!(range.max_minus_min(), u64::MAX);
        assert_eq!(range.max_ilog2(), Some(62));
        assert!(range.min_is_negative());

        let unsigned = ArrayInput::new(PrimitiveArray::from_iter([0u64, u64::MAX]).into_array());
        let range = integer_range(&unsigned, &mut ctx);
        assert_eq!(range.max_minus_min(), u64::MAX);
        assert_eq!(range.max_ilog2(), Some(63));
        assert!(range.min_is_zero());

        let narrow = ArrayInput::new(PrimitiveArray::from_iter([i8::MIN, -1]).into_array());
        let range = integer_range(&narrow, &mut ctx);
        assert_eq!(range.max_minus_min(), 127);
        assert_eq!(range.max_ilog2(), Some(7));
    }
}
