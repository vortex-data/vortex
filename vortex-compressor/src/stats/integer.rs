// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Integer compression statistics.

use std::hash::Hash;

use num_traits::PrimInt;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::primitive::NativeValue;
use vortex_array::dtype::IntegerPType;
use vortex_array::expr::stats::Stat;
use vortex_array::match_each_integer_ptype;
use vortex_array::scalar::PValue;
use vortex_array::scalar::Scalar;
use vortex_buffer::BitBuffer;
use vortex_error::VortexError;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::AllOr;

use super::GenerateStatsOptions;
use super::cardinality::CardinalityEstimator;

/// Expected relative error for the default cardinality estimator precision.
///
/// Cloudflare's default `P=12` HLL++ parameters document this as `1.04 / sqrt(2^12)`.
const DISTINCT_COUNT_ERROR_NUMERATOR: usize = 65;
/// Denominator for the default cardinality estimator expected relative error.
const DISTINCT_COUNT_ERROR_DENOMINATOR: usize = 4_000;

/// Information about the distinct values in an integer array.
///
/// The `distinct_count` is an estimate computed using Cloudflare's cardinality estimator, which
/// yields exact counts for small cardinalities (<= 128 for the default parameters) and a
/// HyperLogLog++ approximation for larger cardinalities. The most frequent value is tracked using
/// the Boyer-Moore majority candidate algorithm, so `most_frequent_value` and `top_frequency` are
/// only guaranteed to reflect the true majority element when some value accounts for more than
/// half of the non-null entries; otherwise they are treated as a best-effort estimate.
#[derive(Debug, Clone)]
pub struct DistinctInfo<T> {
    /// The estimated count of unique values. This _must_ be non-zero.
    distinct_count: u32,
    /// The most frequent value (Boyer-Moore majority candidate).
    most_frequent_value: T,
    /// The exact number of times `most_frequent_value` occurs in the array.
    top_frequency: u32,
}

/// Typed statistics for a specific integer type.
#[derive(Debug, Clone)]
pub struct TypedStats<T> {
    /// The minimum value.
    min: T,
    /// The maximum value.
    max: T,
    /// Distinct value information, or `None` if not computed.
    distinct: Option<DistinctInfo<T>>,
}

impl<T> TypedStats<T> {
    /// Returns the distinct value information, if computed.
    pub fn distinct(&self) -> Option<&DistinctInfo<T>> {
        self.distinct.as_ref()
    }
}

impl<T> TypedStats<T> {
    /// Get the count of distinct values, if we have computed it already.
    fn distinct_count(&self) -> Option<u32> {
        Some(self.distinct.as_ref()?.distinct_count)
    }

    /// Get the most commonly occurring value and its count, if we have computed it already.
    fn most_frequent_value_and_count(&self) -> Option<(&T, u32)> {
        let distinct = self.distinct.as_ref()?;
        Some((&distinct.most_frequent_value, distinct.top_frequency))
    }
}

/// Type-erased container for one of the [`TypedStats`] variants.
///
/// Building the `TypedStats` is considerably faster and cheaper than building a type-erased
/// set of stats. We then perform a variety of access methods on them.
#[derive(Clone, Debug)]
pub enum ErasedStats {
    /// Stats for `u8` arrays.
    U8(TypedStats<u8>),
    /// Stats for `u16` arrays.
    U16(TypedStats<u16>),
    /// Stats for `u32` arrays.
    U32(TypedStats<u32>),
    /// Stats for `u64` arrays.
    U64(TypedStats<u64>),
    /// Stats for `i8` arrays.
    I8(TypedStats<i8>),
    /// Stats for `i16` arrays.
    I16(TypedStats<i16>),
    /// Stats for `i32` arrays.
    I32(TypedStats<i32>),
    /// Stats for `i64` arrays.
    I64(TypedStats<i64>),
}

impl ErasedStats {
    /// Returns `true` if the minimum value is zero.
    pub fn min_is_zero(&self) -> bool {
        match &self {
            ErasedStats::U8(x) => x.min == 0,
            ErasedStats::U16(x) => x.min == 0,
            ErasedStats::U32(x) => x.min == 0,
            ErasedStats::U64(x) => x.min == 0,
            ErasedStats::I8(x) => x.min == 0,
            ErasedStats::I16(x) => x.min == 0,
            ErasedStats::I32(x) => x.min == 0,
            ErasedStats::I64(x) => x.min == 0,
        }
    }

    /// Returns `true` if the minimum value is negative.
    pub fn min_is_negative(&self) -> bool {
        match &self {
            ErasedStats::U8(_)
            | ErasedStats::U16(_)
            | ErasedStats::U32(_)
            | ErasedStats::U64(_) => false,
            ErasedStats::I8(x) => x.min < 0,
            ErasedStats::I16(x) => x.min < 0,
            ErasedStats::I32(x) => x.min < 0,
            ErasedStats::I64(x) => x.min < 0,
        }
    }

    /// Difference between max and min.
    pub fn max_minus_min(&self) -> u64 {
        match &self {
            ErasedStats::U8(x) => (x.max - x.min) as u64,
            ErasedStats::U16(x) => (x.max - x.min) as u64,
            ErasedStats::U32(x) => (x.max - x.min) as u64,
            ErasedStats::U64(x) => x.max - x.min,
            ErasedStats::I8(x) => (x.max as i16 - x.min as i16) as u64,
            ErasedStats::I16(x) => (x.max as i32 - x.min as i32) as u64,
            ErasedStats::I32(x) => (x.max as i64 - x.min as i64) as u64,
            ErasedStats::I64(x) => u64::try_from(x.max as i128 - x.min as i128)
                .vortex_expect("max minus min result bigger than u64"),
        }
    }

    /// Returns the ilog2 of the max value when transmuted to unsigned, or `None` if zero.
    ///
    /// This matches how BitPacking computes bit width: it reinterprets signed values as
    /// unsigned (preserving bit pattern) and uses `leading_zeros`. For non-negative signed
    /// values, the transmuted value equals the original value.
    ///
    /// This is used to determine if FOR encoding would reduce bit width compared to
    /// direct BitPacking. If `max_ilog2() == max_minus_min_ilog2()`, FOR doesn't help.
    pub fn max_ilog2(&self) -> Option<u32> {
        match &self {
            ErasedStats::U8(x) => x.max.checked_ilog2(),
            ErasedStats::U16(x) => x.max.checked_ilog2(),
            ErasedStats::U32(x) => x.max.checked_ilog2(),
            ErasedStats::U64(x) => x.max.checked_ilog2(),
            // Transmute signed to unsigned (bit pattern preserved) to match BitPacking behavior.
            ErasedStats::I8(x) => (x.max as u8).checked_ilog2(),
            ErasedStats::I16(x) => (x.max as u16).checked_ilog2(),
            ErasedStats::I32(x) => (x.max as u32).checked_ilog2(),
            ErasedStats::I64(x) => (x.max as u64).checked_ilog2(),
        }
    }

    /// Get the count of distinct values, if we have computed it already.
    pub fn distinct_count(&self) -> Option<u32> {
        match &self {
            ErasedStats::U8(x) => x.distinct_count(),
            ErasedStats::U16(x) => x.distinct_count(),
            ErasedStats::U32(x) => x.distinct_count(),
            ErasedStats::U64(x) => x.distinct_count(),
            ErasedStats::I8(x) => x.distinct_count(),
            ErasedStats::I16(x) => x.distinct_count(),
            ErasedStats::I32(x) => x.distinct_count(),
            ErasedStats::I64(x) => x.distinct_count(),
        }
    }

    /// Get the most commonly occurring value and its count.
    pub fn most_frequent_value_and_count(&self) -> Option<(PValue, u32)> {
        match &self {
            ErasedStats::U8(x) => {
                let (top_value, count) = x.most_frequent_value_and_count()?;
                Some(((*top_value).into(), count))
            }
            ErasedStats::U16(x) => {
                let (top_value, count) = x.most_frequent_value_and_count()?;
                Some(((*top_value).into(), count))
            }
            ErasedStats::U32(x) => {
                let (top_value, count) = x.most_frequent_value_and_count()?;
                Some(((*top_value).into(), count))
            }
            ErasedStats::U64(x) => {
                let (top_value, count) = x.most_frequent_value_and_count()?;
                Some(((*top_value).into(), count))
            }
            ErasedStats::I8(x) => {
                let (top_value, count) = x.most_frequent_value_and_count()?;
                Some(((*top_value).into(), count))
            }
            ErasedStats::I16(x) => {
                let (top_value, count) = x.most_frequent_value_and_count()?;
                Some(((*top_value).into(), count))
            }
            ErasedStats::I32(x) => {
                let (top_value, count) = x.most_frequent_value_and_count()?;
                Some(((*top_value).into(), count))
            }
            ErasedStats::I64(x) => {
                let (top_value, count) = x.most_frequent_value_and_count()?;
                Some(((*top_value).into(), count))
            }
        }
    }
}

/// Implements `From<TypedStats<$T>>` for [`ErasedStats`].
macro_rules! impl_from_typed {
    ($T:ty, $variant:path) => {
        impl From<TypedStats<$T>> for ErasedStats {
            fn from(typed: TypedStats<$T>) -> Self {
                $variant(typed)
            }
        }
    };
}

impl_from_typed!(u8, ErasedStats::U8);
impl_from_typed!(u16, ErasedStats::U16);
impl_from_typed!(u32, ErasedStats::U32);
impl_from_typed!(u64, ErasedStats::U64);
impl_from_typed!(i8, ErasedStats::I8);
impl_from_typed!(i16, ErasedStats::I16);
impl_from_typed!(i32, ErasedStats::I32);
impl_from_typed!(i64, ErasedStats::I64);

/// Array of integers and relevant stats for compression.
#[derive(Clone, Debug)]
pub struct IntegerStats {
    /// Cache for `validity.false_count()`.
    null_count: u32,
    /// Cache for `validity.true_count()`.
    value_count: u32,
    /// The average run length.
    average_run_length: u32,
    /// Type-erased typed statistics.
    erased: ErasedStats,
}

impl IntegerStats {
    /// Generates stats, returning an error on failure.
    fn generate_opts_fallible(
        input: &PrimitiveArray,
        opts: GenerateStatsOptions,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Self> {
        match_each_integer_ptype!(input.ptype(), |T| {
            typed_int_stats::<T>(input, opts.count_distinct_values, ctx)
        })
    }

    /// Get the count of distinct values, if we have computed it already.
    pub fn distinct_count(&self) -> Option<u32> {
        self.erased.distinct_count()
    }

    /// Returns true if the estimated distinct count could equal `count` within the estimator's
    /// expected error bound.
    pub fn estimated_distinct_count_could_equal(&self, count: usize) -> bool {
        let Some(distinct_count) = self.distinct_count() else {
            return true;
        };

        let error_bound = distinct_count_error_bound(count);
        (distinct_count as usize).abs_diff(count) <= error_bound
    }

    /// Get the most commonly occurring value and its count, if we have computed it already.
    pub fn most_frequent_value_and_count(&self) -> Option<(PValue, u32)> {
        self.erased.most_frequent_value_and_count()
    }
}

/// Returns the absolute error bound for an expected distinct count.
fn distinct_count_error_bound(count: usize) -> usize {
    count
        .saturating_mul(DISTINCT_COUNT_ERROR_NUMERATOR)
        .div_ceil(DISTINCT_COUNT_ERROR_DENOMINATOR)
}

impl IntegerStats {
    /// Generates stats with default options.
    pub fn generate(input: &PrimitiveArray, ctx: &mut ExecutionCtx) -> Self {
        Self::generate_opts(input, GenerateStatsOptions::default(), ctx)
    }

    /// Generates stats with provided options.
    pub fn generate_opts(
        input: &PrimitiveArray,
        opts: GenerateStatsOptions,
        ctx: &mut ExecutionCtx,
    ) -> Self {
        Self::generate_opts_fallible(input, opts, ctx)
            .vortex_expect("IntegerStats::generate_opts should not fail")
    }

    /// Returns the number of null values.
    pub fn null_count(&self) -> u32 {
        self.null_count
    }

    /// Returns the number of non-null values.
    pub fn value_count(&self) -> u32 {
        self.value_count
    }

    /// Returns the average run length.
    pub fn average_run_length(&self) -> u32 {
        self.average_run_length
    }

    /// Returns the type-erased typed statistics.
    pub fn erased(&self) -> &ErasedStats {
        &self.erased
    }
}

/// Computes typed integer statistics for a specific integer type.
fn typed_int_stats<T>(
    array: &PrimitiveArray,
    count_distinct_values: bool,
    ctx: &mut ExecutionCtx,
) -> VortexResult<IntegerStats>
where
    T: IntegerPType + PrimInt + for<'a> TryFrom<&'a Scalar, Error = VortexError>,
    TypedStats<T>: Into<ErasedStats>,
    NativeValue<T>: Eq + Hash,
{
    // Special case: empty array.
    if array.is_empty() {
        return Ok(IntegerStats {
            null_count: 0,
            value_count: 0,
            average_run_length: 0,
            erased: TypedStats {
                min: T::max_value(),
                max: T::min_value(),
                distinct: None,
            }
            .into(),
        });
    }

    if array.all_invalid(ctx)? {
        return Ok(IntegerStats {
            null_count: u32::try_from(array.len())?,
            value_count: 0,
            average_run_length: 0,
            erased: TypedStats {
                min: T::max_value(),
                max: T::min_value(),
                distinct: Some(DistinctInfo {
                    distinct_count: 0,
                    most_frequent_value: T::zero(),
                    top_frequency: 0,
                }),
            }
            .into(),
        });
    }

    let validity = array
        .as_ref()
        .validity()?
        .execute_mask(array.as_ref().len(), ctx)?;
    let null_count = validity.false_count();
    let value_count = validity.true_count();

    // Initialize loop state.
    let head_idx = validity
        .first()
        .vortex_expect("All null masks have been handled before");
    let buffer = array.to_buffer::<T>();
    let head = buffer[head_idx];

    let mut loop_state = LoopState::<T> {
        estimator: CardinalityEstimator::new(),
        bm_candidate: head,
        bm_votes: 0,
        prev: head,
        runs: 1,
    };

    let sliced = buffer.slice(head_idx..array.len());
    let mut chunks = sliced.as_slice().chunks_exact(64);
    match validity.bit_buffer() {
        AllOr::All => {
            for chunk in &mut chunks {
                inner_loop_nonnull(
                    chunk.try_into().ok().vortex_expect("chunk size must be 64"),
                    count_distinct_values,
                    &mut loop_state,
                )
            }
            let remainder = chunks.remainder();
            inner_loop_naive(
                remainder,
                count_distinct_values,
                &BitBuffer::new_set(remainder.len()),
                &mut loop_state,
            );
        }
        AllOr::None => unreachable!("All invalid arrays have been handled before"),
        AllOr::Some(v) => {
            let mask = v.slice(head_idx..array.len());
            let mut offset = 0;
            for chunk in &mut chunks {
                let validity = mask.slice(offset..(offset + 64));
                offset += 64;

                match validity.true_count() {
                    // All nulls -> no stats to update.
                    0 => continue,
                    // Inner loop for when validity check can be elided.
                    64 => inner_loop_nonnull(
                        chunk.try_into().ok().vortex_expect("chunk size must be 64"),
                        count_distinct_values,
                        &mut loop_state,
                    ),
                    // Inner loop for when we need to check validity.
                    _ => inner_loop_nullable(
                        chunk.try_into().ok().vortex_expect("chunk size must be 64"),
                        count_distinct_values,
                        &validity,
                        &mut loop_state,
                    ),
                }
            }
            // Final iteration, run naive loop.
            let remainder = chunks.remainder();
            inner_loop_naive(
                remainder,
                count_distinct_values,
                &mask.slice(offset..(offset + remainder.len())),
                &mut loop_state,
            );
        }
    }

    let runs = loop_state.runs;

    let array_ref = array.as_ref();
    let min = array_ref
        .statistics()
        .compute_as::<T>(Stat::Min, ctx)
        .vortex_expect("min should be computed");

    let max = array_ref
        .statistics()
        .compute_as::<T>(Stat::Max, ctx)
        .vortex_expect("max should be computed");

    let distinct = count_distinct_values.then(|| {
        // The cardinality estimator is exact for small cardinalities and approximate beyond.
        // We clamp to at least 1 because we are inside the non-empty/non-all-null branch.
        let distinct_count = u32::try_from(loop_state.estimator.estimate())
            .vortex_expect("there are more than `u32::MAX` distinct values")
            .max(1);

        // Count the Boyer-Moore majority candidate exactly via a second pass. If any value
        // accounts for more than half of the non-null entries, this counts that value; otherwise
        // the returned count is a best-effort estimate for whichever candidate survived.
        let top_frequency = count_occurrences::<T>(
            buffer.as_slice(),
            validity.bit_buffer(),
            loop_state.bm_candidate,
        );

        DistinctInfo {
            distinct_count,
            most_frequent_value: loop_state.bm_candidate,
            top_frequency,
        }
    });

    let typed = TypedStats { min, max, distinct };

    let null_count = u32::try_from(null_count)?;
    let value_count = u32::try_from(value_count)?;

    Ok(IntegerStats {
        null_count,
        value_count,
        average_run_length: value_count / runs,
        erased: typed.into(),
    })
}

/// Internal loop state for integer stats computation.
struct LoopState<T>
where
    T: IntegerPType,
    NativeValue<T>: Eq + Hash,
{
    /// The previous value seen.
    prev: T,
    /// The run count.
    runs: u32,
    /// Cloudflare's cardinality estimator, used to approximate the number of distinct values
    /// without materializing an exact hash map.
    estimator: CardinalityEstimator<NativeValue<T>>,
    /// Boyer-Moore majority candidate; holds the current candidate for the most frequent value.
    bm_candidate: T,
    /// Boyer-Moore vote counter for `bm_candidate`.
    bm_votes: u32,
}

/// Updates the Boyer-Moore majority-vote state for a single value.
#[inline(always)]
fn boyer_moore_observe<T>(state: &mut LoopState<T>, value: T)
where
    T: IntegerPType,
    NativeValue<T>: Eq + Hash,
{
    if state.bm_votes == 0 {
        state.bm_candidate = value;
        state.bm_votes = 1;
    } else if value == state.bm_candidate {
        state.bm_votes += 1;
    } else {
        state.bm_votes -= 1;
    }
}

/// Counts exact occurrences of `needle` in `buffer`, restricted to valid positions according to
/// `validity`.
fn count_occurrences<T: IntegerPType>(buffer: &[T], validity: AllOr<&BitBuffer>, needle: T) -> u32 {
    let count = match validity {
        AllOr::All => buffer.iter().filter(|&&v| v == needle).count(),
        AllOr::None => 0,
        AllOr::Some(mask) => buffer
            .iter()
            .enumerate()
            .filter(|&(idx, &v)| mask.value(idx) && v == needle)
            .count(),
    };
    u32::try_from(count).vortex_expect("occurrences cannot exceed `u32::MAX`")
}

/// Inner loop for non-null chunks of 64 values.
#[inline(always)]
fn inner_loop_nonnull<T: IntegerPType>(
    values: &[T; 64],
    count_distinct_values: bool,
    state: &mut LoopState<T>,
) where
    NativeValue<T>: Eq + Hash,
{
    for &value in values {
        if count_distinct_values {
            state.estimator.insert(&NativeValue(value));
            boyer_moore_observe(state, value);
        }

        if value != state.prev {
            state.prev = value;
            state.runs += 1;
        }
    }
}

/// Inner loop for nullable chunks of 64 values.
#[inline(always)]
fn inner_loop_nullable<T: IntegerPType>(
    values: &[T; 64],
    count_distinct_values: bool,
    is_valid: &BitBuffer,
    state: &mut LoopState<T>,
) where
    NativeValue<T>: Eq + Hash,
{
    for (idx, &value) in values.iter().enumerate() {
        if is_valid.value(idx) {
            if count_distinct_values {
                state.estimator.insert(&NativeValue(value));
                boyer_moore_observe(state, value);
            }

            if value != state.prev {
                state.prev = value;
                state.runs += 1;
            }
        }
    }
}

/// Fallback inner loop for remainder values.
#[inline(always)]
fn inner_loop_naive<T: IntegerPType>(
    values: &[T],
    count_distinct_values: bool,
    is_valid: &BitBuffer,
    state: &mut LoopState<T>,
) where
    NativeValue<T>: Eq + Hash,
{
    for (idx, &value) in values.iter().enumerate() {
        if is_valid.value(idx) {
            if count_distinct_values {
                state.estimator.insert(&NativeValue(value));
                boyer_moore_observe(state, value);
            }

            if value != state.prev {
                state.prev = value;
                state.runs += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::iter;
    use std::sync::LazyLock;

    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::session::ArraySession;
    use vortex_array::validity::Validity;
    use vortex_buffer::BitBuffer;
    use vortex_buffer::Buffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::IntegerStats;
    use super::typed_int_stats;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(array_session);

    #[test]
    fn test_naive_count_distinct_values() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let array = PrimitiveArray::new(buffer![217u8, 0], Validity::NonNullable);
        let stats = typed_int_stats::<u8>(&array, true, &mut ctx)?;
        assert_eq!(stats.distinct_count().unwrap(), 2);
        Ok(())
    }

    #[test]
    fn test_naive_count_distinct_values_nullable() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let array = PrimitiveArray::new(
            buffer![217u8, 0],
            Validity::from(BitBuffer::from(vec![true, false])),
        );
        let stats = typed_int_stats::<u8>(&array, true, &mut ctx)?;
        assert_eq!(stats.distinct_count().unwrap(), 1);
        Ok(())
    }

    #[test]
    fn test_count_distinct_values() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let array = PrimitiveArray::new((0..128u8).collect::<Buffer<u8>>(), Validity::NonNullable);
        let stats = typed_int_stats::<u8>(&array, true, &mut ctx)?;
        assert_eq!(stats.distinct_count().unwrap(), 128);
        Ok(())
    }

    #[test]
    fn test_count_distinct_values_nullable() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let array = PrimitiveArray::new(
            (0..128u8).collect::<Buffer<u8>>(),
            Validity::from(BitBuffer::from_iter(
                iter::repeat_n(vec![true, false], 64).flatten(),
            )),
        );
        let stats = typed_int_stats::<u8>(&array, true, &mut ctx)?;
        assert_eq!(stats.distinct_count().unwrap(), 64);
        Ok(())
    }

    #[test]
    fn test_integer_stats_leading_nulls() {
        let mut ctx = SESSION.create_execution_ctx();
        let ints = PrimitiveArray::new(buffer![0, 1, 2], Validity::from_iter([false, true, true]));

        let stats = IntegerStats::generate_opts(
            &ints,
            crate::stats::GenerateStatsOptions {
                count_distinct_values: true,
            },
            &mut ctx,
        );

        assert_eq!(stats.value_count, 2);
        assert_eq!(stats.null_count, 1);
        assert_eq!(stats.average_run_length, 1);
        assert_eq!(stats.distinct_count().unwrap(), 2);
    }

    #[test]
    fn test_most_frequent_value_dominates() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        // A value that appears in 95% of the array must be recovered exactly by the
        // Boyer-Moore tracking plus second-pass count.
        let top = -1i32;
        let mut data: Vec<i32> = vec![top; 950];
        data.extend(0..50i32);
        let array = PrimitiveArray::new(Buffer::copy_from(&data), Validity::NonNullable);
        let stats = typed_int_stats::<i32>(&array, true, &mut ctx)?;
        let (top_value, top_count) = stats
            .erased()
            .most_frequent_value_and_count()
            .expect("distinct info must be present");
        assert_eq!(top_value, top.into());
        assert_eq!(top_count, 950);
        Ok(())
    }

    #[test]
    fn test_cardinality_estimate_large_unique() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        // For 1024 distinct values the estimator falls back to HyperLogLog++; verify the
        // estimate is within the expected error bound (~1.6% for the default P/W).
        let array =
            PrimitiveArray::new((0..1024u32).collect::<Buffer<u32>>(), Validity::NonNullable);
        let stats = typed_int_stats::<u32>(&array, true, &mut ctx)?;
        let estimated = stats.distinct_count().unwrap();
        let error_ratio = (estimated as f64 - 1024.0).abs() / 1024.0;
        assert!(
            error_ratio < 0.05,
            "estimator error {error_ratio} exceeds 5% for 1024 distinct values"
        );
        Ok(())
    }

    #[test]
    fn test_estimated_distinct_count_could_equal() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();

        let unique =
            PrimitiveArray::new((0..1024u32).collect::<Buffer<u32>>(), Validity::NonNullable);
        let unique_stats = typed_int_stats::<u32>(&unique, true, &mut ctx)?;
        assert!(unique_stats.estimated_distinct_count_could_equal(1024));

        let low_cardinality = PrimitiveArray::new(
            (0..1024u32).map(|value| value % 8).collect::<Buffer<u32>>(),
            Validity::NonNullable,
        );
        let low_cardinality_stats = typed_int_stats::<u32>(&low_cardinality, true, &mut ctx)?;
        assert!(!low_cardinality_stats.estimated_distinct_count_could_equal(1024));

        Ok(())
    }
}
