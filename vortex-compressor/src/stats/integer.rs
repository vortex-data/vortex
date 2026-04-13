// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Integer compression statistics.

use std::hash::Hash;

use num_traits::PrimInt;
use rustc_hash::FxBuildHasher;
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
use vortex_utils::aliases::hash_map::HashMap;

use super::GenerateStatsOptions;

/// Information about the distinct values in an integer array.
#[derive(Debug, Clone)]
pub struct DistinctInfo<T> {
    /// The unique values and their occurrences.
    distinct_values: HashMap<NativeValue<T>, u32, FxBuildHasher>,
    /// The count of unique values. This _must_ be non-zero.
    distinct_count: u32,
    /// The most frequent value.
    most_frequent_value: T,
    /// The number of times the most frequent value occurs.
    top_frequency: u32,
}

impl<T> DistinctInfo<T> {
    /// Returns a reference to the distinct values map.
    pub fn distinct_values(&self) -> &HashMap<NativeValue<T>, u32, FxBuildHasher> {
        &self.distinct_values
    }
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
    ) -> VortexResult<Self> {
        match_each_integer_ptype!(input.ptype(), |T| {
            typed_int_stats::<T>(input, opts.count_distinct_values)
        })
    }

    /// Get the count of distinct values, if we have computed it already.
    pub fn distinct_count(&self) -> Option<u32> {
        self.erased.distinct_count()
    }

    /// Get the most commonly occurring value and its count, if we have computed it already.
    pub fn most_frequent_value_and_count(&self) -> Option<(PValue, u32)> {
        self.erased.most_frequent_value_and_count()
    }
}

impl IntegerStats {
    /// Generates stats with default options.
    pub fn generate(input: &PrimitiveArray) -> Self {
        Self::generate_opts(input, GenerateStatsOptions::default())
    }

    /// Generates stats with provided options.
    pub fn generate_opts(input: &PrimitiveArray, opts: GenerateStatsOptions) -> Self {
        Self::generate_opts_fallible(input, opts)
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

    if array.all_invalid()? {
        return Ok(IntegerStats {
            null_count: u32::try_from(array.len())?,
            value_count: 0,
            average_run_length: 0,
            erased: TypedStats {
                min: T::max_value(),
                max: T::min_value(),
                distinct: Some(DistinctInfo {
                    distinct_values: HashMap::with_capacity_and_hasher(0, FxBuildHasher),
                    distinct_count: 0,
                    most_frequent_value: T::zero(),
                    top_frequency: 0,
                }),
            }
            .into(),
        });
    }

    let validity = array.validity_mask()?;
    let null_count = validity.false_count();
    let value_count = validity.true_count();

    // Initialize loop state.
    let head_idx = validity
        .first()
        .vortex_expect("All null masks have been handled before");
    let buffer = array.to_buffer::<T>();
    let head = buffer[head_idx];

    let mut loop_state = LoopState {
        distinct_values: if count_distinct_values {
            HashMap::with_capacity_and_hasher(array.len() / 2, FxBuildHasher)
        } else {
            HashMap::with_hasher(FxBuildHasher)
        },
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
        .compute_as::<T>(Stat::Min)
        .vortex_expect("min should be computed");

    let max = array_ref
        .statistics()
        .compute_as::<T>(Stat::Max)
        .vortex_expect("max should be computed");

    let distinct = count_distinct_values.then(|| {
        let (&top_value, &top_count) = loop_state
            .distinct_values
            .iter()
            .max_by_key(|&(_, &count)| count)
            .vortex_expect("we know this is non-empty");

        DistinctInfo {
            distinct_count: u32::try_from(loop_state.distinct_values.len())
                .vortex_expect("there are more than `u32::MAX` distinct values"),
            most_frequent_value: top_value.0,
            top_frequency: top_count,
            distinct_values: loop_state.distinct_values,
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
struct LoopState<T> {
    /// The previous value seen.
    prev: T,
    /// The run count.
    runs: u32,
    /// The distinct values map.
    distinct_values: HashMap<NativeValue<T>, u32, FxBuildHasher>,
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
            *state.distinct_values.entry(NativeValue(value)).or_insert(0) += 1;
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
                *state.distinct_values.entry(NativeValue(value)).or_insert(0) += 1;
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
                *state.distinct_values.entry(NativeValue(value)).or_insert(0) += 1;
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

    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::validity::Validity;
    use vortex_buffer::BitBuffer;
    use vortex_buffer::Buffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use super::IntegerStats;
    use super::typed_int_stats;

    #[test]
    fn test_naive_count_distinct_values() -> VortexResult<()> {
        let array = PrimitiveArray::new(buffer![217u8, 0], Validity::NonNullable);
        let stats = typed_int_stats::<u8>(&array, true)?;
        assert_eq!(stats.distinct_count().unwrap(), 2);
        Ok(())
    }

    #[test]
    fn test_naive_count_distinct_values_nullable() -> VortexResult<()> {
        let array = PrimitiveArray::new(
            buffer![217u8, 0],
            Validity::from(BitBuffer::from(vec![true, false])),
        );
        let stats = typed_int_stats::<u8>(&array, true)?;
        assert_eq!(stats.distinct_count().unwrap(), 1);
        Ok(())
    }

    #[test]
    fn test_count_distinct_values() -> VortexResult<()> {
        let array = PrimitiveArray::new((0..128u8).collect::<Buffer<u8>>(), Validity::NonNullable);
        let stats = typed_int_stats::<u8>(&array, true)?;
        assert_eq!(stats.distinct_count().unwrap(), 128);
        Ok(())
    }

    #[test]
    fn test_count_distinct_values_nullable() -> VortexResult<()> {
        let array = PrimitiveArray::new(
            (0..128u8).collect::<Buffer<u8>>(),
            Validity::from(BitBuffer::from_iter(
                iter::repeat_n(vec![true, false], 64).flatten(),
            )),
        );
        let stats = typed_int_stats::<u8>(&array, true)?;
        assert_eq!(stats.distinct_count().unwrap(), 64);
        Ok(())
    }

    #[test]
    fn test_integer_stats_leading_nulls() {
        let ints = PrimitiveArray::new(buffer![0, 1, 2], Validity::from_iter([false, true, true]));

        let stats = IntegerStats::generate_opts(
            &ints,
            crate::stats::GenerateStatsOptions {
                count_distinct_values: true,
            },
        );

        assert_eq!(stats.value_count, 2);
        assert_eq!(stats.null_count, 1);
        assert_eq!(stats.average_run_length, 1);
        assert_eq!(stats.distinct_count().unwrap(), 2);
    }
}
