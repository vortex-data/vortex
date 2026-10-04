// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Integer compression statistics.

use std::hash::Hash;

use num_traits::PrimInt;
use rustc_hash::FxBuildHasher;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::primitive::NativeValue;
use vortex_array::dtype::IntegerPType;
use vortex_array::expr::stats::Precision;
use vortex_array::expr::stats::Stat;
use vortex_array::expr::stats::StatsProvider;
use vortex_array::match_each_integer_ptype;
use vortex_array::scalar::PValue;
use vortex_array::scalar::Scalar;
use vortex_error::VortexError;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_utils::aliases::hash_map::HashMap;

use super::GenerateStatsOptions;
use super::accumulator::Distinct;
use super::accumulator::Fused;
use super::accumulator::MinMax;
use super::accumulator::RunCount;
use super::accumulator::accumulate;

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

impl<T: Copy> DistinctInfo<T> {
    /// Summarizes a non-empty map of distinct values to their occurrence counts.
    pub(super) fn new(distinct_values: HashMap<NativeValue<T>, u32, FxBuildHasher>) -> Self {
        let (&top_value, &top_frequency) = distinct_values
            .iter()
            .max_by_key(|&(_, &count)| count)
            .vortex_expect("distinct values are non-empty");
        Self {
            distinct_count: u32::try_from(distinct_values.len())
                .vortex_expect("there are more than `u32::MAX` distinct values"),
            most_frequent_value: top_value.0,
            top_frequency,
            distinct_values,
        }
    }

    /// Returns the number of distinct values.
    pub fn distinct_count(&self) -> u32 {
        self.distinct_count
    }

    /// Returns the most frequent value and its number of occurrences.
    pub fn most_frequent(&self) -> (T, u32) {
        (self.most_frequent_value, self.top_frequency)
    }
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

    /// Get the most commonly occurring value and its count, if we have computed it already.
    pub fn most_frequent_value_and_count(&self) -> Option<(PValue, u32)> {
        self.erased.most_frequent_value_and_count()
    }
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
///
/// Every statistic is computed by a fused [`accumulate`] pass. Counting distinct values needs the
/// value bounds up front to pick a counter, so for types wider than a byte without cached bounds it
/// takes a second pass.
fn typed_int_stats<T>(
    array: &PrimitiveArray,
    count_distinct_values: bool,
    ctx: &mut ExecutionCtx,
) -> VortexResult<IntegerStats>
where
    T: IntegerPType + PrimInt + for<'a> TryFrom<&'a Scalar, Error = VortexError>,
    TypedStats<T>: Into<ErasedStats>,
    NativeValue<T>: Eq + Hash,
    PValue: From<T>,
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

    let validity = array
        .as_ref()
        .validity()?
        .execute_mask(array.as_ref().len(), ctx)?;
    let null_count = u32::try_from(validity.false_count())?;
    let value_count = u32::try_from(validity.true_count())?;

    if value_count == 0 {
        return Ok(IntegerStats {
            null_count,
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

    let values = array.as_slice::<T>();
    let len = values.len();
    let cached = cached_min_max::<T>(array);
    let expect_valid = "validity has valid values";

    let ((min, max), runs, distinct) = match (cached, count_distinct_values) {
        (Some(bounds), false) => {
            let runs = accumulate(values, &validity, RunCount::new()).vortex_expect(expect_valid);
            (bounds, runs, None)
        }
        (Some((min, max)), true) => {
            let (distinct, runs) = accumulate(values, &validity, Distinct::new(min, max, len))
                .vortex_expect(expect_valid);
            ((min, max), runs, Some(distinct))
        }
        (None, false) => {
            // Measured: fusing min/max with the run count is as fast or faster when every value
            // is valid, and running them one after the other over each block is as fast or
            // faster with nulls.
            let (bounds, runs) = if validity.all_true() {
                accumulate(values, &validity, Fused((MinMax::new(), RunCount::new())))
            } else {
                accumulate(values, &validity, (MinMax::new(), RunCount::new()))
            }
            .vortex_expect(expect_valid);
            (bounds, runs, None)
        }
        (None, true) => match Distinct::for_full_domain(len) {
            Some(distinct) => {
                let (bounds, (distinct, runs)) =
                    accumulate(values, &validity, (MinMax::new(), distinct))
                        .vortex_expect(expect_valid);
                (bounds, runs, Some(distinct))
            }
            None => {
                let (min, max) =
                    accumulate(values, &validity, MinMax::new()).vortex_expect(expect_valid);
                let (distinct, runs) = accumulate(values, &validity, Distinct::new(min, max, len))
                    .vortex_expect(expect_valid);
                ((min, max), runs, Some(distinct))
            }
        },
    };

    if cached.is_none() {
        let stats = array.as_ref().statistics();
        stats.set(Stat::Min, Precision::Exact(PValue::from(min).into()));
        stats.set(Stat::Max, Precision::Exact(PValue::from(max).into()));
    }

    Ok(IntegerStats {
        null_count,
        value_count,
        average_run_length: value_count / runs,
        erased: TypedStats { min, max, distinct }.into(),
    })
}

/// Returns the exact min and max already cached on the array, if both are.
fn cached_min_max<T>(array: &PrimitiveArray) -> Option<(T, T)>
where
    T: for<'a> TryFrom<&'a Scalar, Error = VortexError>,
{
    let stats = array.as_ref().statistics();
    let get = |stat| {
        stats
            .get(stat)
            .as_exact()
            .and_then(|scalar| T::try_from(&scalar).ok())
    };
    Some((get(Stat::Min)?, get(Stat::Max)?))
}

#[cfg(test)]
mod tests {
    use std::iter;

    use rstest::rstest;
    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::validity::Validity;
    use vortex_buffer::BitBuffer;
    use vortex_buffer::Buffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_utils::aliases::hash_map::HashMap;

    use super::ErasedStats;
    use super::IntegerStats;
    use super::typed_int_stats;

    #[test]
    fn test_naive_count_distinct_values() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let array = PrimitiveArray::new(buffer![217u8, 0], Validity::NonNullable);
        let stats = typed_int_stats::<u8>(&array, true, &mut ctx)?;
        assert_eq!(stats.distinct_count().unwrap(), 2);
        Ok(())
    }

    #[test]
    fn test_naive_count_distinct_values_nullable() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
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
        let mut ctx = array_session().create_execution_ctx();
        let array = PrimitiveArray::new((0..128u8).collect::<Buffer<u8>>(), Validity::NonNullable);
        let stats = typed_int_stats::<u8>(&array, true, &mut ctx)?;
        assert_eq!(stats.distinct_count().unwrap(), 128);
        Ok(())
    }

    #[test]
    fn test_count_distinct_values_nullable() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
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
    fn dense_distinct_signed_full_range() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let values: Buffer<i8> = (i8::MIN..=i8::MAX)
            .chain(i8::MIN..=i8::MAX)
            .chain(iter::once(i8::MAX))
            .collect();
        let array = PrimitiveArray::new(values, Validity::NonNullable);

        let stats = typed_int_stats::<i8>(&array, true, &mut ctx)?;
        assert_eq!(stats.distinct_count(), Some(256));
        assert_eq!(
            stats.most_frequent_value_and_count(),
            Some((i8::MAX.into(), 3))
        );
        Ok(())
    }

    #[test]
    fn dense_distinct_ignores_values_under_nulls() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let array = PrimitiveArray::new(
            buffer![1000i32, 1001, 1000, 5_000_000, 1003],
            Validity::from_iter([true, true, true, false, true]),
        );

        let stats = typed_int_stats::<i32>(&array, true, &mut ctx)?;
        assert_eq!(stats.distinct_count(), Some(3));
        assert_eq!(
            stats.most_frequent_value_and_count(),
            Some((1000i32.into(), 2))
        );
        Ok(())
    }

    #[test]
    fn wide_range_distinct() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let array = PrimitiveArray::new(buffer![0u64, u64::MAX, 0], Validity::NonNullable);

        let stats = typed_int_stats::<u64>(&array, true, &mut ctx)?;
        assert_eq!(stats.distinct_count(), Some(2));
        assert_eq!(
            stats.most_frequent_value_and_count(),
            Some((0u64.into(), 2))
        );
        Ok(())
    }

    #[test]
    fn test_integer_stats_leading_nulls() {
        let mut ctx = array_session().create_execution_ctx();
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

    /// Returns the per-value counts and the run count of the valid values, computed naively.
    fn naive_stats(values: &[u32], valid: &[bool]) -> (HashMap<u32, u32>, u32) {
        let mut counts = HashMap::default();
        let mut runs = 0;
        let mut prev = None;
        for (&value, _) in values.iter().zip(valid).filter(|(_, ok)| **ok) {
            *counts.entry(value).or_insert(0) += 1;
            if prev.replace(value) != Some(value) {
                runs += 1;
            }
        }
        (counts, runs)
    }

    /// Checks the chunked loops against a naive reference. The run lengths cover aligned
    /// constant chunks (64), runs that cross chunk boundaries (3, 100), and chunks where every
    /// value changes (1).
    #[rstest]
    fn test_matches_naive_reference(
        #[values(1, 63, 64, 65, 64 * 20 + 17)] len: u32,
        #[values(1, 3, 64, 100)] run_len: u32,
        #[values(None, Some(97), Some(3))] null_every: Option<u32>,
    ) -> VortexResult<()> {
        let values: Vec<u32> = (0..len).map(|i| (i / run_len) % 16).collect();
        let valid: Vec<bool> = (0..len)
            .map(|i| null_every.is_none_or(|n| i % n != 0))
            .collect();
        let (expected, runs) = naive_stats(&values, &valid);

        let validity = match null_every {
            None => Validity::NonNullable,
            Some(_) => Validity::from(BitBuffer::from(valid)),
        };
        let array = PrimitiveArray::new(Buffer::from(values), validity);
        let mut ctx = array_session().create_execution_ctx();
        let stats = typed_int_stats::<u32>(&array, true, &mut ctx)?;

        let ErasedStats::U32(typed) = stats.erased() else {
            unreachable!()
        };
        let actual: HashMap<u32, u32> = typed
            .distinct()
            .map(|d| d.distinct_values().iter().map(|(k, &c)| (k.0, c)).collect())
            .unwrap_or_default();
        assert_eq!(actual, expected);
        let value_count: u32 = expected.values().sum();
        assert_eq!(stats.value_count, value_count);
        if value_count > 0 {
            assert_eq!(stats.average_run_length, value_count / runs);
        }
        Ok(())
    }
}
