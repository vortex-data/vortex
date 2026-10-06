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
use vortex_mask::AllOr;
use vortex_mask::Mask;
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

impl<T: PrimInt> DistinctInfo<T> {
    /// Describes an array without valid values.
    fn empty() -> Self {
        Self {
            distinct_values: HashMap::with_hasher(FxBuildHasher),
            distinct_count: 0,
            most_frequent_value: T::zero(),
            top_frequency: 0,
        }
    }

    /// Summarizes a non-empty map of distinct values to their occurrence counts.
    fn new(distinct_values: HashMap<NativeValue<T>, u32, FxBuildHasher>) -> Self {
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

    if array.all_invalid(ctx)? {
        return Ok(IntegerStats {
            null_count: u32::try_from(array.len())?,
            value_count: 0,
            average_run_length: 0,
            erased: TypedStats {
                min: T::max_value(),
                max: T::min_value(),
                distinct: Some(DistinctInfo::empty()),
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
    let values = array.as_slice::<T>();

    // Counting distinct values needs the bounds up front to choose a counter. Bytes are always
    // counted over their whole domain, so their bounds come with the counts in one pass.
    let cached = cached_min_max::<T>(array);
    let bounds = match cached {
        Some(bounds) => Some(bounds),
        None if count_distinct_values && size_of::<T>() > 1 => {
            let scan = scan::<T, NoCounter, true, false>(values, &validity, &mut NoCounter);
            Some((scan.min, scan.max))
        }
        None => None,
    };

    let (min, max, runs, distinct) = match bounds {
        // Every valid value is the same, so there is one run of all of them.
        Some((min, max)) if min == max => {
            let distinct = count_distinct_values.then(|| {
                let mut distinct_values = HashMap::with_capacity_and_hasher(1, FxBuildHasher);
                distinct_values.insert(NativeValue(min), value_count);
                DistinctInfo::new(distinct_values)
            });
            (min, max, 1, distinct)
        }
        Some((min, max)) if !count_distinct_values => {
            let scan = scan::<T, NoCounter, false, true>(values, &validity, &mut NoCounter);
            (min, max, scan.runs, None)
        }
        None if !count_distinct_values => {
            let scan = scan::<T, NoCounter, true, true>(values, &validity, &mut NoCounter);
            (scan.min, scan.max, scan.runs, None)
        }
        Some((min, max)) => {
            let (scan, distinct) = count_distinct::<T, false>(values, &validity, min, max);
            (min, max, scan.runs, Some(distinct))
        }
        None => {
            let (scan, distinct) =
                count_distinct::<T, true>(values, &validity, T::min_value(), T::max_value());
            (scan.min, scan.max, scan.runs, Some(distinct))
        }
    };

    // The bounds replace the `min_max` aggregate, which also cached the null count.
    if cached.is_none() {
        let stats = array.as_ref().statistics();
        stats.set(Stat::Min, Precision::Exact(PValue::from(min).into()));
        stats.set(Stat::Max, Precision::Exact(PValue::from(max).into()));
        stats.set(Stat::NullCount, Precision::exact(u64::from(null_count)));
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

/// Value ranges up to this many values may be counted in a dense array instead of a hash map.
const DENSE_DISTINCT_MAX_RANGE: usize = 1 << 16;

/// Value ranges up to this many values are always counted in a dense array, regardless of the array
/// length. This covers every `u8` and `i8` array.
const DENSE_DISTINCT_ALWAYS_RANGE: usize = 1 << 8;

/// Counts the distinct valid values within `min..=max`, and the runs, in one pass that also
/// computes the bounds if `MIN_MAX`. The counter is chosen once, so the pass is monomorphized on
/// it.
fn count_distinct<T, const MIN_MAX: bool>(
    values: &[T],
    validity: &Mask,
    min: T,
    max: T,
) -> (Scan<T>, DistinctInfo<T>)
where
    T: IntegerPType + PrimInt,
    NativeValue<T>: Eq + Hash,
{
    let min_i128 = min.to_i128().vortex_expect("integers fit in i128");
    let max_i128 = max.to_i128().vortex_expect("integers fit in i128");
    let range_len = usize::try_from(max_i128 - min_i128 + 1).unwrap_or(usize::MAX);

    // A dense counter is only worthwhile when it is not much larger than the array itself.
    if range_len <= DENSE_DISTINCT_ALWAYS_RANGE
        || (range_len <= DENSE_DISTINCT_MAX_RANGE && range_len <= values.len())
    {
        let mut counts = vec![0u32; range_len];
        let scan = scan::<T, _, MIN_MAX, true>(
            values,
            validity,
            &mut Dense {
                min_index: min.as_(),
                counts: &mut counts,
            },
        );
        let distinct_values = counts
            .iter()
            .enumerate()
            .filter(|&(_, &count)| count > 0)
            .map(|(index, &count)| {
                let value = <T as num_traits::NumCast>::from(min_i128 + index as i128)
                    .vortex_expect("values between min and max fit in the type");
                (NativeValue(value), count)
            })
            .collect();
        (scan, DistinctInfo::new(distinct_values))
    } else {
        let mut distinct_values =
            HashMap::with_capacity_and_hasher(values.len() / 2, FxBuildHasher);
        let scan = scan::<T, _, MIN_MAX, true>(values, validity, &mut distinct_values);
        (scan, DistinctInfo::new(distinct_values))
    }
}

/// Counts occurrences of valid values.
trait Counter<T> {
    /// Whether this counts at all. The pass skips all counting work otherwise.
    const COUNTS: bool = true;

    /// Adds `count` occurrences of `value`.
    fn add(&mut self, value: T, count: u32);
}

/// Counts nothing.
struct NoCounter;

impl<T> Counter<T> for NoCounter {
    const COUNTS: bool = false;

    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn add(&mut self, _value: T, _count: u32) {}
}

/// Counts indexed by `value - min`, used when the value range is narrow. This avoids hashing
/// entirely.
struct Dense<'a> {
    /// `min` cast to `usize`. Casting preserves values modulo `2^usize::BITS`, so the wrapping
    /// difference of two cast values is the exact difference of any two values in the range.
    min_index: usize,
    /// The number of occurrences of `min + index`.
    counts: &'a mut [u32],
}

impl<T: IntegerPType> Counter<T> for Dense<'_> {
    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn add(&mut self, value: T, count: u32) {
        self.counts[value.as_().wrapping_sub(self.min_index)] += count;
    }
}

impl<T: Copy> Counter<T> for HashMap<NativeValue<T>, u32, FxBuildHasher>
where
    NativeValue<T>: Eq + Hash,
{
    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn add(&mut self, value: T, count: u32) {
        *self.entry(NativeValue(value)).or_insert(0) += count;
    }
}

/// The values per step of a [`scan`], one validity word.
const CHUNK: usize = 64;

/// [`CHUNK`] as a `u32`.
const CHUNK_U32: u32 = 64;

/// The result of a [`scan`]. Statistics it did not compute keep their initial values: `T::MAX`
/// for `min`, `T::MIN` for `max` and 1 for `runs`.
struct Scan<T> {
    /// The smallest valid value.
    min: T,
    /// The largest valid value.
    max: T,
    /// The number of runs of equal consecutive valid values.
    runs: u32,
}

/// The state of a [`scan`], kept in locals so that it stays in registers: the counter is reached
/// through a pointer, so a count update could otherwise alias it.
struct ScanState<T> {
    /// The last valid value.
    prev: T,
    /// Occurrences of `prev` in the current run not yet added to the counter.
    pending: u32,
    /// The number of runs so far.
    runs: u32,
}

/// Computes, in one pass over the valid values, the bounds if `MIN_MAX`, the runs if `RUNS`,
/// and the counts of `counter`, which also computes the runs.
///
/// Every value is loaded once per chunk of 64, whose validity is one word. Nulls in a chunk are
/// filled with the last valid value before them, which changes neither the bounds nor the runs, so
/// that the bounds and runs take the same vectorized kernel with and without nulls. Counting skips
/// the nulls instead.
///
/// Not inlined: in the caller, which covers every integer type, the vectorizer gives up on the
/// lanes.
fn scan<T, C, const MIN_MAX: bool, const RUNS: bool>(
    values: &[T],
    validity: &Mask,
    counter: &mut C,
) -> Scan<T>
where
    T: PrimInt,
    C: Counter<T>,
{
    let mut bounds = Bounds::new();
    let runs = scan_into::<T, C, MIN_MAX, RUNS>(values, validity, counter, &mut bounds);
    let (min, max) = bounds.finish();
    Scan { min, max, runs }
}

/// Runs a [`scan`] that folds the bounds into `bounds`, and returns the runs.
///
/// The bounds are behind a reference so that each chunk folds them with vector loads and stores.
/// As locals of the loop, they are split into scalars that the vectorizer leaves scalar.
#[inline(never)]
fn scan_into<T, C, const MIN_MAX: bool, const RUNS: bool>(
    values: &[T],
    validity: &Mask,
    counter: &mut C,
    bounds: &mut Bounds<T>,
) -> u32
where
    T: PrimInt,
    C: Counter<T>,
{
    let head = validity
        .first()
        .vortex_expect("all-null arrays are handled before");
    let mut state = ScanState {
        prev: values[head],
        pending: 0,
        runs: 1,
    };
    let (chunks, remainder) = values.as_chunks::<CHUNK>();

    let remainder_valid = match validity.bit_buffer() {
        AllOr::None => unreachable!("all-null arrays are handled before"),
        AllOr::All => {
            for chunk in chunks {
                state.chunk::<C, MIN_MAX, RUNS>(chunk, bounds, counter);
            }
            (1u64 << remainder.len()) - 1
        }
        AllOr::Some(bits) => {
            let bit_chunks = bits.chunks();
            for (chunk, word) in chunks.iter().zip(bit_chunks.iter()) {
                match word {
                    0 => {}
                    u64::MAX => state.chunk::<C, MIN_MAX, RUNS>(chunk, bounds, counter),
                    _ => state.partial_chunk::<C, MIN_MAX, RUNS>(chunk, word, bounds, counter),
                }
            }
            bit_chunks.remainder_bits()
        }
    };

    // Pad the trailing values into a last chunk, whose padding is null.
    if remainder_valid != 0 {
        let mut last = [state.prev; CHUNK];
        last[..remainder.len()].copy_from_slice(remainder);
        state.partial_chunk::<C, MIN_MAX, RUNS>(&last, remainder_valid, bounds, counter);
    }

    if C::COUNTS {
        counter.add(state.prev, state.pending);
    }
    state.runs
}

impl<T: PrimInt> ScanState<T> {
    /// Accumulates a chunk of valid values.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn chunk<C: Counter<T>, const MIN_MAX: bool, const RUNS: bool>(
        &mut self,
        values: &[T; CHUNK],
        bounds: &mut Bounds<T>,
        counter: &mut C,
    ) {
        if MIN_MAX {
            bounds.fold(values);
        }
        if C::COUNTS || RUNS {
            let changes = transitions(&self.prev, values);
            self.runs += changes;
            if !C::COUNTS {
                self.prev = values[CHUNK - 1];
            } else if changes == 0 {
                self.pending += CHUNK_U32;
            } else {
                for &value in values {
                    if value != self.prev {
                        counter.add(self.prev, self.pending);
                        self.prev = value;
                        self.pending = 0;
                    }
                    self.pending += 1;
                }
            }
        }
    }

    /// Accumulates the valid values of a chunk, whose bits are set in `valid`.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn partial_chunk<C: Counter<T>, const MIN_MAX: bool, const RUNS: bool>(
        &mut self,
        values: &[T; CHUNK],
        valid: u64,
        bounds: &mut Bounds<T>,
        counter: &mut C,
    ) {
        if C::COUNTS {
            if MIN_MAX {
                bounds.fold(&forward_fill(values, valid, self.prev));
            }
            let mut valid = valid;
            while valid != 0 {
                let value = values[valid.trailing_zeros() as usize];
                if value != self.prev {
                    counter.add(self.prev, self.pending);
                    self.prev = value;
                    self.pending = 0;
                    self.runs += 1;
                }
                self.pending += 1;
                valid &= valid - 1;
            }
        } else {
            self.chunk::<C, MIN_MAX, RUNS>(
                &forward_fill(values, valid, self.prev),
                bounds,
                counter,
            );
        }
    }
}

/// The bounds of each lane of a [`scan`], apart from the rest of its state so that the vectorizer
/// keeps them in registers.
struct Bounds<T> {
    /// The minimum so far of each lane.
    min: [T; CHUNK],
    /// The maximum so far of each lane.
    max: [T; CHUNK],
}

impl<T: PrimInt> Bounds<T> {
    /// Returns empty bounds.
    fn new() -> Self {
        Self {
            min: [T::max_value(); CHUNK],
            max: [T::min_value(); CHUNK],
        }
    }

    /// Reduces the lanes to the overall minimum and maximum.
    fn finish(&self) -> (T, T) {
        let min = self
            .min
            .iter()
            .fold(T::max_value(), |acc, &lane| acc.min(lane));
        let max = self
            .max
            .iter()
            .fold(T::min_value(), |acc, &lane| acc.max(lane));
        (min, max)
    }

    /// Folds a chunk into the bounds of each lane.
    ///
    /// The lane count depends on the width and the instruction set, and was chosen by
    /// measurement. With too few lanes for the vector width, the vectorizer packs lanes across
    /// groups and loads them one by one: 16-bit values use a whole chunk, and so do 32-bit values
    /// with 256-bit vectors. Without 64-bit vector compares in the baseline instruction set,
    /// 64-bit values use only a few lanes. `size_of` and `cfg!` are constants, so the dispatch
    /// folds.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn fold(&mut self, values: &[T; CHUNK]) {
        let avx2 = cfg!(target_feature = "avx2");
        match size_of::<T>() {
            1 => self.lanes::<32>(values),
            2 => self.lanes::<CHUNK>(values),
            4 if avx2 => self.lanes::<CHUNK>(values),
            4 => self.lanes::<16>(values),
            _ if avx2 => self.lanes::<8>(values),
            _ => self.lanes::<4>(values),
        }
    }

    /// Folds a chunk into the bounds of the first `L` lanes.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn lanes<const L: usize>(&mut self, values: &[T; CHUNK]) {
        let (Some(min), Some(max)) = (
            self.min.first_chunk_mut::<L>(),
            self.max.first_chunk_mut::<L>(),
        ) else {
            unreachable!("L is at most CHUNK")
        };
        for group in values.as_chunks::<L>().0 {
            for i in 0..L {
                min[i] = if group[i] < min[i] { group[i] } else { min[i] };
                max[i] = if group[i] > max[i] { group[i] } else { max[i] };
            }
        }
    }
}

/// Returns `values` with each null, whose bit is unset in `valid`, replaced by the closest valid
/// value before it, or by `prev`, the last valid value before the chunk.
#[allow(clippy::inline_always)]
#[inline(always)]
fn forward_fill<T: Copy>(values: &[T; CHUNK], valid: u64, prev: T) -> [T; CHUNK] {
    let mut filled = *values;
    let mut nulls = !valid;
    while nulls != 0 {
        let i = nulls.trailing_zeros() as usize;
        filled[i] = if i == 0 { prev } else { filled[i - 1] };
        nulls &= nulls - 1;
    }
    filled
}

/// Counts the value changes in `values`, including the change from `prev` to `values[0]`.
#[allow(clippy::inline_always)]
#[inline(always)]
fn transitions<T: PartialEq>(prev: &T, values: &[T; CHUNK]) -> u32 {
    // Branch-free. At most 64, so a `u8` accumulator lets the comparison use full-width byte
    // lanes.
    let changes = u8::from(values[0] != *prev)
        + values
            .iter()
            .zip(&values[1..])
            .map(|(a, b)| u8::from(a != b))
            .sum::<u8>();
    u32::from(changes)
}

#[cfg(test)]
mod tests {
    // Test values wrap into their type, and test lengths are small.
    #![allow(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::cast_sign_loss
    )]

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
    use super::Stat;
    use super::StatsProvider;
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
    fn caches_bounds_and_null_count() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let array = PrimitiveArray::new(
            buffer![5i64, 3, 9, 4],
            Validity::from_iter([true, false, true, true]),
        );
        typed_int_stats::<i64>(&array, false, &mut ctx)?;

        let stats = array.as_ref().statistics();
        let exact = |stat| stats.get(stat).as_exact();
        let int = |stat| exact(stat).and_then(|scalar| i64::try_from(&scalar).ok());
        assert_eq!(int(Stat::Min), Some(4));
        assert_eq!(int(Stat::Max), Some(9));
        let null_count = exact(Stat::NullCount).and_then(|scalar| u64::try_from(&scalar).ok());
        assert_eq!(null_count, Some(1));
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
    /// value changes (1). With nulls, `null_chunks` also nulls two whole chunks.
    #[rstest]
    fn test_matches_naive_reference(
        #[values(1, 63, 64, 65, 64 * 20 + 17)] len: u32,
        #[values(1, 3, 64, 100)] run_len: u32,
        #[values(None, Some(97), Some(3))] null_every: Option<u32>,
        #[values(false, true)] null_chunks: bool,
        #[values(false, true)] count_distinct_values: bool,
    ) -> VortexResult<()> {
        let values: Vec<u32> = (0..len).map(|i| (i / run_len) % 16).collect();
        let valid: Vec<bool> = (0..len)
            .map(|i| {
                null_every.is_none_or(|n| i % n != 0 && !(null_chunks && (64..192).contains(&i)))
            })
            .collect();
        let (expected, runs) = naive_stats(&values, &valid);

        let validity = match null_every {
            None => Validity::NonNullable,
            Some(_) => Validity::from(BitBuffer::from(valid)),
        };
        let array = PrimitiveArray::new(Buffer::from(values), validity);
        let mut ctx = array_session().create_execution_ctx();
        let stats = typed_int_stats::<u32>(&array, count_distinct_values, &mut ctx)?;

        let ErasedStats::U32(typed) = stats.erased() else {
            unreachable!()
        };
        if count_distinct_values {
            let actual: HashMap<u32, u32> = typed
                .distinct()
                .map(|d| d.distinct_values().iter().map(|(k, &c)| (k.0, c)).collect())
                .unwrap_or_default();
            assert_eq!(actual, expected);
        }
        let value_count: u32 = expected.values().sum();
        assert_eq!(stats.value_count, value_count);
        if value_count > 0 {
            assert_eq!(stats.average_run_length, value_count / runs);
        }
        Ok(())
    }

    /// One case of the naive comparison: the length, the run length, the value range, every how
    /// many values a null falls (if any), whether distinct values are counted, and whether the
    /// bounds were cached by a previous pass.
    type Case = (usize, usize, i64, Option<usize>, bool, bool);

    /// The cases of the naive comparison. Runs cross chunks, ranges are narrow or wide, and with
    /// nulls, whole chunks are null as well as values at both ends. One test loops over all of
    /// them, so that they don't each start a test process.
    fn naive_cases() -> impl Iterator<Item = Case> {
        let lens = [1, 64, 64 * 40 + 17];
        lens.into_iter().flat_map(|len| {
            [1, 3, 100].into_iter().flat_map(move |run_len| {
                [5, 300, 1_000_000].into_iter().flat_map(move |range| {
                    [None, Some(2), Some(9), Some(1000)]
                        .into_iter()
                        .flat_map(move |null_every| {
                            [(false, false), (false, true), (true, false), (true, true)]
                                .into_iter()
                                .map(move |(count_distinct, cached)| {
                                    (len, run_len, range, null_every, count_distinct, cached)
                                })
                        })
                })
            })
        })
    }

    /// Checks every pass against a naive reference, for each width: the bounds with the runs, the
    /// bounds alone, and the counts with and without known bounds.
    macro_rules! matches_naive {
        ($name:ident, $T:ty, $variant:ident) => {
            #[test]
            fn $name() -> VortexResult<()> {
                for case in naive_cases() {
                    let (len, run_len, range, null_every, count_distinct, cached) = case;
                    let values: Vec<$T> = (0..len)
                        .map(|i| {
                            let x = ((i / run_len) as i64).wrapping_mul(7919) % range - range / 3;
                            x as $T
                        })
                        .collect();
                    let valid: Vec<bool> = (0..len)
                        .map(|i| null_every.is_none_or(|n| i % n != 0 && !(128..192).contains(&i)))
                        .collect();
                    let valid_values: Vec<$T> = values
                        .iter()
                        .zip(&valid)
                        .filter(|(_, ok)| **ok)
                        .map(|(&v, _)| v)
                        .collect();
                    if valid_values.is_empty() {
                        continue;
                    }
                    let mut counts: HashMap<$T, u32> = HashMap::default();
                    for &v in &valid_values {
                        *counts.entry(v).or_insert(0) += 1;
                    }
                    let runs = 1 + valid_values.windows(2).filter(|w| w[0] != w[1]).count() as u32;

                    let validity = match null_every {
                        None => Validity::NonNullable,
                        Some(_) => Validity::from(BitBuffer::from(valid)),
                    };
                    let array = PrimitiveArray::new(Buffer::from(values), validity);
                    let mut ctx = array_session().create_execution_ctx();
                    if cached {
                        typed_int_stats::<$T>(&array, false, &mut ctx)?;
                    }
                    let stats = typed_int_stats::<$T>(&array, count_distinct, &mut ctx)?;
                    let ErasedStats::$variant(typed) = stats.erased() else {
                        unreachable!()
                    };
                    assert_eq!(typed.min, *valid_values.iter().min().unwrap(), "{case:?}");
                    assert_eq!(typed.max, *valid_values.iter().max().unwrap(), "{case:?}");
                    assert_eq!(
                        stats.average_run_length,
                        valid_values.len() as u32 / runs,
                        "{case:?}"
                    );
                    let actual: Option<HashMap<$T, u32>> = typed
                        .distinct()
                        .map(|d| d.distinct_values().iter().map(|(k, &c)| (k.0, c)).collect());
                    assert_eq!(actual, count_distinct.then_some(counts), "{case:?}");
                }
                Ok(())
            }
        };
    }

    matches_naive!(u8_matches_naive, u8, U8);
    matches_naive!(i16_matches_naive, i16, I16);
    matches_naive!(u32_matches_naive, u32, U32);
    matches_naive!(i64_matches_naive, i64, I64);
    matches_naive!(u64_matches_naive, u64, U64);
}
