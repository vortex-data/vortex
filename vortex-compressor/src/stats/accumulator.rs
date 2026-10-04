// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Composable single-pass integer statistics.
//!
//! Each statistic is an [`IntAccumulator`]. A tuple of accumulators is itself an accumulator, so
//! any combination of statistics is computed by a single pass over the values with
//! [`accumulate`]:
//!
//! ```ignore
//! let (min_max, runs) = accumulate(values, &validity, (MinMax::new(), RunCount::new()));
//! ```
//!
//! The pass is fused at the granularity of [`CHUNK`] values rather than per value: every
//! accumulator runs its own kernel over a chunk while the chunk is still in L1, so each kernel
//! keeps its own vectorized, branch-free inner loop, and the values are read from memory once.
//!
//! Adding a statistic only takes a new [`IntAccumulator`] implementation; it composes with the
//! existing ones without changing them.

// The per-chunk kernels must inline into the fused pass to vectorize together.
#![allow(clippy::inline_always)]

use std::hash::Hash;

use num_traits::PrimInt;
use rustc_hash::FxBuildHasher;
use vortex_array::arrays::primitive::NativeValue;
use vortex_array::dtype::IntegerPType;
use vortex_error::VortexExpect;
use vortex_mask::AllOr;
use vortex_mask::Mask;
use vortex_utils::aliases::hash_map::HashMap;

use super::integer::DistinctInfo;

/// The number of values that every accumulator processes per fused step. This matches the width
/// of one validity word.
pub const CHUNK: usize = 64;

/// [`CHUNK`] as a `u32`.
const CHUNK_U32: u32 = 64;

/// A statistic over the valid values of an integer array, computed in a single pass.
///
/// [`accumulate`] calls [`start`](Self::start) once with the first valid value, then feeds every
/// valid value, including the first, exactly once and in order: fully valid runs of [`CHUNK`]
/// values through [`chunk`](Self::chunk), partially valid runs through
/// [`masked_chunk`](Self::masked_chunk), and the trailing values through [`push`](Self::push).
pub trait IntAccumulator<T: Copy> {
    /// The computed statistic.
    type Output;

    /// Initializes the state from the first valid value.
    fn start(&mut self, head: T);

    /// Accumulates [`CHUNK`] consecutive valid values.
    fn chunk(&mut self, values: &[T; CHUNK]);

    /// Accumulates the values of `values` whose bit is set in `valid`, least significant bit
    /// first.
    #[inline(always)]
    fn masked_chunk(&mut self, values: &[T; CHUNK], valid: u64) {
        push_set_bits(self, values, valid);
    }

    /// Accumulates one valid value.
    fn push(&mut self, value: T);

    /// Returns the statistic.
    fn finish(self) -> Self::Output;
}

/// Computes `acc` over the valid values of `values` in a single pass.
///
/// Returns `None` if there are no valid values.
pub fn accumulate<T, A>(values: &[T], validity: &Mask, mut acc: A) -> Option<A::Output>
where
    T: Copy,
    A: IntAccumulator<T>,
{
    debug_assert_eq!(values.len(), validity.len());
    match validity.bit_buffer() {
        AllOr::None => return None,
        AllOr::All => {
            acc.start(*values.first()?);
            let (chunks, remainder) = values.as_chunks::<CHUNK>();
            for chunk in chunks {
                acc.chunk(chunk);
            }
            for &value in remainder {
                acc.push(value);
            }
        }
        AllOr::Some(bits) => {
            acc.start(values[validity.first()?]);
            let bit_chunks = bits.chunks();
            let (chunks, remainder) = values.as_chunks::<CHUNK>();
            for (chunk, word) in chunks.iter().zip(bit_chunks.iter()) {
                match word {
                    0 => {}
                    u64::MAX => acc.chunk(chunk),
                    _ => acc.masked_chunk(chunk, word),
                }
            }
            push_set_bits(&mut acc, remainder, bit_chunks.remainder_bits());
        }
    }
    Some(acc.finish())
}

/// Pushes `values[i]` for every set bit `i` of `word`.
#[inline(always)]
fn push_set_bits<T: Copy, A: IntAccumulator<T> + ?Sized>(acc: &mut A, values: &[T], mut word: u64) {
    while word != 0 {
        acc.push(values[word.trailing_zeros() as usize]);
        word &= word - 1;
    }
}

/// Whether a chunk has few enough nulls that [`forward_fill`] beats pushing each valid value.
#[inline(always)]
fn is_mostly_valid(valid: u64) -> bool {
    valid.count_ones() >= CHUNK_U32 / 2
}

/// Returns `values` with each null replaced by the closest valid value before it, or by `prev`,
/// the last valid value before the chunk.
///
/// Repeating a valid value changes neither the extrema, the runs, nor the sortedness, so
/// accumulators of those can process the filled chunk with their fully valid kernel. The cost is
/// one store per null.
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

/// Implements [`IntAccumulator`] for a tuple of accumulators by forwarding to each element.
macro_rules! impl_tuple_accumulator {
    ($($name:ident),+) => {
        impl<T: Copy, $($name: IntAccumulator<T>),+> IntAccumulator<T> for ($($name,)+) {
            type Output = ($($name::Output,)+);

            #[inline(always)]
            fn start(&mut self, head: T) {
                #[allow(non_snake_case)]
                let ($($name,)+) = self;
                $($name.start(head);)+
            }

            #[inline(always)]
            fn chunk(&mut self, values: &[T; CHUNK]) {
                #[allow(non_snake_case)]
                let ($($name,)+) = self;
                $($name.chunk(values);)+
            }

            #[inline(always)]
            fn masked_chunk(&mut self, values: &[T; CHUNK], valid: u64) {
                #[allow(non_snake_case)]
                let ($($name,)+) = self;
                $($name.masked_chunk(values, valid);)+
            }

            #[inline(always)]
            fn push(&mut self, value: T) {
                #[allow(non_snake_case)]
                let ($($name,)+) = self;
                $($name.push(value);)+
            }

            #[inline]
            fn finish(self) -> Self::Output {
                #[allow(non_snake_case)]
                let ($($name,)+) = self;
                ($($name.finish(),)+)
            }
        }
    };
}

impl_tuple_accumulator!(A);
impl_tuple_accumulator!(A, B);
impl_tuple_accumulator!(A, B, C);
impl_tuple_accumulator!(A, B, C, D);
impl_tuple_accumulator!(A, B, C, D, E);
impl_tuple_accumulator!(A, B, C, D, E, F);

/// Counts the value changes in `values`, including the change from `prev` to `values[0]`.
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

/// The number of independent lanes that [`MinMax`] reduces into. Keeping the per-lane state
/// across chunks lets the reduction stay in vector registers, with a single horizontal reduction
/// in [`finish`](IntAccumulator::finish).
const MIN_MAX_LANES: usize = 32;

/// The minimum and maximum valid values.
#[derive(Debug, Clone, Copy)]
pub struct MinMax<T> {
    /// The minimum so far of each lane.
    min: [T; MIN_MAX_LANES],
    /// The maximum so far of each lane.
    max: [T; MIN_MAX_LANES],
    /// The first valid value, which stands in for leading nulls in [`forward_fill`].
    head: T,
}

impl<T: PrimInt> MinMax<T> {
    /// Creates an accumulator.
    pub fn new() -> Self {
        Self {
            min: [T::max_value(); MIN_MAX_LANES],
            max: [T::min_value(); MIN_MAX_LANES],
            head: T::zero(),
        }
    }

    /// Folds `values` into the lanes.
    #[inline(always)]
    fn fold(&mut self, values: &[T; CHUNK]) {
        // Without 64-bit vector compares in the baseline instruction set, wide lanes spill out of
        // registers, so 64-bit values use fewer lanes. `size_of` is a constant, so this folds.
        if size_of::<T>() >= 8 {
            fold_lanes::<T, 4>(&mut self.min, &mut self.max, values);
        } else {
            fold_lanes::<T, MIN_MAX_LANES>(&mut self.min, &mut self.max, values);
        }
    }
}

impl<T: PrimInt> Default for MinMax<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: PrimInt> IntAccumulator<T> for MinMax<T> {
    /// `(min, max)`.
    type Output = (T, T);

    #[inline(always)]
    fn start(&mut self, head: T) {
        self.head = head;
    }

    #[inline(always)]
    fn chunk(&mut self, values: &[T; CHUNK]) {
        self.fold(values);
    }

    #[inline(always)]
    fn masked_chunk(&mut self, values: &[T; CHUNK], valid: u64) {
        if is_mostly_valid(valid) {
            self.fold(&forward_fill(values, valid, self.head));
        } else {
            push_set_bits(self, values, valid);
        }
    }

    #[inline(always)]
    fn push(&mut self, value: T) {
        self.min[0] = self.min[0].min(value);
        self.max[0] = self.max[0].max(value);
    }

    #[inline]
    fn finish(self) -> (T, T) {
        (
            self.min.into_iter().fold(T::max_value(), T::min),
            self.max.into_iter().fold(T::min_value(), T::max),
        )
    }
}

/// Folds `values` into the first `L` lanes of `min` and `max`.
#[inline(always)]
fn fold_lanes<T: PrimInt, const L: usize>(
    min: &mut [T; MIN_MAX_LANES],
    max: &mut [T; MIN_MAX_LANES],
    values: &[T; CHUNK],
) {
    let (Some(min), Some(max)) = (min.first_chunk_mut::<L>(), max.first_chunk_mut::<L>()) else {
        unreachable!("L is at most MIN_MAX_LANES")
    };
    for group in values.as_chunks::<L>().0 {
        for i in 0..L {
            min[i] = if group[i] < min[i] { group[i] } else { min[i] };
            max[i] = if group[i] > max[i] { group[i] } else { max[i] };
        }
    }
}

/// The number of runs of equal consecutive valid values. Nulls do not break runs.
#[derive(Debug, Clone, Copy)]
pub struct RunCount<T> {
    /// The last value seen.
    prev: T,
    /// The number of runs so far.
    runs: u32,
}

impl<T: PrimInt> RunCount<T> {
    /// Creates an accumulator.
    pub fn new() -> Self {
        Self {
            prev: T::zero(),
            runs: 0,
        }
    }
}

impl<T: PrimInt> Default for RunCount<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: PrimInt> IntAccumulator<T> for RunCount<T> {
    type Output = u32;

    #[inline(always)]
    fn start(&mut self, head: T) {
        self.prev = head;
        self.runs = 1;
    }

    #[inline(always)]
    fn chunk(&mut self, values: &[T; CHUNK]) {
        self.runs += transitions(&self.prev, values);
        self.prev = values[CHUNK - 1];
    }

    #[inline(always)]
    fn masked_chunk(&mut self, values: &[T; CHUNK], valid: u64) {
        if is_mostly_valid(valid) {
            self.chunk(&forward_fill(values, valid, self.prev));
        } else {
            push_set_bits(self, values, valid);
        }
    }

    #[inline(always)]
    fn push(&mut self, value: T) {
        self.runs += u32::from(value != self.prev);
        self.prev = value;
    }

    #[inline]
    fn finish(self) -> u32 {
        self.runs
    }
}

/// Whether the valid values are sorted, non-strictly and strictly, in ascending order.
#[derive(Debug, Clone, Copy)]
pub struct Sorted<T> {
    /// The last value seen.
    prev: T,
    /// The number of adjacent pairs that decrease.
    decreases: u32,
    /// The number of adjacent pairs that are equal.
    repeats: u32,
}

impl<T: PrimInt> Sorted<T> {
    /// Creates an accumulator.
    pub fn new() -> Self {
        Self {
            prev: T::zero(),
            decreases: 0,
            repeats: 0,
        }
    }
}

impl<T: PrimInt> Default for Sorted<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Whether values are sorted, as computed by [`Sorted`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SortedResult {
    /// Every value is at least the previous one.
    pub sorted: bool,
    /// Every value is greater than the previous one.
    pub strict_sorted: bool,
}

impl<T: PrimInt> IntAccumulator<T> for Sorted<T> {
    type Output = SortedResult;

    #[inline(always)]
    fn start(&mut self, head: T) {
        self.prev = head;
    }

    #[inline(always)]
    fn chunk(&mut self, values: &[T; CHUNK]) {
        // Count rather than short-circuit, so both reductions stay branch-free and vectorize.
        let pairs =
            || std::iter::once((&self.prev, &values[0])).chain(values.iter().zip(&values[1..]));
        let decreases = pairs().map(|(a, b)| u8::from(b < a)).sum::<u8>();
        let repeats = pairs().map(|(a, b)| u8::from(b == a)).sum::<u8>();
        self.decreases += u32::from(decreases);
        self.repeats += u32::from(repeats);
        self.prev = values[CHUNK - 1];
    }

    #[inline(always)]
    fn masked_chunk(&mut self, values: &[T; CHUNK], valid: u64) {
        if is_mostly_valid(valid) {
            self.chunk(&forward_fill(values, valid, self.prev));
            // Each filled null repeats the value before it, adding exactly one equal pair.
            self.repeats -= (!valid).count_ones();
        } else {
            push_set_bits(self, values, valid);
        }
    }

    #[inline(always)]
    fn push(&mut self, value: T) {
        self.decreases += u32::from(value < self.prev);
        self.repeats += u32::from(value == self.prev);
        self.prev = value;
    }

    #[inline]
    fn finish(self) -> SortedResult {
        // `start` and the first `push` or `chunk` compare the head with itself once.
        let repeats = self.repeats.saturating_sub(1);
        SortedResult {
            sorted: self.decreases == 0,
            strict_sorted: self.decreases == 0 && repeats == 0,
        }
    }
}

/// Value ranges up to this many values may be counted in a dense array instead of a hash map.
const DENSE_DISTINCT_MAX_RANGE: usize = 1 << 16;

/// Value ranges up to this many values are always counted in a dense array, regardless of the array
/// length. This covers every `u8` and `i8` array.
const DENSE_DISTINCT_ALWAYS_RANGE: usize = 1 << 8;

/// Where the occurrences of each distinct value are counted.
enum Counts<T> {
    /// Counts indexed by `value - min`, used when the value range is narrow. This avoids hashing
    /// entirely.
    Dense {
        /// The smallest countable value.
        min: T,
        /// `min` cast to `usize`. Casting preserves values modulo `2^usize::BITS`, so the wrapping
        /// difference of two cast values is the exact difference of any two values in the range.
        min_index: usize,
        /// The number of occurrences of `min + index`.
        counts: Vec<u32>,
    },
    /// Counts keyed by value.
    Hashed(HashMap<NativeValue<T>, u32, FxBuildHasher>),
}

/// The occurrences of each distinct valid value, and the number of runs.
///
/// Consecutive equal values are counted once per run, so the counts are only updated on value
/// changes. That tracks the runs for free, so the run count is part of the output: composing this
/// with [`RunCount`] would detect every value change twice.
pub struct Distinct<T> {
    /// The occurrences counted so far.
    counts: Counts<T>,
    /// The value of the current run.
    prev: T,
    /// Occurrences of `prev` in the current run not yet added to `counts`.
    pending: u32,
    /// The number of runs so far.
    runs: u32,
}

impl<T: IntegerPType> Distinct<T>
where
    NativeValue<T>: Eq + Hash,
{
    /// Creates an accumulator for valid values within `min..=max`, choosing dense counting when
    /// the range is narrow relative to `len`, the number of values.
    pub fn new(min: T, max: T, len: usize) -> Self {
        let range_len = match (min.to_i128(), max.to_i128()) {
            (Some(min), Some(max)) => usize::try_from(max - min + 1).unwrap_or(usize::MAX),
            _ => usize::MAX,
        };

        // A dense counter is only worthwhile when it is not much larger than the array itself.
        let counts = if range_len <= DENSE_DISTINCT_ALWAYS_RANGE
            || (range_len <= DENSE_DISTINCT_MAX_RANGE && range_len <= len)
        {
            Counts::Dense {
                min,
                min_index: min.as_(),
                counts: vec![0; range_len],
            }
        } else {
            Counts::Hashed(HashMap::with_capacity_and_hasher(len / 2, FxBuildHasher))
        };

        Self {
            counts,
            prev: T::zero(),
            pending: 0,
            runs: 0,
        }
    }

    /// Creates an accumulator that needs no bounds, if `T` is narrow enough to always count densely
    /// over its whole domain.
    pub fn for_full_domain(len: usize) -> Option<Self> {
        (size_of::<T>() == 1).then(|| Self::new(T::min_value(), T::max_value(), len))
    }

    /// Adds the pending occurrences of `prev` to the counts.
    #[inline(always)]
    fn flush(&mut self) {
        let (value, count) = (self.prev, self.pending);
        match &mut self.counts {
            Counts::Dense {
                min_index, counts, ..
            } => counts[value.as_().wrapping_sub(*min_index)] += count,
            Counts::Hashed(map) => *map.entry(NativeValue(value)).or_insert(0) += count,
        }
        self.pending = 0;
    }
}

impl<T: IntegerPType> IntAccumulator<T> for Distinct<T>
where
    NativeValue<T>: Eq + Hash,
{
    /// The distinct values and the run count.
    type Output = (DistinctInfo<T>, u32);

    #[inline(always)]
    fn start(&mut self, head: T) {
        self.prev = head;
        self.runs = 1;
    }

    #[inline(always)]
    fn chunk(&mut self, values: &[T; CHUNK]) {
        if transitions(&self.prev, values) == 0 {
            self.pending += CHUNK_U32;
            return;
        }
        for &value in values {
            self.push(value);
        }
    }

    #[inline(always)]
    fn push(&mut self, value: T) {
        if value != self.prev {
            self.flush();
            self.prev = value;
            self.runs += 1;
        }
        self.pending += 1;
    }

    fn finish(mut self) -> (DistinctInfo<T>, u32) {
        self.flush();
        let distinct_values: HashMap<NativeValue<T>, u32, FxBuildHasher> = match self.counts {
            Counts::Dense { min, counts, .. } => {
                let min = min.to_i128().vortex_expect("integers fit in i128");
                counts
                    .iter()
                    .enumerate()
                    .filter(|&(_, &count)| count > 0)
                    .map(|(index, &count)| {
                        let value = <T as num_traits::NumCast>::from(min + index as i128)
                            .vortex_expect("values between min and max fit in the type");
                        (NativeValue(value), count)
                    })
                    .collect()
            }
            Counts::Hashed(map) => map,
        };
        (DistinctInfo::new(distinct_values), self.runs)
    }
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation)]
mod tests {
    use rstest::rstest;
    use vortex_mask::Mask;

    use super::*;

    /// Computes min, max, the run count and the per-value counts of the valid values naively.
    fn naive(values: &[i32], valid: &[bool]) -> Option<(i32, i32, u32, HashMap<i32, u32>)> {
        let valid_values: Vec<i32> = values
            .iter()
            .zip(valid)
            .filter(|(_, ok)| **ok)
            .map(|(v, _)| *v)
            .collect();
        let min = *valid_values.iter().min()?;
        let max = *valid_values.iter().max()?;
        let runs = 1 + valid_values.windows(2).filter(|w| w[0] != w[1]).count() as u32;
        let mut counts = HashMap::default();
        for v in valid_values {
            *counts.entry(v).or_insert(0) += 1;
        }
        Some((min, max, runs, counts))
    }

    /// Computes whether the valid values are sorted naively.
    fn naive_sorted(values: &[i32], valid: &[bool]) -> SortedResult {
        let valid_values: Vec<i32> = values
            .iter()
            .zip(valid)
            .filter(|(_, ok)| **ok)
            .map(|(v, _)| *v)
            .collect();
        SortedResult {
            sorted: valid_values.is_sorted(),
            strict_sorted: valid_values.windows(2).all(|w| w[0] < w[1]),
        }
    }

    #[rstest]
    fn sorted_inputs(
        #[values(1, 2, 64, 65, 200)] len: i32,
        #[values(None, Some(5))] null_every: Option<i32>,
    ) {
        let validity = Mask::from_iter((0..len).map(|i| null_every.is_none_or(|n| i % n != 1)));
        let strict: Vec<i32> = (0..len).collect();
        let repeated: Vec<i32> = (0..len).map(|i| i / 2).collect();
        let mut descending = strict.clone();
        descending.reverse();
        for values in [strict, repeated, descending] {
            let valid: Vec<bool> = validity.to_bit_buffer().iter().collect();
            assert_eq!(
                accumulate(&values, &validity, Sorted::new()).unwrap(),
                naive_sorted(&values, &valid),
                "{values:?}"
            );
        }
    }

    #[rstest]
    fn fused_matches_naive(
        #[values(0, 1, 63, 64, 65, 64 * 20 + 17)] len: usize,
        #[values(1, 3, 64, 100)] run_len: usize,
        #[values(None, Some(1), Some(3), Some(97), Some(usize::MAX))] null_every: Option<usize>,
        #[values(16, 100_000)] range: i32,
        #[values(false, true)] invert_validity: bool,
    ) {
        let values: Vec<i32> = (0..len)
            .map(|i| ((i / run_len) as i32).wrapping_mul(7919) % range - range / 2)
            .collect();
        // `Some(usize::MAX)` makes only the first value null; `Some(1)` makes every value null.
        let valid: Vec<bool> = (0..len)
            .map(|i| null_every.is_none_or(|n| (i % n != 0) != invert_validity))
            .collect();
        let validity = match null_every {
            None => Mask::new_true(len),
            Some(_) => Mask::from_iter(valid.iter().copied()),
        };

        let expected = naive(&values, &valid);
        let Some((min, max, runs, counts)) = expected else {
            assert!(accumulate(&values, &validity, MinMax::<i32>::new()).is_none());
            return;
        };

        let ((actual_min, actual_max), actual_runs, (distinct, distinct_runs), sorted) =
            accumulate(
                &values,
                &validity,
                (
                    MinMax::new(),
                    RunCount::new(),
                    Distinct::new(min, max, values.len()),
                    Sorted::new(),
                ),
            )
            .unwrap();
        assert_eq!(distinct_runs, runs);
        assert_eq!((actual_min, actual_max), (min, max));
        assert_eq!(sorted, naive_sorted(&values, &valid));
        assert_eq!(actual_runs, runs);
        let actual_counts: HashMap<i32, u32> = distinct
            .distinct_values()
            .iter()
            .map(|(k, &c)| (k.0, c))
            .collect();
        assert_eq!(actual_counts, counts);
    }

    #[test]
    fn full_domain_distinct_needs_no_bounds() {
        let values: Vec<i8> = (i8::MIN..=i8::MAX).chain([i8::MAX, 0]).collect();
        let (distinct, _) = accumulate(
            &values,
            &Mask::new_true(values.len()),
            Distinct::for_full_domain(values.len()).unwrap(),
        )
        .unwrap();
        assert_eq!(distinct.distinct_count(), 256);
        assert_eq!(distinct.most_frequent(), (i8::MAX, 2));
        assert!(Distinct::<i32>::for_full_domain(10).is_none());
    }
}
