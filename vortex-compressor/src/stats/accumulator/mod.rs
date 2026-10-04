// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Composable single-pass integer statistics.
//!
//! Each statistic is an [`IntAccumulator`]. A tuple of accumulators is itself an accumulator, so
//! any combination of statistics is computed by a single pass over the values:
//!
//! ```ignore
//! let ((min, max), sum) = accumulate(values, &validity, (MinMax::new(), Sum::new()))?;
//! ```
//!
//! The pass is fused at the granularity of [`CHUNK`] values rather than per value: every
//! accumulator runs its own kernel over a chunk while the chunk is still in L1, so each kernel
//! keeps its own vectorized, branch-free inner loop, and the values are read from memory once.
//!
//! Only the pass is generic. [`compute`] stores the results type-erased in an [`IntStats`],
//! keyed by an [`IntStat`] that does not depend on the integer type, so callers read them without
//! being generic themselves:
//!
//! ```ignore
//! let stats = match_each_integer_ptype!(array.ptype(), |T| {
//!     compute(array.as_slice::<T>(), &validity, (MinMax::new(), Sum::new()))
//! });
//! let (min, max) = stats.get::<MinMaxStat>();
//! ```
//!
//! Adding a statistic only takes a new [`IntAccumulator`] and [`IntStat`]; it composes with the
//! existing ones without changing them.

// The per-chunk kernels must inline into the fused pass to vectorize together.
#![allow(clippy::inline_always)]

mod bits;
mod delta;
mod distinct;
mod min_max;
mod run_count;
mod sorted;
mod sum;
#[cfg(test)]
mod tests;

use std::any::Any;
use std::any::TypeId;

pub use bits::BitWidthHistogram;
pub use bits::BitWidthHistogramStat;
pub use bits::CommonBits;
pub use bits::CommonBitsResult;
pub use bits::CommonBitsStat;
pub use delta::DeltaRange;
pub use delta::DeltaRangeStat;
pub use distinct::Distinct;
pub use distinct::DistinctStat;
pub use distinct::DistinctSummary;
pub use min_max::MinMax;
pub use min_max::MinMaxStat;
use num_traits::AsPrimitive;
use num_traits::PrimInt;
pub use run_count::RunCount;
pub use run_count::RunCountStat;
pub use sorted::Sorted;
pub use sorted::SortedResult;
pub use sorted::SortedStat;
pub use sum::Sum;
pub use sum::SumStat;
use vortex_array::dtype::IntegerPType;
use vortex_array::scalar::PValue;
use vortex_mask::AllOr;
use vortex_mask::Mask;

/// An integer type that every statistic supports.
pub trait IntValue: IntegerPType + AsPrimitive<i64> + AsPrimitive<u64> + AsPrimitive<i128> {
    /// The narrowest signed type that holds the exact difference of any two values, so that
    /// differences vectorize in lanes as narrow as possible.
    type Delta: PrimInt + Into<i128> + Send + Sync + 'static;

    /// Returns the exact difference `self - prev`.
    fn delta(self, prev: Self) -> Self::Delta;

    /// Converts the value to a [`PValue`].
    fn to_pvalue(self) -> PValue;
}

/// Implements [`IntValue`] for integer types.
macro_rules! impl_int_value {
    ($($T:ty => $Delta:ty),+) => {
        $(impl IntValue for $T {
            type Delta = $Delta;

            #[inline(always)]
            fn delta(self, prev: Self) -> $Delta {
                <$Delta>::from(self) - <$Delta>::from(prev)
            }

            fn to_pvalue(self) -> PValue {
                self.into()
            }
        })+
    };
}

impl_int_value!(
    u8 => i16, u16 => i32, u32 => i64, u64 => i128,
    i8 => i16, i16 => i32, i32 => i64, i64 => i128
);

/// The number of independent lanes that lane-wise reductions keep across chunks, so that the
/// reduction stays in vector registers with a single horizontal reduction when finishing.
const LANES: usize = CHUNK;

/// Folds `values` into `lanes` with `f`.
///
/// Narrow accumulators use every lane, so that each lane group fills whole vector registers.
/// Without 64-bit vector compares in the baseline instruction set, wide accumulators spill out of
/// registers, so they use only the first few lanes. `size_of` is a constant, so this folds.
#[inline(always)]
fn fold_lanes<A: Copy, T: Copy>(
    lanes: &mut [A; LANES],
    values: &[T; CHUNK],
    f: impl Fn(A, T) -> A,
) {
    match size_of::<A>() {
        // A whole chunk per group: with two groups of 16-bit lanes, the vectorizer packs across
        // the groups and loads lane by lane.
        2 => fold_first_lanes::<A, T, CHUNK>(lanes, values, f),
        ..8 => fold_first_lanes::<A, T, 32>(lanes, values, f),
        _ => fold_first_lanes::<A, T, 4>(lanes, values, f),
    }
}

/// Folds `values` into the first `L` of `lanes` with `f`.
#[inline(always)]
fn fold_first_lanes<A: Copy, T: Copy, const L: usize>(
    lanes: &mut [A; LANES],
    values: &[T; CHUNK],
    f: impl Fn(A, T) -> A,
) {
    let Some(lanes) = lanes.first_chunk_mut::<L>() else {
        unreachable!("L is at most LANES")
    };
    for group in values.as_chunks::<L>().0 {
        for i in 0..L {
            lanes[i] = f(lanes[i], group[i]);
        }
    }
}

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
    drive(values, validity, &mut acc).then(|| acc.finish())
}

/// Computes `acc` over the valid values of `values` in a single pass, and returns its statistics
/// type-erased.
///
/// Returns an empty set if there are no valid values.
pub fn compute<T, A>(values: &[T], validity: &Mask, mut acc: A) -> IntStats
where
    T: Copy,
    A: ErasedAccumulator<T>,
{
    let mut stats = IntStats::default();
    if drive(values, validity, &mut acc) {
        acc.finish_into(&mut stats);
    }
    stats
}

/// Feeds the valid values of `values` to `acc`, returning `false` if there are none.
#[inline(always)]
fn drive<T, A>(values: &[T], validity: &Mask, acc: &mut A) -> bool
where
    T: Copy,
    A: IntAccumulator<T>,
{
    debug_assert_eq!(values.len(), validity.len());
    match validity.bit_buffer() {
        AllOr::None => return false,
        AllOr::All => {
            let Some(&head) = values.first() else {
                return false;
            };
            acc.start(head);
            let (chunks, remainder) = values.as_chunks::<CHUNK>();
            for chunk in chunks {
                acc.chunk(chunk);
            }
            for &value in remainder {
                acc.push(value);
            }
        }
        AllOr::Some(bits) => {
            let Some(head) = validity.first() else {
                return false;
            };
            acc.start(values[head]);
            let bit_chunks = bits.chunks();
            let (chunks, remainder) = values.as_chunks::<CHUNK>();
            for (chunk, word) in chunks.iter().zip(bit_chunks.iter()) {
                match word {
                    0 => {}
                    u64::MAX => acc.chunk(chunk),
                    _ => acc.masked_chunk(chunk, word),
                }
            }
            push_set_bits(acc, remainder, bit_chunks.remainder_bits());
        }
    }
    true
}

/// The identity of a statistic, independent of the integer type it is computed over.
pub trait IntStat: 'static {
    /// The type-erased value of the statistic.
    type Value: Send + Sync + 'static;
}

/// An accumulator that can store its statistics type-erased in an [`IntStats`].
pub trait ErasedAccumulator<T: Copy>: IntAccumulator<T> {
    /// Finishes the accumulator and inserts its statistics into `stats`.
    fn finish_into(self, stats: &mut IntStats);
}

/// Type-erased statistics, keyed by [`IntStat`].
#[derive(Default)]
pub struct IntStats {
    /// The values, keyed by the [`TypeId`] of their [`IntStat`]. There are only a few statistics,
    /// so a vector beats a map.
    entries: Vec<(TypeId, Box<dyn Any + Send + Sync>)>,
}

impl IntStats {
    /// Returns the value of `S`, if it was computed.
    pub fn get<S: IntStat>(&self) -> Option<&S::Value> {
        self.entries
            .iter()
            .find(|(id, _)| *id == TypeId::of::<S>())
            .and_then(|(_, value)| value.downcast_ref())
    }

    /// Sets the value of `S`.
    pub fn insert<S: IntStat>(&mut self, value: S::Value) {
        let value: Box<dyn Any + Send + Sync> = Box::new(value);
        match self
            .entries
            .iter_mut()
            .find(|(id, _)| *id == TypeId::of::<S>())
        {
            Some((_, existing)) => *existing = value,
            None => self.entries.push((TypeId::of::<S>(), value)),
        }
    }

    /// Returns the number of statistics.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if no statistics were computed.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
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

/// Returns `values` with each null replaced by `fill`.
#[inline(always)]
fn fill_nulls<T: Copy>(values: &[T; CHUNK], valid: u64, fill: T) -> [T; CHUNK] {
    let mut filled = *values;
    let mut nulls = !valid;
    while nulls != 0 {
        filled[nulls.trailing_zeros() as usize] = fill;
        nulls &= nulls - 1;
    }
    filled
}

/// Implements [`IntAccumulator`] and [`ErasedAccumulator`] for a tuple of accumulators by forwarding to each element.
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

        impl<T: Copy, $($name: ErasedAccumulator<T>),+> ErasedAccumulator<T> for ($($name,)+) {
            #[inline]
            fn finish_into(self, stats: &mut IntStats) {
                #[allow(non_snake_case)]
                let ($($name,)+) = self;
                $($name.finish_into(stats);)+
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
impl_tuple_accumulator!(A, B, C, D, E, F, G);
impl_tuple_accumulator!(A, B, C, D, E, F, G, H);

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
