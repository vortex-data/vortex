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
pub trait IntValue:
    IntegerPType
    + AsPrimitive<u8>
    + AsPrimitive<u16>
    + AsPrimitive<i32>
    + AsPrimitive<i64>
    + AsPrimitive<u64>
    + AsPrimitive<i128>
{
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

/// The capacity of a lane array. Lane-wise reductions keep one partial result per lane across
/// chunks, so that they stay in vector registers with a single horizontal reduction at the end.
const LANES: usize = CHUNK;

/// Folds `values` into the lanes `a` and `b` with `fa` and `fb`, in one loop.
///
/// The lane count depends on the accumulator width, and was chosen by measurement. Narrow lanes
/// use two or four 128-bit registers per array, leaving registers for a few reductions fused in
/// one loop. 16-bit lanes use a whole chunk: with several groups of 16-bit lanes, the vectorizer
/// packs across the groups and loads lane by lane. Without 64-bit vector compares in the baseline
/// instruction set, wide lanes spill out of registers, so they use only a few. `size_of` is a
/// constant, so the dispatch folds.
#[inline(always)]
fn fold_lanes2<A: Copy, B: Copy, T: Copy>(
    a: &mut [A; LANES],
    b: &mut [B; LANES],
    values: &[T; CHUNK],
    fa: impl Fn(A, T) -> A,
    fb: impl Fn(B, T) -> B,
) {
    match size_of::<A>().max(size_of::<B>()) {
        1 => fold_first_lanes::<A, B, T, 32>(a, b, values, fa, fb),
        2 => fold_first_lanes::<A, B, T, CHUNK>(a, b, values, fa, fb),
        4 => fold_first_lanes::<A, B, T, 16>(a, b, values, fa, fb),
        8 => fold_first_lanes::<A, B, T, 4>(a, b, values, fa, fb),
        _ => fold_first_lanes::<A, B, T, 2>(a, b, values, fa, fb),
    }
}

/// Folds `values` into `lanes` with `f`. See [`fold_lanes2`].
#[inline(always)]
fn fold_lanes<A: Copy, T: Copy>(
    lanes: &mut [A; LANES],
    values: &[T; CHUNK],
    f: impl Fn(A, T) -> A,
) {
    let mut unused = [(); LANES];
    fold_lanes2(lanes, &mut unused, values, f, |(), _| ());
}

/// Folds `values` into the first `L` lanes of `a` and `b`.
#[inline(always)]
fn fold_first_lanes<A: Copy, B: Copy, T: Copy, const L: usize>(
    a: &mut [A; LANES],
    b: &mut [B; LANES],
    values: &[T; CHUNK],
    fa: impl Fn(A, T) -> A,
    fb: impl Fn(B, T) -> B,
) {
    let (Some(a), Some(b)) = (a.first_chunk_mut::<L>(), b.first_chunk_mut::<L>()) else {
        unreachable!("L is at most LANES")
    };
    for group in values.as_chunks::<L>().0 {
        for i in 0..L {
            a[i] = fa(a[i], group[i]);
            b[i] = fb(b[i], group[i]);
        }
    }
}

/// Reduces the lanes to one value.
#[inline(always)]
fn reduce_lanes<A: Copy>(lanes: &[A; LANES], init: A, f: impl Fn(A, A) -> A) -> A {
    lanes.iter().fold(init, |acc, &lane| f(acc, lane))
}

/// The number of values that every accumulator processes per fused step. This matches the width
/// of one validity word.
pub const CHUNK: usize = 64;

/// [`CHUNK`] as a `u32`.
const CHUNK_U32: u32 = 64;

/// The bytes of values in a block. Every accumulator in a tuple passes over a block in turn, so a
/// block must stay in L1 meanwhile.
const BLOCK_BYTES: usize = 8 << 10;

/// The most chunks in a block, for the narrowest values.
const MAX_BLOCK_CHUNKS: usize = BLOCK_BYTES / CHUNK;

/// A statistic over the valid values of an integer array, computed in a single pass.
///
/// [`accumulate`] calls [`start`](Self::start) once with the first valid value, then feeds every
/// valid value, including the first, exactly once and in order. It splits the values into blocks
/// that fit in L1 and passes each to [`block`](Self::block), or with nulls to
/// [`filled_block`](Self::filled_block), then the trailing values to [`push`](Self::push).
///
/// With nulls, the driver fills each block once, replacing every null with the closest valid value
/// before it, and every accumulator shares the filled block. Repeating a valid value changes
/// neither the extrema, the bits, the runs nor the sortedness, so most statistics run their fully
/// valid kernel on a filled chunk and correct for the repeats.
///
/// The default block methods feed each chunk of [`CHUNK`] values to [`chunk`](Self::chunk), or
/// with nulls to [`filled_chunk`](Self::filled_chunk), and then call
/// [`end_block`](Self::end_block).
///
/// A tuple of accumulators passes each block to each element in turn, so each element runs its
/// own loop with its state in registers while the block stays in L1. [`Fused`] instead passes each
/// chunk to every element in one loop, for cheap statistics whose state fits in registers
/// together.
pub trait IntAccumulator<T: Copy> {
    /// The computed statistic.
    type Output;

    /// Whether [`filled_chunk`](Self::filled_chunk) reads the filled nulls. If no accumulator in
    /// a pass does, the driver skips filling and passes chunks with arbitrary values in place of
    /// nulls.
    const USES_FILL: bool = false;

    /// Initializes the state from the first valid value.
    fn start(&mut self, head: T);

    /// Accumulates [`CHUNK`] consecutive valid values.
    fn chunk(&mut self, values: &[T; CHUNK]);

    /// Accumulates the values of a chunk with nulls, whose bits are set in `valid`, least
    /// significant bit first.
    ///
    /// If [`USES_FILL`](Self::USES_FILL), `filled` holds the chunk with each null replaced by the
    /// closest valid value before it, so a null repeats its predecessor, or the last valid value
    /// before the chunk. Otherwise the nulls hold arbitrary values. The default pushes the valid
    /// values.
    #[inline(always)]
    fn filled_chunk(&mut self, filled: &[T; CHUNK], valid: u64) {
        push_set_bits(self, filled, valid);
    }

    /// Accumulates one valid value.
    fn push(&mut self, value: T);

    /// Accumulates a block of fully valid chunks.
    #[inline(always)]
    fn block(&mut self, chunks: &[[T; CHUNK]]) {
        for chunk in chunks {
            self.chunk(chunk);
        }
        self.end_block();
    }

    /// Accumulates a block of filled chunks with one validity word per chunk. See
    /// [`filled_chunk`](Self::filled_chunk).
    #[inline(always)]
    fn filled_block(&mut self, filled: &[[T; CHUNK]], valid: &[u64]) {
        for (chunk, &word) in filled.iter().zip(valid) {
            match word {
                0 => {}
                u64::MAX => self.chunk(chunk),
                _ => self.filled_chunk(chunk, word),
            }
        }
        self.end_block();
    }

    /// Called after each block, for example to flush narrow per-block lanes.
    #[inline(always)]
    fn end_block(&mut self) {}

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
    let block_chunks = BLOCK_BYTES / size_of::<T>() / CHUNK;
    match validity.bit_buffer() {
        AllOr::None => return false,
        AllOr::All => {
            let Some(&head) = values.first() else {
                return false;
            };
            acc.start(head);
            let (chunks, remainder) = values.as_chunks::<CHUNK>();
            for block in chunks.chunks(block_chunks) {
                acc.block(block);
            }
            for &value in remainder {
                acc.push(value);
            }
        }
        AllOr::Some(bits) => {
            let Some(head) = validity.first() else {
                return false;
            };
            // The last valid value seen, which fills the nulls that lead a chunk.
            let mut prev = values[head];
            acc.start(prev);
            let bit_chunks = bits.chunks();
            let mut words = bit_chunks.iter();
            let (chunks, remainder) = values.as_chunks::<CHUNK>();
            let mut block_words = [0u64; MAX_BLOCK_CHUNKS];
            let mut filled: Vec<[T; CHUNK]> = Vec::with_capacity(block_chunks.min(chunks.len()));
            for block in chunks.chunks(block_chunks) {
                let block_words = &mut block_words[..block.len()];
                for (slot, word) in block_words.iter_mut().zip(&mut words) {
                    *slot = word;
                }
                if block_words.iter().all(|&word| word == u64::MAX) {
                    acc.block(block);
                    prev = block[block.len() - 1][CHUNK - 1];
                } else if !A::USES_FILL {
                    if block_words.iter().any(|&word| word != 0) {
                        acc.filled_block(block, block_words);
                    }
                } else if block_words.iter().any(|&word| word != 0) {
                    filled.clear();
                    filled.extend_from_slice(block);
                    for (chunk, &word) in filled.iter_mut().zip(block_words.iter()) {
                        if word != 0 {
                            forward_fill(chunk, word, prev);
                            prev = chunk[CHUNK - 1];
                        }
                    }
                    acc.filled_block(&filled, block_words);
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

/// Replaces each null of `chunk`, whose bit is unset in `valid`, with the closest valid value
/// before it, or with `prev`, the last valid value before the chunk. The cost is one store per
/// null.
#[inline(always)]
fn forward_fill<T: Copy>(chunk: &mut [T; CHUNK], valid: u64, prev: T) {
    let mut nulls = !valid;
    while nulls != 0 {
        let i = nulls.trailing_zeros() as usize;
        chunk[i] = if i == 0 { prev } else { chunk[i - 1] };
        nulls &= nulls - 1;
    }
}

/// Returns the indices of the nulls of a chunk, whose bits are unset in `valid`.
#[inline(always)]
fn null_indices(valid: u64) -> impl Iterator<Item = usize> {
    let mut nulls = !valid;
    std::iter::from_fn(move || {
        (nulls != 0).then(|| {
            let i = nulls.trailing_zeros() as usize;
            nulls &= nulls - 1;
            i
        })
    })
}

/// Implements [`IntAccumulator`] and [`ErasedAccumulator`] for a tuple of accumulators by forwarding to each element.
macro_rules! impl_tuple_accumulator {
    ($($name:ident),+) => {
        impl<T: Copy, $($name: IntAccumulator<T>),+> IntAccumulator<T> for ($($name,)+) {
            type Output = ($($name::Output,)+);

            const USES_FILL: bool = false $(|| $name::USES_FILL)+;

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
            fn filled_chunk(&mut self, filled: &[T; CHUNK], valid: u64) {
                #[allow(non_snake_case)]
                let ($($name,)+) = self;
                $($name.filled_chunk(filled, valid);)+
            }

            #[inline(always)]
            fn push(&mut self, value: T) {
                #[allow(non_snake_case)]
                let ($($name,)+) = self;
                $($name.push(value);)+
            }

            #[inline(always)]
            fn block(&mut self, chunks: &[[T; CHUNK]]) {
                #[allow(non_snake_case)]
                let ($($name,)+) = self;
                $($name.block(chunks);)+
            }

            #[inline(always)]
            fn filled_block(&mut self, filled: &[[T; CHUNK]], valid: &[u64]) {
                #[allow(non_snake_case)]
                let ($($name,)+) = self;
                $($name.filled_block(filled, valid);)+
            }

            #[inline(always)]
            fn end_block(&mut self) {
                #[allow(non_snake_case)]
                let ($($name,)+) = self;
                $($name.end_block();)+
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

/// Accumulates every element of `A`, typically a tuple, in one loop per block, passing each chunk
/// to every element in turn.
///
/// This saves loop overhead and repeated loads for cheap statistics, as long as their state fits
/// in registers together. Otherwise the state spills each chunk, and a plain tuple, which runs
/// one loop per element over each block, is faster.
#[derive(Debug, Clone, Copy, Default)]
pub struct Fused<A>(pub A);

impl<T: Copy, A: IntAccumulator<T>> IntAccumulator<T> for Fused<A> {
    type Output = A::Output;

    const USES_FILL: bool = A::USES_FILL;

    #[inline(always)]
    fn start(&mut self, head: T) {
        self.0.start(head);
    }

    #[inline(always)]
    fn chunk(&mut self, values: &[T; CHUNK]) {
        self.0.chunk(values);
    }

    #[inline(always)]
    fn filled_chunk(&mut self, filled: &[T; CHUNK], valid: u64) {
        self.0.filled_chunk(filled, valid);
    }

    #[inline(always)]
    fn push(&mut self, value: T) {
        self.0.push(value);
    }

    #[inline(always)]
    fn end_block(&mut self) {
        self.0.end_block();
    }

    #[inline]
    fn finish(self) -> A::Output {
        self.0.finish()
    }
}

impl<T: Copy, A: ErasedAccumulator<T>> ErasedAccumulator<T> for Fused<A> {
    #[inline]
    fn finish_into(self, stats: &mut IntStats) {
        self.0.finish_into(stats);
    }
}

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
