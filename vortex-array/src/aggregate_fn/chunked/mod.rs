// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Aggregates over primitive values, computed a chunk of 64 values at a time.
//!
//! A chunk's validity is one word of the validity bitmap. A [`ChunkAccumulator`] defines one
//! aggregate over chunks, and [`accumulate`] drives it over a slice and its validity: fully valid
//! chunks take a branch-free path, fully null chunks cost almost nothing, and only the remaining
//! chunks look at individual bits.
//!
//! A tuple of accumulators, or an [`Option`] of one, is itself an accumulator, so several
//! aggregates share one pass over the data:
//!
//! ```ignore
//! let mut acc = (MinMax::<i32>::new(), IsSorted::<i32, false>::new());
//! accumulate(values, &validity, &mut acc);
//! let (min_max, sorted) = (acc.0.finish(), acc.1.finish());
//! ```

mod float;
mod is_constant;
mod is_sorted;
mod min_max;
mod stats;
mod sum;
#[cfg(test)]
mod tests;

pub use float::FloatKey;
pub use float::FloatSum;
pub use float::Keyed;
pub use is_constant::IsConstant;
pub use is_sorted::IsSorted;
pub use min_max::MinMax;
pub(crate) use stats::compute_primitive_stats;
pub use sum::Sum;
use vortex_mask::AllOr;
use vortex_mask::Mask;

/// A totally ordered value with a lowest and a highest value, which [`MinMax`] starts from.
pub trait Extremes: Copy + Ord {
    /// The lowest value.
    const LOWEST: Self;
    /// The highest value.
    const HIGHEST: Self;
}

macro_rules! impl_extremes {
    ($($T:ty),+) => {
        $(impl Extremes for $T {
            const LOWEST: Self = <$T>::MIN;
            const HIGHEST: Self = <$T>::MAX;
        })+
    };
}

impl_extremes!(
    u8,
    u16,
    u32,
    u64,
    i8,
    i16,
    i32,
    i64,
    i128,
    crate::dtype::i256
);

/// The values per chunk: one validity word.
pub const CHUNK: usize = 64;

/// The most chunks per block: 8 KiB of 64-bit values, which stay in L1 while every accumulator of a
/// tuple reads them.
pub const BLOCK: usize = 16;

/// One aggregate over chunks of [`CHUNK`] values.
pub trait ChunkAccumulator<T: Copy> {
    /// Accumulates a chunk of valid values.
    fn chunk(&mut self, values: &[T; CHUNK]);

    /// Accumulates the first `len` values of a chunk, of which those whose bit is set in `valid`
    /// are valid. `valid` has no bits at or beyond `len`, and may be zero.
    fn partial_chunk(&mut self, values: &[T; CHUNK], valid: u64, len: usize);

    /// Accumulates consecutive chunks of valid values, at most [`BLOCK`] of them.
    ///
    /// A tuple hands each member a whole block, so that each runs its own loop over it.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn block(&mut self, chunks: &[[T; CHUNK]]) {
        for chunk in chunks {
            self.chunk(chunk);
        }
    }

    /// Whether the result is already decided, so that the rest of the input can be skipped.
    fn is_done(&self) -> bool {
        false
    }
}

/// Accumulates `values`, whose validity is `validity`, into `acc`.
///
/// Not inlined: the accumulator then stays behind a reference, so that lane state such as
/// [`MinMax`]'s is folded with vector loads and stores rather than split into scalars.
#[inline(never)]
pub fn accumulate<T: Copy, A: ChunkAccumulator<T>>(values: &[T], validity: &Mask, acc: &mut A) {
    debug_assert_eq!(values.len(), validity.len());
    let (chunks, remainder) = values.as_chunks::<CHUNK>();

    let remainder_valid = match validity.bit_buffer() {
        AllOr::All => {
            for block in chunks.chunks(BLOCK) {
                if acc.is_done() {
                    return;
                }
                acc.block(block);
            }
            low_bits(remainder.len())
        }
        AllOr::None => {
            for chunk in chunks {
                if acc.is_done() {
                    return;
                }
                acc.partial_chunk(chunk, 0, CHUNK);
            }
            0
        }
        AllOr::Some(bits) => {
            let words = bits.chunks();
            // The start of the run of valid chunks not yet accumulated.
            let mut start = 0;
            for (i, word) in words.iter().enumerate() {
                if word == u64::MAX {
                    if i + 1 - start == BLOCK {
                        if acc.is_done() {
                            return;
                        }
                        acc.block(&chunks[start..=i]);
                        start = i + 1;
                    }
                    continue;
                }
                if acc.is_done() {
                    return;
                }
                if start < i {
                    acc.block(&chunks[start..i]);
                }
                acc.partial_chunk(&chunks[i], word, CHUNK);
                start = i + 1;
            }
            if start < chunks.len() && !acc.is_done() {
                acc.block(&chunks[start..]);
            }
            words.remainder_bits()
        }
    };

    if let Some(&first) = remainder.first()
        && !acc.is_done()
    {
        let mut last = [first; CHUNK];
        last[..remainder.len()].copy_from_slice(remainder);
        acc.partial_chunk(&last, remainder_valid, remainder.len());
    }
}

/// Returns a word whose low `len` bits are set.
#[inline]
pub(crate) const fn low_bits(len: usize) -> u64 {
    if len >= CHUNK {
        u64::MAX
    } else {
        (1 << len) - 1
    }
}

/// Returns `values` with each null, whose bit is unset in `valid`, replaced by the closest valid
/// value before it, or by `fill` if there is none.
#[allow(clippy::inline_always)]
#[inline(always)]
pub(crate) fn forward_fill<T: Copy>(values: &[T; CHUNK], valid: u64, fill: T) -> [T; CHUNK] {
    let mut filled = *values;
    let mut nulls = !valid;
    while nulls != 0 {
        let i = nulls.trailing_zeros() as usize;
        filled[i] = if i == 0 { fill } else { filled[i - 1] };
        nulls &= nulls - 1;
    }
    filled
}

/// An absent accumulator does nothing, so that the aggregates of a tuple can be chosen at runtime.
impl<T: Copy, A: ChunkAccumulator<T>> ChunkAccumulator<T> for Option<A> {
    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn chunk(&mut self, values: &[T; CHUNK]) {
        if let Some(acc) = self {
            acc.chunk(values);
        }
    }

    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn partial_chunk(&mut self, values: &[T; CHUNK], valid: u64, len: usize) {
        if let Some(acc) = self {
            acc.partial_chunk(values, valid, len);
        }
    }

    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn block(&mut self, chunks: &[[T; CHUNK]]) {
        if let Some(acc) = self {
            acc.block(chunks);
        }
    }

    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn is_done(&self) -> bool {
        self.as_ref().is_none_or(A::is_done)
    }
}

/// Implements [`ChunkAccumulator`] for a tuple, whose members each see every chunk until they are
/// done.
macro_rules! impl_tuple {
    ($($A:ident $i:tt),+) => {
        impl<T: Copy, $($A: ChunkAccumulator<T>),+> ChunkAccumulator<T> for ($($A,)+) {
            #[allow(clippy::inline_always)]
            #[inline(always)]
            fn chunk(&mut self, values: &[T; CHUNK]) {
                $(
                    if !self.$i.is_done() {
                        self.$i.chunk(values);
                    }
                )+
            }

            #[allow(clippy::inline_always)]
            #[inline(always)]
            fn partial_chunk(&mut self, values: &[T; CHUNK], valid: u64, len: usize) {
                $(
                    if !self.$i.is_done() {
                        self.$i.partial_chunk(values, valid, len);
                    }
                )+
            }

            #[allow(clippy::inline_always)]
            #[inline(always)]
            fn block(&mut self, chunks: &[[T; CHUNK]]) {
                $(
                    if !self.$i.is_done() {
                        self.$i.block(chunks);
                    }
                )+
            }

            #[allow(clippy::inline_always)]
            #[inline(always)]
            fn is_done(&self) -> bool {
                $(self.$i.is_done())&&+
            }
        }
    };
}

impl_tuple!(A 0, B 1);
impl_tuple!(A 0, B 1, C 2);
impl_tuple!(A 0, B 1, C 2, D 3);
impl_tuple!(A 0, B 1, C 2, D 3, E 4);
impl_tuple!(A 0, B 1, C 2, D 3, E 4, F 5);
