// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Reduction kernels: fold every lane of an [`IndexedSource`] into one accumulator.
//!
//! [`IndexedFoldExt::fold_masked`] folds in lane order into a single accumulator, which vectorizes
//! only when the compiler may reorder the fold, such as an integer sum or minimum.
//! [`IndexedFoldExt::fold_lanes`] and [`IndexedFoldExt::fold_lanes_masked`] fold into independent
//! per-lane accumulators and merge them at the end, which vectorizes any fold, such as a float sum.

use vortex_buffer::BitBuffer;

use crate::lane_kernels::source::IndexedSource;

/// Extension trait providing reduction kernels on any [`IndexedSource`].
///
/// Bring the trait into scope (`use vortex_compute::lane_kernels::IndexedFoldExt;`) to call
/// them with method syntax: `values.fold_masked(&mask, i32::MAX, i32::MAX, |acc, v| acc.min(v))`.
pub trait IndexedFoldExt: IndexedSource + Sized {
    /// Fold every lane into `init` with `f`, reading null lanes as `null`.
    ///
    /// `null` must be an identity of `f`, such as zero for a sum or the type's maximum for a
    /// minimum, so that null lanes leave the accumulator unchanged. Every lane of a 64-lane chunk
    /// is read and selected against its validity bit, with no branch per lane, so a reduction the
    /// compiler can reassociate, such as an integer sum, minimum or maximum, vectorizes the same
    /// way as an unmasked one. Runs of 32 lanes without a valid lane are skipped.
    ///
    /// # Panics
    ///
    /// Panics if `self.len() != mask.len()`.
    #[inline]
    fn fold_masked<A, F>(self, mask: &BitBuffer, null: Self::Item, init: A, f: F) -> A
    where
        Self::Item: Copy,
        F: Fn(A, Self::Item) -> A,
    {
        let values = self;
        let len = values.len();
        assert_eq!(len, mask.len(), "values and mask must have the same length");

        let chunks = mask.chunks();
        let mut acc = init;
        for (chunk_idx, word) in chunks.iter().enumerate() {
            let base = chunk_idx * 64;
            for (half, bits) in halves(word) {
                if bits != 0 {
                    acc = fold_lanes(&values, null, acc, &f, bits, base + half, 32);
                }
            }
        }
        let remainder = len % 64;
        let word = chunks.remainder_bits();
        let base = len - remainder;
        for (half, bits) in halves(word) {
            let lanes = remainder.saturating_sub(half).min(32);
            if bits != 0 && lanes != 0 {
                acc = fold_lanes(&values, null, acc, &f, bits, base + half, lanes);
            }
        }
        acc
    }

    /// Fold every lane into `LANES` independent accumulators with `f`, then combine them with
    /// `merge`.
    ///
    /// Lane `i` goes into accumulator `i % LANES`, so each accumulator is a separate chain and one
    /// row of `LANES` lanes is a single vector step. The compiler needs no permission to reorder
    /// `f`, so a fold it cannot reassociate, such as a float sum, a float minimum or an integer
    /// sum that also collects overflow evidence, vectorizes the same way as an integer sum. The
    /// result can differ from an in-order fold only in the order `f` sees the lanes, such as the
    /// rounding of a float sum.
    ///
    /// `init` starts every accumulator, so it must be an identity of `merge`. Pick `LANES` so that
    /// the accumulators fill a few vector registers: `LANES * size_of::<A>()` of 64 to 128 bytes
    /// hides the latency of `f` on NEON, AVX2 and AVX-512 alike. `LANES` must divide 64.
    ///
    /// Prefer an accumulator of one primitive. A pair of integers vectorizes, but a pair of
    /// floats, such as a minimum and a maximum, stays scalar on NEON and spills on SSE2, where two
    /// folds of one float each vectorize on every target.
    #[inline]
    fn fold_lanes<const LANES: usize, A, F, M>(self, init: A, f: F, merge: M) -> A
    where
        A: Copy,
        F: Fn(A, Self::Item) -> A,
        M: Fn(A, A) -> A,
    {
        const { assert!(LANES > 0 && 64 % LANES == 0, "LANES must divide 64") };

        let values = self;
        let len = values.len();

        // Short of one row, such as a small group of a grouped aggregate, an in-order fold into one
        // accumulator skips the merge of the others, which `init` leaves unchanged.
        if len < LANES {
            let mut acc = init;
            for i in 0..len {
                // SAFETY: i < len.
                acc = f(acc, unsafe { values.get_unchecked(i) });
            }
            return acc;
        }

        let mut acc = [init; LANES];
        let rows = len / LANES;
        for row in 0..rows {
            row_barrier();
            fold_row(&values, &mut acc, &f, row * LANES, LANES);
        }
        fold_row(&values, &mut acc, &f, rows * LANES, len % LANES);

        merge_lanes(acc, merge)
    }

    /// Fold the valid lanes into `LANES` independent accumulators with `f`, reading null lanes as
    /// `null`, then combine the accumulators with `merge`.
    ///
    /// This is [`fold_lanes`] with a validity mask. `null` must be an identity of `f`, as in
    /// [`fold_masked`]. Each lane is selected against a constant bit of its row of the validity
    /// word, with no branch per lane and no per-lane variable shift, so the fold vectorizes on
    /// every target at the same cost at any validity density. Words with no valid lane are
    /// skipped.
    ///
    /// [`fold_lanes`]: IndexedFoldExt::fold_lanes
    /// [`fold_masked`]: IndexedFoldExt::fold_masked
    ///
    /// # Panics
    ///
    /// Panics if `self.len() != mask.len()`.
    #[inline]
    fn fold_lanes_masked<const LANES: usize, A, F, M>(
        self,
        mask: &BitBuffer,
        null: Self::Item,
        init: A,
        f: F,
        merge: M,
    ) -> A
    where
        Self::Item: Copy,
        A: Copy,
        F: Fn(A, Self::Item) -> A,
        M: Fn(A, A) -> A,
    {
        const { assert!(LANES > 0 && 64 % LANES == 0, "LANES must divide 64") };

        let values = self;
        let len = values.len();
        assert_eq!(len, mask.len(), "values and mask must have the same length");

        let chunks = mask.chunks();
        let mut acc = [init; LANES];

        for (chunk_idx, word) in chunks.iter().enumerate() {
            if word == 0 {
                continue;
            }

            let base = chunk_idx * 64;
            for row in 0..64 / LANES {
                row_barrier();
                let bits = word >> (row * LANES);
                fold_row_masked(&values, null, &mut acc, &f, bits, base + row * LANES, LANES);
            }
        }

        let remainder = len % 64;
        let word = chunks.remainder_bits();
        if word != 0 {
            let base = len - remainder;
            for row in 0..remainder.div_ceil(LANES) {
                let lanes = (remainder - row * LANES).min(LANES);
                let bits = word >> (row * LANES);
                fold_row_masked(&values, null, &mut acc, &f, bits, base + row * LANES, lanes);
            }
        }

        merge_lanes(acc, merge)
    }
}

/// Keep the vectorizers from packing one lane of several rows.
///
/// Each accumulator of [`IndexedFoldExt::fold_lanes`] is a reduction over the rows, and the
/// vectorizers take a reduction they recognize, such as an integer minimum or sum, and vectorize it
/// across rows: one vector holds the same lane of several rows and fills it with a strided load of
/// each lane. The loop vectorizer does so over the loop of rows, and the SLP vectorizer over the
/// rows of a 64-lane chunk, which the masked fold unrolls. The opaque barrier before each row is a
/// point no load moves across, which leaves each row to the SLP vectorizer to pack the row's
/// contiguous lanes into full vectors. It emits no instruction.
#[allow(clippy::inline_always)]
#[inline(always)]
fn row_barrier() {
    std::hint::black_box(());
}

/// Fold `lanes` lanes from `base`, lane `i` into `acc[i]`.
#[allow(clippy::inline_always)]
#[inline(always)]
fn fold_row<S, A, F, const LANES: usize>(
    values: &S,
    acc: &mut [A; LANES],
    f: &F,
    base: usize,
    lanes: usize,
) where
    S: IndexedSource,
    A: Copy,
    F: Fn(A, S::Item) -> A,
{
    for (lane, acc) in acc.iter_mut().enumerate().take(lanes) {
        // SAFETY: the caller guarantees base + lanes <= len.
        let value = unsafe { values.get_unchecked(base + lane) };
        *acc = f(*acc, value);
    }
}

/// Fold `lanes` lanes from `base`, lane `i` into `acc[i]`, reading the lanes whose bit in `bits`
/// is unset as `null`.
///
/// Lanes of at most 32 bits test a 32-bit copy of the row bits, so the lane mask is as wide as the
/// lane and needs no narrowing. Wider lanes, and rows wider than 32 lanes, test all 64 bits, except
/// where [`NARROW_WIDE_LANES`] says otherwise.
#[allow(clippy::inline_always)]
#[inline(always)]
fn fold_row_masked<S, A, F, const LANES: usize>(
    values: &S,
    null: S::Item,
    acc: &mut [A; LANES],
    f: &F,
    bits: u64,
    base: usize,
    lanes: usize,
) where
    S: IndexedSource,
    S::Item: Copy,
    A: Copy,
    F: Fn(A, S::Item) -> A,
{
    #[expect(
        clippy::cast_possible_truncation,
        reason = "rows of at most 32 lanes keep their low 32 bits"
    )]
    let narrow_bits = bits as u32;
    let narrow = LANES <= 32 && (size_of::<S::Item>() <= 4 || NARROW_WIDE_LANES);

    for (lane, acc) in acc.iter_mut().enumerate().take(lanes) {
        // SAFETY: the caller guarantees base + lanes <= len.
        let value = unsafe { values.get_unchecked(base + lane) };
        let valid = if narrow {
            narrow_bits & LANE_BITS[lane % 32] != 0
        } else {
            bits & WIDE_LANE_BITS[lane] != 0
        };
        *acc = f(*acc, if valid { value } else { null });
    }
}

/// Combine the accumulators pairwise, halving their count each step.
#[allow(clippy::inline_always)]
#[inline(always)]
fn merge_lanes<A, M, const LANES: usize>(mut acc: [A; LANES], merge: M) -> A
where
    A: Copy,
    M: Fn(A, A) -> A,
{
    let mut width = LANES;
    while width > 1 {
        width /= 2;
        for lane in 0..width {
            acc[lane] = merge(acc[lane], acc[lane + width]);
        }
    }
    acc[0]
}

/// The two 32-lane halves of a 64-lane validity word, each with the offset of its first lane.
#[expect(
    clippy::cast_possible_truncation,
    reason = "each half keeps exactly its 32 bits"
)]
fn halves(word: u64) -> [(usize, u32); 2] {
    [(0, word as u32), (32, (word >> 32) as u32)]
}

/// `LANE_BITS[i]` selects lane `i` of a 32-lane validity word. Testing a lane against a constant
/// bit, rather than shifting the word by the lane index, needs no per-lane variable shift, which
/// baseline x86-64 lacks, so the selection vectorizes on every target.
const LANE_BITS: [u32; 32] = {
    let mut bits = [0; 32];
    let mut lane = 0;
    while lane < 32 {
        bits[lane] = 1 << lane;
        lane += 1;
    }
    bits
};

/// Whether 64-bit lanes test 32-bit row bits and widen the lane mask. x86-64 below SSE4.1 has no
/// 64-bit vector compare, which makes the widening the cheaper of the two, by up to a third on
/// masked `f64` folds. Every other target compares 64-bit lanes faster.
const NARROW_WIDE_LANES: bool = cfg!(all(target_arch = "x86_64", not(target_feature = "sse4.1")));

/// `WIDE_LANE_BITS[i]` selects lane `i` of a 64-lane validity word, as [`LANE_BITS`] does for a
/// 32-lane one.
const WIDE_LANE_BITS: [u64; 64] = {
    let mut bits = [0; 64];
    let mut lane = 0;
    while lane < 64 {
        bits[lane] = 1 << lane;
        lane += 1;
    }
    bits
};

/// Fold `lanes` lanes from `base`, reading the lanes whose bit in `bits` is unset as `null`.
#[allow(clippy::inline_always)]
#[inline(always)]
fn fold_lanes<S, A, F>(
    values: &S,
    null: S::Item,
    mut acc: A,
    f: &F,
    bits: u32,
    base: usize,
    lanes: usize,
) -> A
where
    S: IndexedSource,
    S::Item: Copy,
    F: Fn(A, S::Item) -> A,
{
    for (lane, lane_bit) in LANE_BITS.iter().enumerate().take(lanes) {
        // SAFETY: the caller guarantees base + lanes <= len.
        let value = unsafe { values.get_unchecked(base + lane) };
        acc = f(acc, if bits & lane_bit != 0 { value } else { null });
    }
    acc
}

impl<S: IndexedSource> IndexedFoldExt for S {}

#[cfg(test)]
mod tests {
    use vortex_buffer::BitBuffer;

    use super::IndexedFoldExt;
    use crate::lane_kernels::LaneZip;

    #[test]
    fn fold_masked_reads_null_lanes_as_identity() {
        let values: Vec<i32> = (0..130).map(|v| v - 65).collect();
        let mask = BitBuffer::from_iter((0..130).map(|i| i % 3 == 0));
        let expected: i64 = values
            .iter()
            .enumerate()
            .filter(|(i, _)| i % 3 == 0)
            .map(|(_, &v)| i64::from(v))
            .sum();

        let sum = values[..].fold_masked(&mask, 0, 0i64, |acc, v| acc + i64::from(v));
        let min = values[..].fold_masked(&mask, i32::MAX, i32::MAX, |acc, v| acc.min(v));

        assert_eq!(sum, expected);
        assert_eq!(min, -65);
    }

    #[test]
    fn fold_masked_handles_unaligned_masks() {
        let values = [5u8, 1, 9, 2, 7];
        let mask = BitBuffer::from_iter([false, true, true, false, true, true, false]).slice(2..7);

        let max = values[..].fold_masked(&mask, u8::MIN, u8::MIN, |acc, v| acc.max(v));

        // Valid lanes 0, 2 and 3 hold 5, 9 and 2.
        assert_eq!(max, 9);
    }

    #[test]
    fn fold_lanes_matches_an_in_order_fold() {
        for len in [0, 1, 7, 8, 63, 64, 65, 130, 1000] {
            let values: Vec<i64> = (0..len).map(|v| v * 7 - 300).collect();
            let expected: i64 = values.iter().sum();

            assert_eq!(values[..].fold_lanes::<8, _, _, _>(0, |a, v| a + v, |a, b| a + b), expected);
            assert_eq!(values[..].fold_lanes::<64, _, _, _>(0, |a, v| a + v, |a, b| a + b), expected);
        }
    }

    #[test]
    fn fold_lanes_masked_reads_null_lanes_as_identity() {
        for len in [0, 1, 31, 33, 64, 65, 130, 1000] {
            let values: Vec<i32> = (0..len).map(|v| v - 65).collect();
            let mask = BitBuffer::from_iter((0..len).map(|i| i % 3 == 0 || i % 7 == 0));
            let expected: i64 = values
                .iter()
                .enumerate()
                .filter(|(i, _)| mask.value(*i))
                .map(|(_, &v)| i64::from(v))
                .sum();

            let narrow =
                values[..].fold_lanes_masked::<16, _, _, _>(&mask, 0, 0, |a, v| a + v, |a, b| a + b);
            let wide = values[..].fold_lanes_masked::<64, _, _, _>(
                &mask,
                0,
                0i64,
                |a, v| a + i64::from(v),
                |a, b| a + b,
            );

            assert_eq!(i64::from(narrow), expected);
            assert_eq!(wide, expected);
        }
    }

    #[test]
    fn fold_lanes_masked_handles_unaligned_masks() {
        let values = [5.0f64, 1.0, 9.0, 2.0, 7.0];
        let mask = BitBuffer::from_iter([false, true, true, false, true, true, false]).slice(2..7);

        let max = values[..].fold_lanes_masked::<4, _, _, _>(
            &mask,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
            f64::max,
            f64::max,
        );

        // Valid lanes 0, 2 and 3 hold 5, 9 and 2.
        assert_eq!(max, 9.0);
    }

    #[test]
    fn fold_lanes_drives_a_lane_zip() {
        let a: Vec<i64> = (0..100).collect();
        let b: Vec<i64> = (0..100).map(|v| 100 - v).collect();
        let expected: i64 = a.iter().zip(&b).map(|(x, y)| x * y).sum();

        let dot = LaneZip::new(&a[..], &b[..]).fold_lanes::<8, _, _, _>(
            0,
            |acc, (x, y)| acc + x * y,
            |x, y| x + y,
        );

        assert_eq!(dot, expected);
    }

    #[test]
    fn fold_masked_of_all_null_lanes_is_init() {
        let values = [3i64; 70];
        let mask = BitBuffer::new_unset(70);

        assert_eq!(values[..].fold_masked(&mask, 0, 11, |acc, v| acc + v), 11);
    }
}
