// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Masked reduction kernels: fold every lane of an [`IndexedSource`] into one accumulator.

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
    fn fold_masked_of_all_null_lanes_is_init() {
        let values = [3i64; 70];
        let mask = BitBuffer::new_unset(70);

        assert_eq!(values[..].fold_masked(&mask, 0, 11, |acc, v| acc + v), 11);
    }
}
