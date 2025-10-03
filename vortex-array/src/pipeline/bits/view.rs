// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::{Debug, Formatter};

use bitvec::prelude::*;
use vortex_error::{VortexError, VortexResult, vortex_err};

use crate::pipeline::{N, N_WORDS};

/// A borrowed fixed-size bit vector of length `N` bits, represented as an array of usize words.
///
/// Internally, it uses a [`BitArray`] to store the bits, but this crate has some
/// performance foot-guns in cases where we can lean on better assumptions, and therefore we wrap
/// it up for use within Vortex.
/// Read-only view into a bit array for selection masking in operator operations.
#[derive(Clone, Copy)]
pub struct BitView<'a> {
    bits: &'a BitArray<[usize; N_WORDS], Lsb0>,
    // TODO(ngates): we may want to expose this for optimizations.
    // If set to Selection::Prefix, then all true bits are at the start of the array.
    // selection: Selection,
    true_count: usize,
}

impl Debug for BitView<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BitView")
            .field("true_count", &self.true_count)
            .field("bits", &self.as_raw())
            .finish()
    }
}

impl BitView<'static> {
    pub fn all_true() -> Self {
        static ALL_TRUE: [usize; N_WORDS] = [usize::MAX; N_WORDS];
        unsafe {
            BitView::new_unchecked(
                std::mem::transmute::<&[usize; N_WORDS], &BitArray<[usize; N_WORDS], Lsb0>>(
                    &ALL_TRUE,
                ),
                N,
            )
        }
    }

    pub fn all_false() -> Self {
        static ALL_FALSE: [usize; N_WORDS] = [0; N_WORDS];
        unsafe {
            BitView::new_unchecked(
                std::mem::transmute::<&[usize; N_WORDS], &BitArray<[usize; N_WORDS], Lsb0>>(
                    &ALL_FALSE,
                ),
                0,
            )
        }
    }
}

impl<'a> BitView<'a> {
    pub fn new(bits: &[usize; N_WORDS]) -> Self {
        let true_count = bits.iter().map(|&word| word.count_ones() as usize).sum();
        let bits: &BitArray<[usize; N_WORDS], Lsb0> = unsafe {
            std::mem::transmute::<&[usize; N_WORDS], &BitArray<[usize; N_WORDS], Lsb0>>(bits)
        };
        BitView { bits, true_count }
    }

    pub(crate) unsafe fn new_unchecked(
        bits: &'a BitArray<[usize; N_WORDS], Lsb0>,
        true_count: usize,
    ) -> Self {
        BitView { bits, true_count }
    }

    /// Returns the number of `true` bits in the view.
    pub fn true_count(&self) -> usize {
        self.true_count
    }

    /// Runs the provided function `f` for each index of a `true` bit in the view.
    pub fn iter_ones<F>(&self, mut f: F)
    where
        F: FnMut(usize),
    {
        match self.true_count {
            0 => {}
            N => (0..N).for_each(&mut f),
            _ => {
                let mut bit_idx = 0;
                for mut raw in self.bits.into_inner() {
                    while raw != 0 {
                        let bit_pos = raw.trailing_zeros();
                        f(bit_idx + bit_pos as usize);
                        raw &= raw - 1; // Clear the bit at `bit_pos`
                    }
                    bit_idx += usize::BITS as usize;
                }
            }
        }
    }

    /// Runs the provided function `f` for each index of a `true` bit in the view.
    pub fn try_iter_ones<F>(&self, mut f: F) -> VortexResult<()>
    where
        F: FnMut(usize) -> VortexResult<()>,
    {
        match self.true_count {
            0 => Ok(()),
            N => {
                for i in 0..N {
                    f(i)?;
                }
                Ok(())
            }
            _ => {
                let mut bit_idx = 0;
                for mut raw in self.bits.into_inner() {
                    while raw != 0 {
                        let bit_pos = raw.trailing_zeros();
                        f(bit_idx + bit_pos as usize)?;
                        raw &= raw - 1; // Clear the bit at `bit_pos`
                    }
                    bit_idx += usize::BITS as usize;
                }
                Ok(())
            }
        }
    }

    /// Runs the provided function `f` for each index of a `true` bit in the view.
    pub fn iter_zeros<F>(&self, mut f: F)
    where
        F: FnMut(usize),
    {
        match self.true_count {
            0 => (0..N).for_each(&mut f),
            N => {}
            _ => {
                let mut bit_idx = 0;
                for mut raw in self.bits.into_inner() {
                    while raw != usize::MAX {
                        let bit_pos = raw.trailing_ones();
                        f(bit_idx + bit_pos as usize);
                        raw |= 1usize << bit_pos; // Set the zero bit to 1
                    }
                    bit_idx += usize::BITS as usize;
                }
            }
        }
    }

    /// Runs the provided function `f` for each range of `true` bits in the view.
    ///
    /// The function `f` receives a tuple `(start, len)` where `start` is the index of the first
    /// `true` bit and `len` is the number of consecutive `true` bits.
    pub fn iter_slices<F>(&self, mut f: F)
    where
        F: FnMut((usize, usize)),
    {
        match self.true_count {
            0 => {}
            N => f((0, N)),
            _ => {
                let mut bit_idx = 0;
                for mut raw in self.bits.into_inner() {
                    let mut offset = 0;
                    while raw != 0 {
                        // Skip leading zeros first
                        let zeros = raw.leading_zeros();
                        offset += zeros;
                        raw <<= zeros;

                        if offset >= 64 {
                            break;
                        }

                        // Count leading ones
                        let ones = raw.leading_ones();
                        if ones > 0 {
                            f((bit_idx + offset as usize, ones as usize));
                            offset += ones;
                            raw <<= ones;
                        }
                    }
                    bit_idx += usize::BITS as usize; // Move to next word
                }
            }
        }
    }

    pub fn as_raw(&self) -> &[usize; N_WORDS] {
        // It's actually remarkably hard to get a reference to the underlying array!
        let raw = self.bits.as_raw_slice();
        unsafe { &*(raw.as_ptr() as *const [usize; N_WORDS]) }
    }
}

impl<'a> From<&'a [usize; N_WORDS]> for BitView<'a> {
    fn from(value: &'a [usize; N_WORDS]) -> Self {
        Self::new(value)
    }
}

impl<'a> From<&'a BitArray<[usize; N_WORDS], Lsb0>> for BitView<'a> {
    fn from(bits: &'a BitArray<[usize; N_WORDS], Lsb0>) -> Self {
        BitView::new(unsafe {
            std::mem::transmute::<&BitArray<[usize; N_WORDS]>, &[usize; N_WORDS]>(bits)
        })
    }
}

impl<'a> TryFrom<&'a BitSlice<usize, Lsb0>> for BitView<'a> {
    type Error = VortexError;

    fn try_from(value: &'a BitSlice<usize, Lsb0>) -> Result<Self, Self::Error> {
        let bits: &BitArray<[usize; N_WORDS], Lsb0> = value
            .try_into()
            .map_err(|e| vortex_err!("Failed to convert BitSlice to BitArray: {}", e))?;
        Ok(BitView::new(unsafe {
            std::mem::transmute::<&BitArray<[usize; N_WORDS]>, &[usize; N_WORDS]>(bits)
        }))
    }
}

#[cfg(test)]
mod tests {
    use vortex_mask::Mask;

    use super::*;
    use crate::pipeline::bits::BitVector;

    #[test]
    fn test_iter_ones_empty() {
        let bits = [0usize; N_WORDS];
        let view = BitView::new(&bits);

        let mut ones = Vec::new();
        view.iter_ones(|idx| ones.push(idx));

        assert_eq!(ones, Vec::<usize>::new());
        assert_eq!(view.true_count(), 0);
    }

    #[test]
    fn test_iter_ones_all_set() {
        let view = BitView::all_true();

        let mut ones = Vec::new();
        view.iter_ones(|idx| ones.push(idx));

        assert_eq!(ones.len(), N);
        assert_eq!(ones, (0..N).collect::<Vec<_>>());
        assert_eq!(view.true_count(), N);
    }

    #[test]
    fn test_iter_zeros_empty() {
        let bits = [0usize; N_WORDS];
        let view = BitView::new(&bits);

        let mut zeros = Vec::new();
        view.iter_zeros(|idx| zeros.push(idx));

        assert_eq!(zeros.len(), N);
        assert_eq!(zeros, (0..N).collect::<Vec<_>>());
    }

    #[test]
    fn test_iter_zeros_all_set() {
        let view = BitView::all_true();

        let mut zeros = Vec::new();
        view.iter_zeros(|idx| zeros.push(idx));

        assert_eq!(zeros, Vec::<usize>::new());
    }

    #[test]
    fn test_iter_ones_single_bit() {
        let mut bits = [0usize; N_WORDS];
        bits[0] = 1; // Set bit 0 (LSB)
        let view = BitView::new(&bits);

        let mut ones = Vec::new();
        view.iter_ones(|idx| ones.push(idx));

        assert_eq!(ones, vec![0]);
        assert_eq!(view.true_count(), 1);
    }

    #[test]
    fn test_iter_zeros_single_bit_unset() {
        let mut bits = [usize::MAX; N_WORDS];
        bits[0] = usize::MAX ^ 1; // Clear bit 0 (LSB)
        let view = BitView::new(&bits);

        let mut zeros = Vec::new();
        view.iter_zeros(|idx| zeros.push(idx));

        assert_eq!(zeros, vec![0]);
    }

    #[test]
    fn test_iter_ones_multiple_bits_first_word() {
        let mut bits = [0usize; N_WORDS];
        bits[0] = 0b1010101; // Set bits 0, 2, 4, 6
        let view = BitView::new(&bits);

        let mut ones = Vec::new();
        view.iter_ones(|idx| ones.push(idx));

        assert_eq!(ones, vec![0, 2, 4, 6]);
        assert_eq!(view.true_count(), 4);
    }

    #[test]
    fn test_iter_zeros_multiple_bits_first_word() {
        let mut bits = [usize::MAX; N_WORDS];
        bits[0] = !0b1010101; // Clear bits 0, 2, 4, 6
        let view = BitView::new(&bits);

        let mut zeros = Vec::new();
        view.iter_zeros(|idx| zeros.push(idx));

        assert_eq!(zeros, vec![0, 2, 4, 6]);
    }

    #[test]
    fn test_iter_ones_across_words() {
        let mut bits = [0usize; N_WORDS];
        bits[0] = 1 << 63; // Set bit 63 of first word
        bits[1] = 1; // Set bit 0 of second word (bit 64 overall)
        bits[2] = 1 << 31; // Set bit 31 of third word (bit 159 overall)
        let view = BitView::new(&bits);

        let mut ones = Vec::new();
        view.iter_ones(|idx| ones.push(idx));

        assert_eq!(ones, vec![63, 64, 159]);
        assert_eq!(view.true_count(), 3);
    }

    #[test]
    fn test_iter_zeros_across_words() {
        let mut bits = [usize::MAX; N_WORDS];
        bits[0] = !(1 << 63); // Clear bit 63 of first word
        bits[1] = !1; // Clear bit 0 of second word (bit 64 overall)
        bits[2] = !(1 << 31); // Clear bit 31 of third word (bit 159 overall)
        let view = BitView::new(&bits);

        let mut zeros = Vec::new();
        view.iter_zeros(|idx| zeros.push(idx));

        assert_eq!(zeros, vec![63, 64, 159]);
    }

    #[test]
    fn test_lsb_bit_ordering() {
        let mut bits = [0usize; N_WORDS];
        bits[0] = 0b11111111; // Set bits 0-7 (LSB ordering)
        let view = BitView::new(&bits);

        let mut ones = Vec::new();
        view.iter_ones(|idx| ones.push(idx));

        assert_eq!(ones, vec![0, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(view.true_count(), 8);
    }

    #[test]
    fn test_iter_ones_and_zeros_complement() {
        let mut bits = [0usize; N_WORDS];
        bits[0] = 0xAAAAAAAAAAAAAAAA; // Alternating pattern
        let view = BitView::new(&bits);

        let mut ones = Vec::new();
        let mut zeros = Vec::new();
        view.iter_ones(|idx| ones.push(idx));
        view.iter_zeros(|idx| zeros.push(idx));

        // Check that ones and zeros together cover all indices
        let mut all_indices = ones.clone();
        all_indices.extend(&zeros);
        all_indices.sort_unstable();

        assert_eq!(all_indices, (0..N).collect::<Vec<_>>());

        // Check they don't overlap
        for one_idx in &ones {
            assert!(!zeros.contains(one_idx));
        }
    }

    #[test]
    fn test_all_false_static() {
        let view = BitView::all_false();

        let mut ones = Vec::new();
        let mut zeros = Vec::new();
        view.iter_ones(|idx| ones.push(idx));
        view.iter_zeros(|idx| zeros.push(idx));

        assert_eq!(ones, Vec::<usize>::new());
        assert_eq!(zeros, (0..N).collect::<Vec<_>>());
        assert_eq!(view.true_count(), 0);
    }

    #[test]
    fn test_compatibility_with_mask_all_true() {
        // Create corresponding BitView
        let view = BitView::all_true();

        // Collect ones from BitView
        let mut bitview_ones = Vec::new();
        view.iter_ones(|idx| bitview_ones.push(idx));

        // Get indices from Mask (all indices for all_true mask)
        let expected_indices: Vec<usize> = (0..N).collect();

        assert_eq!(bitview_ones, expected_indices);
        assert_eq!(view.true_count(), N);
    }

    #[test]
    fn test_compatibility_with_mask_all_false() {
        // Create corresponding BitView
        let view = BitView::all_false();

        // Collect ones from BitView
        let mut bitview_ones = Vec::new();
        view.iter_ones(|idx| bitview_ones.push(idx));

        // Collect zeros from BitView
        let mut bitview_zeros = Vec::new();
        view.iter_zeros(|idx| bitview_zeros.push(idx));

        assert_eq!(bitview_ones, Vec::<usize>::new());
        assert_eq!(bitview_zeros, (0..N).collect::<Vec<_>>());
        assert_eq!(view.true_count(), 0);
    }

    #[test]
    fn test_compatibility_with_mask_from_indices() {
        // Create a Mask from specific indices
        let indices = vec![0, 10, 20, 63, 64, 100, 500, 1023];

        // Create corresponding BitView
        let mut bits = [0usize; N_WORDS];
        for idx in &indices {
            let word_idx = idx / 64;
            let bit_idx = idx % 64;
            bits[word_idx] |= 1usize << bit_idx;
        }
        let view = BitView::new(&bits);

        // Collect ones from BitView
        let mut bitview_ones = Vec::new();
        view.iter_ones(|idx| bitview_ones.push(idx));

        assert_eq!(bitview_ones, indices);
        assert_eq!(view.true_count(), indices.len());
    }

    #[test]
    fn test_compatibility_with_mask_slices() {
        // Create a Mask from slices (ranges)
        let slices = vec![(0, 10), (100, 110), (500, 510)];

        // Create corresponding BitView
        let mut bits = [0usize; N_WORDS];
        for (start, end) in &slices {
            for idx in *start..*end {
                let word_idx = idx / 64;
                let bit_idx = idx % 64;
                bits[word_idx] |= 1usize << bit_idx;
            }
        }
        let view = BitView::new(&bits);

        // Collect ones from BitView
        let mut bitview_ones = Vec::new();
        view.iter_ones(|idx| bitview_ones.push(idx));

        // Expected indices from slices
        let mut expected_indices = Vec::new();
        for (start, end) in &slices {
            expected_indices.extend(*start..*end);
        }

        assert_eq!(bitview_ones, expected_indices);
        assert_eq!(view.true_count(), expected_indices.len());
    }

    #[test]
    fn test_mask_and_bitview_iter_match() {
        // Create a pattern with alternating bits in first word
        let mut bits = [0usize; N_WORDS];
        bits[0] = 0xAAAAAAAAAAAAAAAA; // Alternating 1s and 0s
        bits[1] = 0xFF00FF00FF00FF00; // Alternating bytes

        let view = BitView::new(&bits);

        // Collect indices from BitView
        let mut bitview_ones = Vec::new();
        view.iter_ones(|idx| bitview_ones.push(idx));

        // Create Mask from the same indices
        let mask = Mask::from_indices(N, bitview_ones.clone());

        // Verify the mask returns the same indices
        mask.iter_bools(|iter| {
            let mask_bools: Vec<bool> = iter.collect();

            // Check each bit matches
            for i in 0..N {
                let expected = bitview_ones.contains(&i);
                assert_eq!(mask_bools[i], expected, "Mismatch at index {}", i);
            }
        });
    }

    #[test]
    fn test_mask_and_bitview_all_true() {
        let mask = Mask::AllTrue(5);

        let vector = BitVector::true_until(5);

        let view = vector.as_view();

        // Collect indices from BitView
        let mut bitview_ones = Vec::new();
        view.iter_ones(|idx| bitview_ones.push(idx));

        // Collect indices from BitView
        let mask_ones = mask.iter_bools(|iter| {
            iter.enumerate()
                .filter(|(_, b)| *b)
                .map(|(i, _)| i)
                .collect::<Vec<_>>()
        });

        assert_eq!(bitview_ones, mask_ones);
    }

    #[test]
    fn test_bitview_zeros_complement_mask() {
        // Create a pattern
        let mut bits = [0usize; N_WORDS];
        bits[0] = 0b11110000111100001111000011110000;

        let view = BitView::new(&bits);

        // Collect ones and zeros from BitView
        let mut bitview_ones = Vec::new();
        let mut bitview_zeros = Vec::new();
        view.iter_ones(|idx| bitview_ones.push(idx));
        view.iter_zeros(|idx| bitview_zeros.push(idx));

        // Create masks for ones and zeros
        let ones_mask = Mask::from_indices(N, bitview_ones);
        let zeros_mask = Mask::from_indices(N, bitview_zeros);

        // Verify they are complements
        ones_mask.iter_bools(|ones_iter| {
            zeros_mask.iter_bools(|zeros_iter| {
                let ones_bools: Vec<bool> = ones_iter.collect();
                let zeros_bools: Vec<bool> = zeros_iter.collect();

                for i in 0..N {
                    // Each index should be either in ones or zeros, but not both
                    assert_ne!(
                        ones_bools[i], zeros_bools[i],
                        "Index {} should be in exactly one set",
                        i
                    );
                }
            });
        });
    }
}
