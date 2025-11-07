// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::{Debug, Formatter};

use vortex_error::VortexResult;

use crate::pipeline::{N, N_BYTES, N_WORDS};

/// A borrowed fixed-size bit vector of length `N` bits, represented as an array of usize words.
///
/// This struct is designed to provide a view over a Vortex [`vortex_buffer::BitBuffer`], therefore
/// the bit-ordering is LSB0 (least-significant-bit first).
///
/// Note that [`BitView`] does not support an offset. Therefore, bits are assumed to start at
/// index and end at index `N - 1`.
#[derive(Clone, Copy)]
pub struct BitView<'a> {
    bits: &'a [u8; N_BYTES],
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
        static ALL_TRUE: [u8; N_BYTES] = [u8::MAX; N_BYTES];
        unsafe { BitView::new_unchecked(&ALL_TRUE, N) }
    }

    pub fn all_false() -> Self {
        static ALL_FALSE: [u8; N_BYTES] = [0; N_BYTES];
        unsafe { BitView::new_unchecked(&ALL_FALSE, 0) }
    }
}

impl<'a> BitView<'a> {
    pub fn new(bits: &'a [u8; N_BYTES]) -> Self {
        let ptr = bits.as_ptr().cast::<usize>();
        let true_count = (0..N_WORDS)
            .map(|idx| unsafe { ptr.add(idx).read_unaligned().count_ones() as usize })
            .sum();
        BitView { bits, true_count }
    }

    pub(crate) unsafe fn new_unchecked(bits: &'a [u8; N_BYTES], true_count: usize) -> Self {
        BitView { bits, true_count }
    }

    /// Returns the number of `true` bits in the view.
    pub fn true_count(&self) -> usize {
        self.true_count
    }

    /// Iterate the [`BitView`] in fixed-size words.
    ///
    /// The words are loaded using unaligned loads to ensure correct bit ordering.
    /// For example, bit 0 is located in `word & 1 << 0`, bit 63 is located in `word & 1 << 63`,
    /// assuming the word size is 64 bits.
    fn iter_words(&self) -> impl Iterator<Item = usize> + '_ {
        let ptr = self.bits.as_ptr().cast::<usize>();
        // We use constant N_WORDS to trigger loop unrolling.
        (0..N_WORDS).map(move |idx| unsafe { ptr.add(idx).read_unaligned() })
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
                for mut raw in self.iter_words() {
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
                for mut raw in self.iter_words() {
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
                for mut raw in self.iter_words() {
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
    ///
    /// FIXME(ngates): this code is broken.
    pub fn iter_slices<F>(&self, mut f: F)
    where
        F: FnMut((usize, usize)),
    {
        match self.true_count {
            0 => {}
            N => f((0, N)),
            _ => {
                let mut bit_idx = 0;
                for raw in self.bits {
                    let mut raw = *raw;
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

    pub fn as_raw(&self) -> &[u8; N_BYTES] {
        self.bits
    }
}

#[cfg(test)]
mod tests {
    use bitvec::slice::BitSlice;
    use vortex_buffer::BitBufferMut;

    use super::*;

    #[test]
    fn test_bits() {
        let mut bits = BitBufferMut::new_unset(128);
        bits.set(1);
        bits.set(2);
        bits.set(3);
        bits.set(8);
        bits.set(64);
        let bits = bits.freeze();
        assert_eq!(bits.set_indices().collect::<Vec<_>>(), vec![1, 2, 3, 8, 64]);

        // Can we just transmute and pass it into bitvec crate?
        // Absolutely not is that answer.
        let slice_u64 =
            BitSlice::<u64>::from_slice(unsafe { std::mem::transmute(bits.inner().as_ref()) });
        assert_ne!(
            slice_u64.iter_ones().collect::<Vec<_>>(),
            vec![1, 2, 3, 8, 64]
        );

        // But if we have a &[u8], we can use unaligned load to pull it into the right order.
        unsafe {
            let vec_usize = (0..2)
                .map(|idx| {
                    bits.inner()
                        .as_ptr()
                        .cast::<usize>()
                        .add(idx)
                        .read_unaligned()
                })
                .collect::<Vec<_>>();
            let slice_usize = BitSlice::<usize>::from_slice(&vec_usize);
            assert_eq!(
                slice_usize.iter_ones().collect::<Vec<_>>(),
                vec![1, 2, 3, 8, 64]
            );
        }

        println!(
            "Bits: {:08b} {:08b}",
            bits.inner().as_ref()[0],
            bits.inner().as_ref()[1]
        );
    }

    #[test]
    fn test_iter_ones_empty() {
        let bits = [0; N_BYTES];
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
        let bits = [0; N_BYTES];
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
        let mut bits = [0; N_BYTES];
        bits[0] = 1; // Set bit 0 (LSB)
        let view = BitView::new(&bits);

        let mut ones = Vec::new();
        view.iter_ones(|idx| ones.push(idx));

        assert_eq!(ones, vec![0]);
        assert_eq!(view.true_count(), 1);
    }

    #[test]
    fn test_iter_zeros_single_bit_unset() {
        let mut bits = [u8::MAX; N_BYTES];
        bits[0] = u8::MAX ^ 1; // Clear bit 0 (LSB)
        let view = BitView::new(&bits);

        let mut zeros = Vec::new();
        view.iter_zeros(|idx| zeros.push(idx));

        assert_eq!(zeros, vec![0]);
    }

    #[test]
    fn test_iter_ones_multiple_bits_first_word() {
        let mut bits = [0; N_BYTES];
        bits[0] = 0b1010101; // Set bits 0, 2, 4, 6
        let view = BitView::new(&bits);

        let mut ones = Vec::new();
        view.iter_ones(|idx| ones.push(idx));

        assert_eq!(ones, vec![0, 2, 4, 6]);
        assert_eq!(view.true_count(), 4);
    }

    #[test]
    fn test_iter_zeros_multiple_bits_first_word() {
        let mut bits = [u8::MAX; N_BYTES];
        bits[0] = !0b1010101; // Clear bits 0, 2, 4, 6
        let view = BitView::new(&bits);

        let mut zeros = Vec::new();
        view.iter_zeros(|idx| zeros.push(idx));

        assert_eq!(zeros, vec![0, 2, 4, 6]);
    }

    #[test]
    fn test_iter_ones_across_words() {
        let mut bits = [0; N_BYTES];
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
        let mut bits = [u8::MAX; N_BYTES];
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
        let mut bits = [0; N_BYTES];
        bits[0] = 0b11111111; // Set bits 0-7 (LSB ordering)
        let view = BitView::new(&bits);

        let mut ones = Vec::new();
        view.iter_ones(|idx| ones.push(idx));

        assert_eq!(ones, vec![0, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(view.true_count(), 8);
    }

    #[test]
    fn test_iter_ones_and_zeros_complement() {
        let mut bits = [0; N_BYTES];
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
        let mut bits = [0; N_BYTES];
        for idx in &indices {
            let word_idx = idx / 8;
            let bit_idx = idx % 8;
            bits[word_idx] |= 1u8 << bit_idx;
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
        let mut bits = [0; N_BYTES];
        for (start, end) in &slices {
            for idx in *start..*end {
                let word_idx = idx / 8;
                let bit_idx = idx % 8;
                bits[word_idx] |= 1u8 << bit_idx;
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
}
