// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::borrow::Cow;
use std::fmt::{Debug, Formatter};

use vortex_buffer::BitBuffer;
use vortex_error::VortexResult;

use crate::pipeline::{N, N_BYTES, N_WORDS};

/// A borrowed fixed-size bit vector of length `N` bits, represented as an array of usize words.
///
/// This struct is designed to provide a view over a Vortex [`vortex_buffer::BitBuffer`], therefore
/// the bit-ordering is LSB0 (least-significant-bit first).
///
/// Note that [`BitView`] does not support an offset. Therefore, bits are assumed to start at
/// index and end at index `N - 1`.
pub struct BitView<'a> {
    bits: Cow<'a, [u8; N_BYTES]>,
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
    /// Creates a [`BitView`] from raw bits, computing the true count.
    pub fn new(bits: &'a [u8; N_BYTES]) -> Self {
        let ptr = bits.as_ptr().cast::<usize>();
        let true_count = (0..N_WORDS)
            .map(|idx| unsafe { ptr.add(idx).read_unaligned().count_ones() as usize })
            .sum();
        BitView {
            bits: Cow::Borrowed(bits),
            true_count,
        }
    }

    /// Creates a [`BitView`] from owned raw bits.
    pub fn new_owned(bits: [u8; N_BYTES]) -> Self {
        let ptr = bits.as_ptr().cast::<usize>();
        let true_count = (0..N_WORDS)
            .map(|idx| unsafe { ptr.add(idx).read_unaligned().count_ones() as usize })
            .sum();
        BitView {
            bits: Cow::Owned(bits),
            true_count,
        }
    }

    /// Creates a [`BitView`] from raw bits and a known true count.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `true_count` is correct for the provided `bits`.
    pub(crate) unsafe fn new_unchecked(bits: &'a [u8; N_BYTES], true_count: usize) -> Self {
        BitView {
            bits: Cow::Borrowed(bits),
            true_count,
        }
    }

    /// Creates a [`BitView`] from a byte slice.
    ///
    /// # Panics
    ///
    /// If the length of the slice is not equal to `N_BYTES`.
    pub fn from_slice(bits: &'a [u8]) -> Self {
        assert_eq!(bits.len(), N_BYTES);
        let bits_array = unsafe { &*(bits.as_ptr() as *const [u8; N_BYTES]) };
        BitView::new(bits_array)
    }

    /// Creates a [`BitView`] from a mutable byte array, populating it with the requested prefix
    /// of `true` bits.
    pub fn with_prefix(n_true: usize) -> Self {
        assert!(n_true <= N);

        // We're going to own our own array of bits
        let mut bits = [0u8; N_BYTES];

        // All-true words first
        let n_full_words = n_true / (usize::BITS as usize);
        let remaining_bits = n_true % (usize::BITS as usize);

        let ptr = bits.as_mut_ptr().cast::<usize>();

        // Fill the all-true words
        for word_idx in 0..n_full_words {
            unsafe { ptr.add(word_idx).write_unaligned(usize::MAX) };
        }

        // Fill the remaining bits in the next word
        if remaining_bits > 0 {
            let mask = (1usize << remaining_bits) - 1;
            unsafe { ptr.add(n_full_words).write_unaligned(mask) };
        }

        Self {
            bits: Cow::Owned(bits),
            true_count: n_true,
        }
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
    /// The function `f` receives a [`BitSlice`] containing the inclusive `start` bit as well as
    /// the length.
    pub fn iter_slices<F>(&self, mut f: F)
    where
        F: FnMut(BitSlice),
    {
        if self.true_count == 0 {
            return;
        }

        let mut abs_bit_offset: usize = 0; // Absolute bit index of the *current* word being processed
        let mut slice_start_bit: usize = 0; // Absolute start index of the run of 1s being tracked
        let mut slice_length: usize = 0; // Accumulated length of the run of 1s

        for mut word in self.iter_words() {
            match word {
                0 => {
                    // If a slice was being tracked, the run ends at the start of this word.
                    if slice_length > 0 {
                        f(BitSlice {
                            start: slice_start_bit,
                            len: slice_length,
                        });
                        slice_length = 0;
                    }
                }
                usize::MAX => {
                    // If a slice was not already open, it starts at the beginning of this word.
                    if slice_length == 0 {
                        slice_start_bit = abs_bit_offset;
                    }
                    // Extend the length by a full word (64 bits).
                    slice_length += usize::BITS as usize;
                }
                _ => {
                    while word != 0 {
                        // Find the first set bit (start of a run of 1s)
                        let zeros = word.trailing_zeros() as usize;

                        // If a run was open, and we hit a zero gap, report the finished slice
                        if slice_length > 0 && zeros > 0 {
                            f(BitSlice {
                                start: slice_start_bit,
                                len: slice_length,
                            });
                            slice_length = 0; // Reset state for a new slice
                        }

                        // Advance past the zeros
                        word >>= zeros;

                        if word == 0 {
                            break;
                        }

                        // Find the contiguous ones (the length of the current run segment)
                        let ones = word.trailing_ones() as usize;

                        // If slice_length is 0, we found the *absolute* start of a new slice.
                        if slice_length == 0 {
                            // Calculate the bit index within the *entire* mask where this run starts
                            let current_word_idx = abs_bit_offset + zeros;
                            slice_start_bit = current_word_idx;
                        }

                        // Accumulate the length of the slice
                        slice_length += ones;

                        // Advance past the ones
                        word >>= ones;
                    }
                }
            }

            abs_bit_offset += usize::BITS as usize;
        }

        if slice_length > 0 {
            f(BitSlice {
                start: slice_start_bit,
                len: slice_length,
            });
        }
    }

    pub fn as_raw(&self) -> &[u8; N_BYTES] {
        self.bits.as_ref()
    }
}

/// A slice of bits within a [`BitBuffer`].
///
/// We use this struct to avoid a common mistake of assuming the slices represent (start, end) ranges,
pub struct BitSlice {
    pub start: usize,
    pub len: usize,
}

pub trait BitViewExt {
    /// Iterate the [`BitBuffer`] in fixed-size chunks of [`BitView`].
    ///
    /// The final chunk will be filled with unset padding bits if the bit buffer's length is not
    /// a multiple of `N`.
    ///
    /// # Panics
    ///
    /// If the bit buffer's bit-offset is not zero.
    fn iter_bit_views(&self) -> impl Iterator<Item = BitView<'_>> + '_;
}

impl BitViewExt for BitBuffer {
    fn iter_bit_views(&self) -> impl Iterator<Item = BitView<'_>> + '_ {
        assert_eq!(
            self.offset(),
            0,
            "BitView iteration requires zero bit offset"
        );
        let n_views = self.len().div_ceil(N);
        BitViewIterator {
            bits: self.inner().as_ref(),
            view_idx: 0,
            n_views,
        }
    }
}

struct BitViewIterator<'a> {
    bits: &'a [u8],
    // The index of the view to be returned next
    view_idx: usize,
    // The total number of views
    n_views: usize,
}

impl<'a> Iterator for BitViewIterator<'a> {
    type Item = BitView<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.view_idx == self.n_views {
            return None;
        }

        let start_byte = self.view_idx * N_BYTES;
        let end_byte = start_byte + N_BYTES;

        let bits = if end_byte <= self.bits.len() {
            // Full view from the original bits
            BitView::from_slice(&self.bits[start_byte..end_byte])
        } else {
            // Partial view, copy to scratch
            let remaining_bytes = self.bits.len() - start_byte;
            let mut remaining = [0u8; N_BYTES];
            remaining[..remaining_bytes].copy_from_slice(&self.bits[start_byte..]);
            BitView::new_owned(remaining)
        };

        self.view_idx += 1;
        Some(bits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn test_with_prefix() {
        assert_eq!(BitView::with_prefix(0).true_count(), 0);

        // May as well test all the possible prefix lengths!
        for i in 1..N {
            let view = BitView::with_prefix(i);

            // Collect slices (there should be one slice from 0 to n_true)
            let mut slices = vec![];
            view.iter_slices(|slice| slices.push(slice));

            assert_eq!(slices.len(), 1);
        }
    }
}
