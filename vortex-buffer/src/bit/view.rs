// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::borrow::Cow;
use std::fmt::Debug;
use std::fmt::Formatter;
use std::marker::PhantomData;

use vortex_error::VortexResult;

use crate::BitBuffer;
use crate::BitBufferMut;

/// A borrowed fixed-size mask of length `N` bits.
///
/// Since const generic expressions are not yet stable, we instead define the type over the
/// number of bytes `NB`, and compute `N` as `NB * 8`.
///
/// This struct is designed to provide a view over a Vortex [`BitBuffer`], therefore the
/// bit-ordering is LSB0 (least-significant-bit first).
///
/// Note that [`BitView`] does not support an offset. Therefore, bits are assumed to start at
/// index and end at index `N - 1`.
pub struct BitView<'a, const NB: usize> {
    bits: Cow<'a, [u8; NB]>,
    // TODO(ngates): we may want to expose this for optimizations.
    // If set to Selection::Prefix, then all true bits are at the start of the array.
    // selection: Selection,
    true_count: usize,
}

impl<const NB: usize> Debug for BitView<'_, NB> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct(&format!("BitView[{}]", NB * 8))
            .field("true_count", &self.true_count)
            .field("bits", &self.as_raw())
            .finish()
    }
}

impl<const NB: usize> BitView<'static, NB> {
    const ALL_TRUE: [u8; NB] = [u8::MAX; NB];
    const ALL_FALSE: [u8; NB] = [0; NB];

    /// Creates a [`BitView`] with all bits set to `true`.
    pub const fn all_true() -> Self {
        unsafe { BitView::new_unchecked(&Self::ALL_TRUE, NB * 8) }
    }

    /// Creates a [`BitView`] with all bits set to `false`.
    pub const fn all_false() -> Self {
        unsafe { BitView::new_unchecked(&Self::ALL_FALSE, 0) }
    }
}

impl<'a, const NB: usize> BitView<'a, NB> {
    /// The number of bits in the view.
    pub const N: usize = NB * 8;
    /// The number of machine words in the view.
    pub const N_WORDS: usize = NB * 8 / (usize::BITS as usize);

    const _ASSERT_MULTIPLE_OF_8: () = assert!(
        NB.is_multiple_of(8),
        "NB must be a multiple of 8 for N to be a multiple of 64"
    );

    /// Creates a [`BitView`] from raw bits, computing the true count.
    pub fn new(bits: &'a [u8; NB]) -> Self {
        let ptr = bits.as_ptr().cast::<usize>();
        let true_count = (0..Self::N_WORDS)
            .map(|idx| unsafe { ptr.add(idx).read_unaligned().count_ones() as usize })
            .sum();
        BitView {
            bits: Cow::Borrowed(bits),
            true_count,
        }
    }

    /// Creates a [`BitView`] from owned raw bits.
    pub fn new_owned(bits: [u8; NB]) -> Self {
        let ptr = bits.as_ptr().cast::<usize>();
        let true_count = (0..Self::N_WORDS)
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
    pub(crate) const unsafe fn new_unchecked(bits: &'a [u8; NB], true_count: usize) -> Self {
        BitView {
            bits: Cow::Borrowed(bits),
            true_count,
        }
    }

    /// Creates a [`BitView`] from a byte slice.
    ///
    /// # Panics
    ///
    /// If the length of the slice is not equal to `NB`.
    pub fn from_slice(bits: &'a [u8]) -> Self {
        assert_eq!(bits.len(), NB);
        let bits_array = unsafe { &*(bits.as_ptr() as *const [u8; NB]) };
        BitView::new(bits_array)
    }

    /// Creates a [`BitView`] from a mutable byte array, populating it with the requested prefix
    /// of `true` bits.
    pub fn with_prefix(n_true: usize) -> Self {
        assert!(n_true <= Self::N);

        // We're going to own our own array of bits
        let mut bits = [0u8; NB];

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
    pub fn iter_words(&self) -> impl Iterator<Item = usize> + '_ {
        let ptr = self.bits.as_ptr().cast::<usize>();
        // We use constant N_WORDS to trigger loop unrolling.
        (0..Self::N_WORDS).map(move |idx| unsafe { ptr.add(idx).read_unaligned() })
    }

    /// Runs the provided function `f` for each index of a `true` bit in the view.
    pub fn iter_ones<F>(&self, mut f: F)
    where
        F: FnMut(usize),
    {
        match self.true_count {
            0 => {}
            n if n == Self::N => (0..Self::N).for_each(&mut f),
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
            n if n == Self::N => {
                for i in 0..Self::N {
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
            0 => (0..Self::N).for_each(&mut f),
            n if n == Self::N => {}
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
    ///
    /// FIXME(ngates): this is still broken.
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

    /// Returns the raw bits of the view.
    pub fn as_raw(&self) -> &[u8; NB] {
        self.bits.as_ref()
    }
}

/// A slice of bits within a [`BitBuffer`].
///
/// We use this struct to avoid a common mistake of assuming the slices represent (start, end) ranges,
pub struct BitSlice {
    /// The starting bit index of the slice.
    pub start: usize,
    /// The length of the slice in bits.
    pub len: usize,
}

impl BitBuffer {
    /// Iterate the buffer as [`BitView`]s of size `NB` where the number of bits in each view
    /// is `NB * 8`.
    ///
    /// The final chunk will be filled with unset padding bits if the bit buffer's length is not
    /// a multiple of `N`.
    ///
    /// The number of bits `N` must be a multiple of 64.
    ///
    /// # Panics
    ///
    /// If the bit offset is not zero
    pub fn iter_bit_views<const NB: usize>(&self) -> impl Iterator<Item = BitView<'_, NB>> + '_ {
        assert_eq!(
            self.offset(),
            0,
            "BitView iteration requires zero bit offset"
        );
        BitViewIterator::new(self.inner().as_ref(), self.len())
    }
}

impl BitBufferMut {
    /// Iterate the buffer as [`BitView`]s of size `NB` where the number of bits in each view
    /// is `NB * 8`.
    ///
    /// The final chunk will be filled with unset padding bits if the bit buffer's length is not
    /// a multiple of `N`.
    ///
    /// The number of bits `N` must be a multiple of 64.
    ///
    /// # Panics
    ///
    /// If the bit offset is not zero
    pub fn iter_bit_views<const NB: usize>(&self) -> impl Iterator<Item = BitView<'_, NB>> + '_ {
        assert_eq!(
            self.offset(),
            0,
            "BitView iteration requires zero bit offset"
        );
        BitViewIterator::new(self.inner().as_ref(), self.len())
    }
}

/// Iterator over fixed-size [`BitView`]s within a byte slice.
pub(super) struct BitViewIterator<'a, const NB: usize> {
    bits: &'a [u8],
    // The index of the view to be returned next
    view_idx: usize,
    // The total number of views
    n_views: usize,
    // Phantom to capture `NB`
    _phantom: PhantomData<[u8; NB]>,
}

impl<'a, const NB: usize> BitViewIterator<'a, NB> {
    /// Create a new [`BitViewIterator`].
    fn new(bits: &'a [u8], len: usize) -> Self {
        debug_assert_eq!(len.div_ceil(8), bits.len());
        let n_views = bits.len().div_ceil(NB);
        BitViewIterator {
            bits,
            view_idx: 0,
            n_views,
            _phantom: PhantomData,
        }
    }
}

impl<'a, const NB: usize> Iterator for BitViewIterator<'a, NB> {
    type Item = BitView<'a, NB>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.view_idx == self.n_views {
            return None;
        }

        let start_byte = self.view_idx * NB;
        let end_byte = start_byte + NB;

        let bits = if end_byte <= self.bits.len() {
            // Full view from the original bits
            BitView::from_slice(&self.bits[start_byte..end_byte])
        } else {
            // Partial view, copy to scratch
            let remaining_bytes = self.bits.len() - start_byte;
            let mut remaining = [0u8; NB];
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

    const NB: usize = 128; // Number of bytes
    const N: usize = NB * 8; // Number of bits

    #[test]
    fn test_iter_ones_empty() {
        let bits = [0; NB];
        let view = BitView::<NB>::new(&bits);

        let mut ones = Vec::new();
        view.iter_ones(|idx| ones.push(idx));

        assert_eq!(ones, Vec::<usize>::new());
        assert_eq!(view.true_count(), 0);
    }

    #[test]
    fn test_iter_ones_all_set() {
        let view = BitView::<NB>::all_true();

        let mut ones = Vec::new();
        view.iter_ones(|idx| ones.push(idx));

        assert_eq!(ones.len(), N);
        assert_eq!(ones, (0..N).collect::<Vec<_>>());
        assert_eq!(view.true_count(), N);
    }

    #[test]
    fn test_iter_zeros_empty() {
        let bits = [0; NB];
        let view = BitView::<NB>::new(&bits);

        let mut zeros = Vec::new();
        view.iter_zeros(|idx| zeros.push(idx));

        assert_eq!(zeros.len(), N);
        assert_eq!(zeros, (0..N).collect::<Vec<_>>());
    }

    #[test]
    fn test_iter_zeros_all_set() {
        let view = BitView::<NB>::all_true();

        let mut zeros = Vec::new();
        view.iter_zeros(|idx| zeros.push(idx));

        assert_eq!(zeros, Vec::<usize>::new());
    }

    #[test]
    fn test_iter_ones_single_bit() {
        let mut bits = [0; NB];
        bits[0] = 1; // Set bit 0 (LSB)
        let view = BitView::new(&bits);

        let mut ones = Vec::new();
        view.iter_ones(|idx| ones.push(idx));

        assert_eq!(ones, vec![0]);
        assert_eq!(view.true_count(), 1);
    }

    #[test]
    fn test_iter_zeros_single_bit_unset() {
        let mut bits = [u8::MAX; NB];
        bits[0] = u8::MAX ^ 1; // Clear bit 0 (LSB)
        let view = BitView::new(&bits);

        let mut zeros = Vec::new();
        view.iter_zeros(|idx| zeros.push(idx));

        assert_eq!(zeros, vec![0]);
    }

    #[test]
    fn test_iter_ones_multiple_bits_first_word() {
        let mut bits = [0; NB];
        bits[0] = 0b1010101; // Set bits 0, 2, 4, 6
        let view = BitView::new(&bits);

        let mut ones = Vec::new();
        view.iter_ones(|idx| ones.push(idx));

        assert_eq!(ones, vec![0, 2, 4, 6]);
        assert_eq!(view.true_count(), 4);
    }

    #[test]
    fn test_iter_zeros_multiple_bits_first_word() {
        let mut bits = [u8::MAX; NB];
        bits[0] = !0b1010101; // Clear bits 0, 2, 4, 6
        let view = BitView::new(&bits);

        let mut zeros = Vec::new();
        view.iter_zeros(|idx| zeros.push(idx));

        assert_eq!(zeros, vec![0, 2, 4, 6]);
    }

    #[test]
    fn test_lsb_bit_ordering() {
        let mut bits = [0; NB];
        bits[0] = 0b11111111; // Set bits 0-7 (LSB ordering)
        let view = BitView::new(&bits);

        let mut ones = Vec::new();
        view.iter_ones(|idx| ones.push(idx));

        assert_eq!(ones, vec![0, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(view.true_count(), 8);
    }

    #[test]
    fn test_all_false_static() {
        let view = BitView::<NB>::all_false();

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
        let view = BitView::<NB>::all_true();

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
        let view = BitView::<NB>::all_false();

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
        let mut bits = [0; NB];
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
        let mut bits = [0; NB];
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
        assert_eq!(BitView::<NB>::with_prefix(0).true_count(), 0);

        // May as well test all the possible prefix lengths!
        for i in 1..N {
            let view = BitView::<NB>::with_prefix(i);

            // Collect slices (there should be one slice from 0 to n_true)
            let mut slices = vec![];
            view.iter_slices(|slice| slices.push(slice));

            assert_eq!(slices.len(), 1);
        }
    }
}
