// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::{Debug, Formatter};
use std::ops::Not;
use std::sync::{Arc, LazyLock};

use bitvec::array::BitArray;
use bitvec::order::Lsb0;

use crate::PIPELINE_STEP_COUNT;
use crate::bits::{BitView, BitViewMut};

static EMPTY: LazyLock<BitVector> = LazyLock::new(|| BitVector {
    bits: Arc::new(BitArray::ZERO),
    true_count: 0,
});

static FULL: LazyLock<BitVector> = LazyLock::new(|| BitVector {
    bits: Arc::new(BitArray::ZERO.not()),
    true_count: PIPELINE_STEP_COUNT,
});

/// An owned fixed-size bit vector of length `N` bits, represented as an array of 64-bit words.
///
/// Internally, it uses a [`BitArray`] to store the bits, but this crate has some
/// performance foot-guns in cases where we can lean on better assumptions, and therefore we wrap
/// it up for use within Vortex.
#[derive(Clone)]
pub struct BitVector {
    pub(super) bits: Arc<BitArray<[u64; PIPELINE_STEP_COUNT / 64], Lsb0>>,
    pub(super) true_count: usize,
}

impl Debug for BitVector {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BitVector")
            .field("true_count", &self.true_count)
            //.field("bits", &self.bits.as_raw_slice())
            .finish()
    }
}

impl PartialEq for BitVector {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.bits, &other.bits)
            || (self.true_count == other.true_count && self.bits == other.bits)
    }
}

impl Eq for BitVector {}

impl BitVector {
    pub fn empty() -> &'static BitVector {
        &EMPTY
    }

    pub fn full() -> &'static BitVector {
        &FULL
    }

    pub fn true_until(n: usize) -> Self {
        assert!(
            n <= PIPELINE_STEP_COUNT,
            "Cannot create a BitVector with more than N bits"
        );

        let mut bits = Arc::new(BitArray::<[u64; PIPELINE_STEP_COUNT / 64], Lsb0>::ZERO);
        let bits_mut = Arc::make_mut(&mut bits);

        let mut word = 0;
        let mut remaining = n;
        while remaining >= 64 {
            bits_mut.as_raw_mut_slice()[word] = u64::MAX;
            remaining -= 64;
            word += 1;
        }

        if remaining > 0 {
            // For LSB ordering, set the lower bits (0 to remaining-1)
            bits_mut.as_raw_mut_slice()[word] = (1u64 << remaining) - 1;
        }

        BitVector {
            bits,
            true_count: n,
        }
    }

    pub fn true_count(&self) -> usize {
        self.true_count
    }

    pub fn as_raw(&self) -> &[u64; PIPELINE_STEP_COUNT / 64] {
        // It's actually remarkably hard to get a reference to the underlying array!
        let raw = self.bits.as_raw_slice();
        unsafe { &*(raw.as_ptr() as *const [u64; PIPELINE_STEP_COUNT / 64]) }
    }

    pub fn as_raw_mut(&mut self) -> &mut [u64; PIPELINE_STEP_COUNT / 64] {
        // SAFETY: We assume that the bits are mutable and that the view is valid.
        let raw = Arc::make_mut(&mut self.bits).as_raw_mut_slice();
        unsafe { &mut *(raw.as_mut_ptr() as *mut [u64; PIPELINE_STEP_COUNT / 64]) }
    }

    pub fn fill_from<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = u64>,
    {
        let mut true_count = 0;
        for (dst, word) in self.as_raw_mut().iter_mut().zip(iter) {
            true_count += word.count_ones() as usize;
            *dst = word;
        }
        self.true_count = true_count;
    }

    pub fn as_view(&self) -> BitView<'_> {
        unsafe { BitView::new_unchecked(&self.bits, self.true_count) }
    }

    pub fn as_view_mut(&mut self) -> BitViewMut<'_> {
        unsafe { BitViewMut::new_unchecked(Arc::make_mut(&mut self.bits), self.true_count) }
    }
}

impl From<BitView<'_>> for BitVector {
    fn from(value: BitView<'_>) -> Self {
        let true_count = value.true_count();
        let bits = Arc::new(BitArray::<[u64; PIPELINE_STEP_COUNT / 64], Lsb0>::from(
            *value.as_raw(),
        ));
        BitVector { bits, true_count }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fill_from() {
        let mut vec = BitVector::empty().clone();

        // Fill with a pattern
        let pattern = [0b1010101010101010u64, 0b1111000011110000u64, u64::MAX, 0];

        vec.fill_from(pattern.iter().copied());

        let raw = vec.as_raw();
        assert_eq!(raw[0], 0b1010101010101010u64);
        assert_eq!(raw[1], 0b1111000011110000u64);
        assert_eq!(raw[2], u64::MAX);
        assert_eq!(raw[3], 0);

        // Check true_count is updated correctly
        let expected_count = 0b1010101010101010u64.count_ones() as usize
            + 0b1111000011110000u64.count_ones() as usize
            + u64::MAX.count_ones() as usize;
        assert_eq!(vec.true_count(), expected_count);
    }

    #[test]
    fn test_as_view() {
        let vec = BitVector::true_until(100);
        let view = vec.as_view();

        assert_eq!(view.true_count(), 100);

        // Verify the view sees the same bits
        let mut ones = Vec::new();
        view.iter_ones(|idx| ones.push(idx));
        assert_eq!(ones, (0..100).collect::<Vec<_>>());
    }

    #[test]
    fn test_as_view_mut() {
        let mut vec = BitVector::true_until(64);
        {
            let view_mut = vec.as_view_mut();
            // BitViewMut would allow modifications
            // This test just verifies we can create a mutable view
            assert_eq!(view_mut.true_count(), 64);
        }
        assert_eq!(vec.true_count(), 64);
    }

    #[test]
    fn test_from_bitview() {
        // Create a BitView from raw data
        let mut raw = [0u64; PIPELINE_STEP_COUNT / 64];
        raw[0] = 0b11111111;
        raw[1] = 0b11110000;

        let view = BitView::new(&raw);
        let vec = BitVector::from(view);

        assert_eq!(vec.true_count(), view.true_count());
        assert_eq!(vec.as_raw()[0], 0b11111111);
        assert_eq!(vec.as_raw()[1], 0b11110000);
    }

    #[test]
    fn test_lsb_ordering_verification() {
        // Verify LSB ordering by setting specific bits
        let vec = BitVector::true_until(5);
        let view = vec.as_view();

        // Collect which bits are set
        let mut ones = Vec::new();
        view.iter_ones(|idx| ones.push(idx));

        // With LSB ordering, bits 0-4 should be set
        assert_eq!(ones, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_as_raw_mut() {
        let mut vec = BitVector::empty().clone();

        // Modify through as_raw_mut
        let raw_mut = vec.as_raw_mut();
        raw_mut[0] = 0b1111;
        raw_mut[2] = u64::MAX;

        // Note: true_count is NOT automatically updated when using as_raw_mut
        // This is a low-level API, so the user must manage true_count
        vec.true_count = 4 + 64; // Update manually

        assert_eq!(vec.as_raw()[0], 0b1111);
        assert_eq!(vec.as_raw()[2], u64::MAX);
        assert_eq!(vec.true_count(), 68);
    }

    #[test]
    fn test_boundary_conditions() {
        // Test various boundary values
        let boundaries = [
            1,
            31,
            32,
            33,
            63,
            64,
            65,
            127,
            128,
            129,
            PIPELINE_STEP_COUNT - 1,
            PIPELINE_STEP_COUNT,
        ];

        for &n in &boundaries {
            let vec = BitVector::true_until(n);
            assert_eq!(vec.true_count(), n);

            // Verify correct bits are set via view
            let view = vec.as_view();
            let mut ones = Vec::new();
            view.iter_ones(|idx| ones.push(idx));
            assert_eq!(ones.len(), n);
            if n > 0 {
                assert_eq!(ones[0], 0); // First bit should be 0 (LSB)
                assert_eq!(ones[n - 1], n - 1); // Last bit should be n-1
            }
        }
    }
}
