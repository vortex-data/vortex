// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use bitvec::array::BitArray;
use bitvec::order::Lsb0;

use crate::pipeline::PIPELINE_STEP_COUNT;
use crate::pipeline::bits::BitView;

/// A mutable borrowed fixed-size bit vector of length `N` bits, represented as an array of
/// 64-bit words.
#[derive(Debug)]
pub struct BitViewMut<'a> {
    bits: &'a mut BitArray<[u64; PIPELINE_STEP_COUNT / 64], Lsb0>,
    true_count: usize,
}

impl<'a> BitViewMut<'a> {
    pub fn new(bits: &'a mut [u64; PIPELINE_STEP_COUNT / 64]) -> Self {
        let true_count = bits.iter().map(|&word| word.count_ones() as usize).sum();
        let bits: &mut BitArray<[u64; PIPELINE_STEP_COUNT / 64], Lsb0> =
            unsafe { std::mem::transmute(bits) };
        BitViewMut { bits, true_count }
    }

    pub(crate) unsafe fn new_unchecked(
        bits: &'a mut BitArray<[u64; PIPELINE_STEP_COUNT / 64], Lsb0>,
        true_count: usize,
    ) -> Self {
        BitViewMut { bits, true_count }
    }

    pub fn true_count(&self) -> usize {
        self.true_count
    }

    /// Mask the values in the mask up to the given length.
    pub fn intersect_prefix(&mut self, mut len: usize) {
        assert!(
            len <= PIPELINE_STEP_COUNT,
            "BitViewMut::truncate: length exceeds N"
        );

        let bit_slice = self.bits.as_raw_mut_slice();

        let mut word = 0;
        let mut true_count = 0;
        while len >= 64 {
            true_count += bit_slice[word].count_ones() as usize;
            len -= 64;
            word += 1;
        }

        if len > 0 {
            bit_slice[word] &= !(u64::MAX << len);
            true_count += bit_slice[word].count_ones() as usize;
            word += 1;
        }

        while word < PIPELINE_STEP_COUNT / 64 {
            bit_slice[word] = 0;
            word += 1;
        }

        self.set_true_count(true_count);
    }

    pub fn clear(&mut self) {
        self.bits.as_raw_mut_slice().fill(0);
        self.set_true_count(0);
    }

    pub fn fill_with_words(&mut self, mut iter: impl Iterator<Item = u64>) {
        let mut true_count = 0;
        for word in 0..PIPELINE_STEP_COUNT / 64 {
            if let Some(value) = iter.next() {
                self.bits.as_raw_mut_slice()[word] = value;
                true_count += value.count_ones() as usize;
            } else {
                self.bits.as_raw_mut_slice()[word] = 0;
                break;
            }
        }
        self.set_true_count(true_count);
    }

    pub fn as_view(&self) -> BitView<'_> {
        unsafe { BitView::new_unchecked(self.bits, self.true_count) }
    }

    pub fn as_raw_mut(&mut self) -> &mut [u64; PIPELINE_STEP_COUNT / 64] {
        unsafe { std::mem::transmute(&mut self.bits) }
    }

    #[inline(always)]
    fn set_true_count(&mut self, true_count: usize) {
        self.true_count = true_count;
        debug_assert_eq!(
            self.true_count,
            self.bits
                .as_raw_slice()
                .iter()
                .map(|&word| word.count_ones() as usize)
                .sum::<usize>()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::bits::BitVector;

    #[test]
    fn test_intersect_prefix() {
        let mut bit_vec = BitVector::full().clone();

        let mut view_mut = bit_vec.as_view_mut();
        assert_eq!(view_mut.true_count(), PIPELINE_STEP_COUNT);

        view_mut.intersect_prefix(PIPELINE_STEP_COUNT - 1);
        assert_eq!(view_mut.true_count(), PIPELINE_STEP_COUNT - 1);

        view_mut.intersect_prefix(64);
        assert_eq!(view_mut.true_count(), 64);

        view_mut.intersect_prefix(10);
        assert_eq!(view_mut.true_count(), 10);

        view_mut.intersect_prefix(0);
        assert_eq!(view_mut.true_count(), 0);
    }
}
