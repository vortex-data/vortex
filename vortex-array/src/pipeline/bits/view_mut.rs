// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use bitvec::array::BitArray;
use bitvec::order::Msb0;

use crate::pipeline::N;
use crate::pipeline::bits::BitView;

/// A mutable borrowed fixed-size bit vector of length `N` bits, represented as an array of
/// 64-bit words.
#[derive(Debug)]
pub struct BitViewMut<'a> {
    bits: &'a mut BitArray<[u64; N / 64], Msb0>,
    true_count: usize,
}

impl<'a> BitViewMut<'a> {
    pub fn new(bits: &'a mut [u64; N / 64]) -> Self {
        let true_count = bits.iter().map(|&word| word.count_ones() as usize).sum();
        let bits: &mut BitArray<[u64; N / 64], Msb0> = unsafe { std::mem::transmute(bits) };
        BitViewMut { bits, true_count }
    }

    pub(crate) unsafe fn new_unchecked(
        bits: &'a mut BitArray<[u64; N / 64], Msb0>,
        true_count: usize,
    ) -> Self {
        BitViewMut { bits, true_count }
    }

    pub fn true_count(&self) -> usize {
        self.true_count
    }

    /// Mask the values in the mask up to the given length.
    pub fn intersect_prefix(&mut self, mut len: usize) {
        assert!(len <= N, "BitViewMut::truncate: length exceeds N");

        let mut word = 0;
        let mut true_count = 0;
        while len >= 64 {
            true_count += self.bits.as_raw_mut_slice()[word].count_ones() as usize;
            len -= 64;
            word += 1;
        }

        if len > 0 {
            self.bits.as_raw_mut_slice()[word] &= u64::MAX << (64 - len);
            word += 1;
            true_count += self.bits.as_raw_mut_slice()[word - 1].count_ones() as usize;
        }

        while word < N / 64 {
            self.bits.as_raw_mut_slice()[word] = 0;
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
        for word in 0..N / 64 {
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
        unsafe { BitView::new_unchecked(&self.bits, self.true_count) }
    }

    pub fn as_raw_mut(&mut self) -> &mut [u64; N / 64] {
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
                .sum()
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
        assert_eq!(view_mut.true_count(), N);

        view_mut.intersect_prefix(N - 1);
        assert_eq!(view_mut.true_count(), N - 1);

        view_mut.intersect_prefix(64);
        assert_eq!(view_mut.true_count(), 64);

        view_mut.intersect_prefix(10);
        assert_eq!(view_mut.true_count(), 10);

        view_mut.intersect_prefix(0);
        assert_eq!(view_mut.true_count(), 0);
    }
}
