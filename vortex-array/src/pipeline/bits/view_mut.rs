// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use bitvec::array::BitArray;
use bitvec::order::Lsb0;

use crate::pipeline::bits::BitView;
use crate::pipeline::{N, N_WORDS};

/// A mutable borrowed fixed-size bit vector of length `N` bits, represented as an array of
/// usize words.
/// Mutable view into a bit array for constructing selection masks.
#[derive(Debug)]
pub struct BitViewMut<'a> {
    bits: &'a mut BitArray<[usize; N_WORDS], Lsb0>,
    true_count: usize,
}

impl<'a> BitViewMut<'a> {
    pub fn new(bits: &'a mut [usize; N_WORDS]) -> Self {
        let true_count = bits.iter().map(|&word| word.count_ones() as usize).sum();
        let bits: &mut BitArray<[usize; N_WORDS], Lsb0> = unsafe { std::mem::transmute(bits) };
        BitViewMut { bits, true_count }
    }

    pub(crate) unsafe fn new_unchecked(
        bits: &'a mut BitArray<[usize; N_WORDS], Lsb0>,
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

        let bit_slice = self.bits.as_raw_mut_slice();

        let mut word = 0;
        let mut true_count = 0;
        while len >= usize::BITS as usize {
            true_count += bit_slice[word].count_ones() as usize;
            len -= usize::BITS as usize;
            word += 1;
        }

        if len > 0 {
            bit_slice[word] &= !(usize::MAX << len);
            true_count += bit_slice[word].count_ones() as usize;
            word += 1;
        }

        while word < N_WORDS {
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

        let dst_bytes = unsafe {
            std::slice::from_raw_parts_mut(
                self.bits.as_raw_mut_slice().as_mut_ptr() as *mut u64,
                N_WORDS,
            )
        };

        for word in 0..N / 64 {
            if let Some(value) = iter.next() {
                dst_bytes[word] = value;
                true_count += value.count_ones() as usize;
            }
        }
        self.set_true_count(true_count);
    }

    pub fn as_view(&self) -> BitView<'_> {
        unsafe { BitView::new_unchecked(self.bits, self.true_count) }
    }

    pub fn as_raw_mut(&mut self) -> &mut [usize; N_WORDS] {
        unsafe {
            std::mem::transmute::<&mut BitArray<[usize; N_WORDS], Lsb0>, &mut [usize; N_WORDS]>(
                &mut self.bits,
            )
        }
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
        let mut bit_vec = BitVector::full();

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
