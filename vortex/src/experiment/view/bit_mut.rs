// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::experiment::N;
use crate::experiment::view::bit::BitView;
use bitvec::array::BitArray;
use bitvec::order::Msb0;

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
}
