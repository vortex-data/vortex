// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::experiment::N;
use crate::experiment::mask::{BitVector, BitView};
use bitvec::array::BitArray;
use bitvec::order::Msb0;
use bitvec::slice::BitSlice;
use bitvec::vec::BitVec;
use std::iter;
use std::mem::MaybeUninit;
use vortex_error::{VortexExpect, vortex_err};

/// Some multiple of N worth of bits that can be iterated over as a [`BitView`].
pub struct BitBuffer {
    bits: Vec<[u64; N / 64]>,
    len: usize,
}

impl BitBuffer {
    pub fn with_capacity(capacity: usize) -> Self {
        // Ensure capacity is a multiple of N
        assert_eq!(capacity % N, 0, "Capacity must be a multiple of N");
        let nchunks = capacity / N;
        let mut bits = Vec::with_capacity(nchunks);
        unsafe { bits.set_len(nchunks) }; // Initialize with uninitialized chunks
        Self { bits, len: 0 }
    }

    pub unsafe fn set_len(&mut self, len: usize) {
        assert!(len <= self.bits.len() * N, "Length exceeds capacity");
        self.len = len;
    }

    pub fn iter_views(&self) -> impl Iterator<Item = BitView<'_>> + '_ {
        self.bits.iter().map(move |chunk| {
            let true_count = chunk.iter().map(|word| word.count_ones() as usize).sum();
            let slice = unsafe { BitSlice::<u64, Msb0>::from_slice_unchecked(chunk) };
            let array: &BitArray<[u64; N / 64], Msb0> = slice
                .try_into()
                .map_err(|e| vortex_err!("Invalid BitView chunk: {}", e))
                .vortex_expect("infallible");
            BitView {
                bits: array,
                true_count,
            }
        })
    }
}
