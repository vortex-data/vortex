// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::pipeline::N;
use bitvec::prelude::*;
use vortex_error::{VortexError, vortex_err};

/// A borrowed fixed-size bit vector of length `N` bits, represented as an array of 64-bit words.
///
/// Internally, it uses a [`BitArray`] to store the bits, but this crate has some
/// performance foot-guns in cases where we can lean on better assumptions, and therefore we wrap
/// it up for use within Vortex.
#[derive(Debug, Clone, Copy)]
pub struct BitView<'a> {
    bits: &'a BitArray<[u64; N / 64], Msb0>,
    true_count: usize,
}

impl<'a> BitView<'a> {
    pub fn new(bits: &[u64; N / 64]) -> Self {
        let true_count = bits.iter().map(|&word| word.count_ones() as usize).sum();
        let bits: &BitArray<[u64; N / 64], Msb0> = unsafe { std::mem::transmute(bits) };
        BitView { bits, true_count }
    }

    pub(crate) unsafe fn new_unchecked(
        bits: &'a BitArray<[u64; N / 64], Msb0>,
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
                        raw ^= 1 << bit_pos;

                        f(bit_idx + bit_pos as usize);
                    }
                    bit_idx += 64;
                }
            }
        }
    }

    pub fn as_raw(&self) -> &[u64; N / 64] {
        // It's actually remarkably hard to get a reference to the underlying array!
        let raw = self.bits.as_raw_slice();
        unsafe { &*(raw.as_ptr() as *const [u64; N / 64]) }
    }
}

impl<'a> From<&'a [u64; N / 64]> for BitView<'a> {
    fn from(value: &'a [u64; N / 64]) -> Self {
        Self::new(value)
    }
}

impl<'a> From<&'a BitArray<[u64; N / 64], Msb0>> for BitView<'a> {
    fn from(bits: &'a BitArray<[u64; N / 64], Msb0>) -> Self {
        BitView::new(unsafe { std::mem::transmute(bits) })
    }
}

impl<'a> TryFrom<&'a BitSlice<u64, Msb0>> for BitView<'a> {
    type Error = VortexError;

    fn try_from(value: &'a BitSlice<u64, Msb0>) -> Result<Self, Self::Error> {
        let bits: &BitArray<[u64; N / 64], Msb0> = value
            .try_into()
            .map_err(|e| vortex_err!("Failed to convert BitSlice to BitArray: {}", e))?;
        Ok(BitView::new(unsafe { std::mem::transmute(bits) }))
    }
}
