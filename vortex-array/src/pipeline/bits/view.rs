// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::pipeline::N;
use bitvec::prelude::*;
use std::fmt::{Debug, Formatter};
use vortex_error::{VortexError, vortex_err};

/// A borrowed fixed-size bit vector of length `N` bits, represented as an array of 64-bit words.
///
/// Internally, it uses a [`BitArray`] to store the bits, but this crate has some
/// performance foot-guns in cases where we can lean on better assumptions, and therefore we wrap
/// it up for use within Vortex.
#[derive(Clone, Copy)]
pub struct BitView<'a> {
    bits: &'a BitArray<[u64; N / 64], Msb0>,
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
        unsafe { BitView::new_unchecked(std::mem::transmute(&[u64::MAX; N / 64]), N) }
    }

    pub fn all_false() -> Self {
        unsafe { BitView::new_unchecked(std::mem::transmute(&[0; N / 64]), 0) }
    }
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
        BitView {
            bits: unsafe { std::mem::transmute(bits) },
            true_count,
        }
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
                        f(bit_idx + bit_pos as usize);
                        raw &= raw - 1; // Clear the bit at `bit_pos`
                    }
                    bit_idx += 64;
                }
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
                for mut raw in self.bits.into_inner() {
                    while raw != u64::MAX {
                        let bit_pos = raw.trailing_ones();
                        f(bit_idx + bit_pos as usize);
                        raw |= 1u64 << bit_pos; // Set the zero bit to 1
                    }
                    bit_idx += 64;
                }
            }
        }
    }

    /// Runs the provided function `f` for each range of `true` bits in the view.
    ///
    /// The function `f` receives a tuple `(start, len)` where `start` is the index of the first
    /// `true` bit and `len` is the number of consecutive `true` bits.
    pub fn iter_slices<F>(&self, mut f: F)
    where
        F: FnMut((usize, usize)),
    {
        match self.true_count {
            0 => {}
            N => f((0, N)),
            _ => {
                let mut bit_idx = 0;
                for mut raw in self.bits.into_inner() {
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
                    bit_idx += 64; // Move to next word
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
