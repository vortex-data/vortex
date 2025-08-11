// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::{Debug, Formatter};
use std::ops::Not;
use std::sync::{Arc, LazyLock};

use bitvec::array::BitArray;
use bitvec::order::Msb0;

use crate::pipeline::N;
use crate::pipeline::bits::{BitView, BitViewMut};

static EMPTY: LazyLock<BitVector> = LazyLock::new(|| BitVector {
    bits: Arc::new(BitArray::ZERO),
    true_count: 0,
});

static FULL: LazyLock<BitVector> = LazyLock::new(|| BitVector {
    bits: Arc::new(BitArray::ZERO.not()),
    true_count: N,
});

/// An owned fixed-size bit vector of length `N` bits, represented as an array of 64-bit words.
///
/// Internally, it uses a [`BitArray`] to store the bits, but this crate has some
/// performance foot-guns in cases where we can lean on better assumptions, and therefore we wrap
/// it up for use within Vortex.
#[derive(Clone)]
pub struct BitVector {
    pub(super) bits: Arc<BitArray<[u64; N / 64], Msb0>>,
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
        assert!(n <= N, "Cannot create a BitVector with more than N bits");

        let mut bits = Arc::new(BitArray::<[u64; N / 64], Msb0>::ZERO);
        let bits_mut = Arc::make_mut(&mut bits);

        let mut word = 0;
        let mut remaining = n;
        while remaining >= 64 {
            bits_mut.as_raw_mut_slice()[word] = u64::MAX;
            remaining -= 64;
            word += 1;
        }

        if remaining > 0 {
            bits_mut.as_raw_mut_slice()[word] = u64::MAX << (64 - remaining);
        }

        BitVector {
            bits,
            true_count: n,
        }
    }

    pub fn true_count(&self) -> usize {
        self.true_count
    }

    pub fn as_raw(&self) -> &[u64; N / 64] {
        // It's actually remarkably hard to get a reference to the underlying array!
        let raw = self.bits.as_raw_slice();
        unsafe { &*(raw.as_ptr() as *const [u64; N / 64]) }
    }

    pub fn as_raw_mut(&mut self) -> &mut [u64; N / 64] {
        // SAFETY: We assume that the bits are mutable and that the view is valid.
        let raw = Arc::make_mut(&mut self.bits).as_raw_mut_slice();
        unsafe { &mut *(raw.as_mut_ptr() as *mut [u64; N / 64]) }
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
        let bits = Arc::new(BitArray::<[u64; N / 64], Msb0>::from(*value.as_raw()));
        BitVector { bits, true_count }
    }
}
