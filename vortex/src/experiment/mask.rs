// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::experiment::vector::N;
use bitvec::array::BitArray;
use bitvec::order::Msb0;
use bitvec::slice::BitSlice;
use bitvec::vec::BitVec;
use std::ops::{BitAnd, BitOr, Deref};

pub type BitVector = BitArray<[u64; N / 64], Msb0>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BitMask {
    All,
    None,
    Some(BitVector),
}

impl BitMask {
    pub fn true_count(&self) -> usize {
        match self {
            BitMask::All => N,
            BitMask::None => 0,
            BitMask::Some(bits) => bits.count_ones(),
        }
    }

    pub fn borrow(&self) -> BitMaskView {
        match self {
            BitMask::All => BitMaskView::All,
            BitMask::None => BitMaskView::None,
            BitMask::Some(bits) => BitMaskView::Some(bits),
        }
    }
}

impl BitAnd for &BitMask {
    type Output = BitMask;

    fn bitand(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (BitMask::All, BitMask::All) => BitMask::All,
            (BitMask::None, _) | (_, BitMask::None) => BitMask::None,
            (BitMask::All, BitMask::Some(b)) | (BitMask::Some(b), BitMask::All) => {
                BitMask::Some(b.clone())
            }
            (BitMask::Some(a), BitMask::Some(b)) => BitMask::Some(a.clone() & b),
        }
    }
}

impl BitAnd<&BitVector> for &BitMask {
    type Output = BitMask;

    fn bitand(self, rhs: &BitVector) -> Self::Output {
        match self {
            BitMask::All => BitMask::Some(rhs.clone()),
            BitMask::None => BitMask::None,
            BitMask::Some(a) => BitMask::Some(a.clone() & rhs),
        }
    }
}

impl BitOr for &BitMask {
    type Output = BitMask;

    fn bitor(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (BitMask::None, BitMask::None) => BitMask::None,
            (BitMask::All, _) | (_, BitMask::All) => BitMask::All,
            (BitMask::None, BitMask::Some(b)) | (BitMask::Some(b), BitMask::None) => {
                BitMask::Some(b.clone())
            }
            (BitMask::Some(a), BitMask::Some(b)) => BitMask::Some(a.clone() | b),
        }
    }
}

impl BitOr<&BitVector> for &BitMask {
    type Output = BitMask;

    fn bitor(self, rhs: &BitVector) -> Self::Output {
        match self {
            BitMask::None => BitMask::Some(rhs.clone()),
            BitMask::All => BitMask::All,
            BitMask::Some(a) => BitMask::Some(a.clone() | rhs),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BitMaskView<'a> {
    All,
    None,
    Some(&'a BitSlice<u64, Msb0>),
}

impl<'a> BitMaskView<'a> {
    pub fn to_owned(&self) -> BitMask {
        match self {
            BitMaskView::All => BitMask::All,
            BitMaskView::None => BitMask::None,
            BitMaskView::Some(bits) => {
                BitMask::Some(BitVector::try_from(*bits).expect("known size"))
            }
        }
    }

    /// Returns the number of true bits in the mask.
    pub fn true_count(&self) -> usize {
        match self {
            BitMaskView::All => N,
            BitMaskView::None => 0,
            BitMaskView::Some(bits) => bits.count_ones(),
        }
    }
}

impl<'a> From<&'a BitSlice<u64, Msb0>> for BitMaskView<'a> {
    fn from(bits: &'a BitSlice<u64, Msb0>) -> Self {
        let true_count = bits.count_ones();
        if true_count == N {
            BitMaskView::All
        } else if true_count == 0 {
            BitMaskView::None
        } else {
            BitMaskView::Some(bits)
        }
    }
}

impl BitAnd<&BitVector> for BitMaskView<'_> {
    type Output = BitMask;

    fn bitand(self, rhs: &BitVector) -> Self::Output {
        match self {
            BitMaskView::All => BitMask::Some(rhs.clone()),
            BitMaskView::None => BitMask::None,
            // BitMaskView::Some(a) => BitMask::Some(rhs & rhs),
            BitMaskView::Some(a) => {
                todo!()
            }
        }
    }
}

pub trait BitVectorMaskExt {
    /// Returns an iterator over the mutable array chunks of a BitVector.
    ///
    /// See [`bitvec::slice::ChunksExactMut::remove_alias`] for safety details.
    unsafe fn iter_vector_chunks(&mut self) -> impl Iterator<Item = &'_ mut BitVector>;
}

impl BitVectorMaskExt for BitVec<u64, Msb0> {
    unsafe fn iter_vector_chunks(&mut self) -> impl Iterator<Item = &'_ mut BitVector> {
        let iter = self.chunks_exact_mut(N);
        let iter = unsafe { iter.remove_alias() };
        iter.map(move |chunk: &mut BitSlice<u64, Msb0>| {
            <&mut BitVector>::try_from(chunk).expect("known chunk size")
        })
    }
}
