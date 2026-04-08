// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::hash::Hash;
use std::hash::Hasher;

use vortex_buffer::BitBuffer;
use vortex_buffer::Buffer;
use vortex_mask::Mask;

use crate::patches::Patches;
use crate::validity::Validity;

/// The precision level for structural equality and hashing of arrays.
///
/// This configuration option defines how precise the hash/equals results are given the set of
/// data buffers backing the array.
#[derive(Clone, Copy, Debug)]
pub enum Precision {
    /// Data buffers are compared by their pointer and length only. This is the fastest option, but
    /// may lead to false negatives if two arrays contain identical data but are backed by
    /// different buffers.
    Ptr,
    /// Data buffers are compared by their full content. This is the slowest option, but guarantees
    /// that two arrays with identical data will be considered equal.
    Value,
}

/// A hash trait for arrays that represents structural equality with a configurable level of
/// precision. This trait is used primarily to implement common subtree elimination and other
/// array-based caching mechanisms.
///
/// The precision of the hash defines what level of structural equality is represented. See
/// [`Precision`] for more details.
///
/// Note that where [`Precision::Ptr`] is used, the hash is only valid for the lifetime of the
/// object.
pub trait ArrayHash {
    fn array_hash<H: Hasher>(&self, state: &mut H, precision: Precision);
}

/// A dynamic version of [`ArrayHash`].
pub trait DynArrayHash: private::SealedHash {
    fn dyn_array_hash(&self, state: &mut dyn Hasher, precision: Precision);
}

impl<T: ArrayHash + ?Sized> DynArrayHash for T {
    fn dyn_array_hash(&self, mut state: &mut dyn Hasher, precision: Precision) {
        ArrayHash::array_hash(self, &mut state, precision);
    }
}

/// An equality trait for arrays that represents structural equality with a configurable level of
/// precision. This trait is used primarily to implement common subtree elimination and other
/// array-based caching mechanisms.
///
/// The precision of the equality check defines what level of structural equality is represented. See
/// [`Precision`] for more details.
pub trait ArrayEq {
    fn array_eq(&self, other: &Self, precision: Precision) -> bool;
}

/// A dynamic version of [`ArrayEq`].
pub trait DynArrayEq: private::SealedEq {
    fn dyn_array_eq(&self, other: &dyn Any, precision: Precision) -> bool;
}

impl<T: ArrayEq + 'static> DynArrayEq for T {
    fn dyn_array_eq(&self, other: &dyn Any, precision: Precision) -> bool {
        other
            .downcast_ref::<Self>()
            .is_some_and(|other| ArrayEq::array_eq(self, other, precision))
    }
}

mod private {
    use crate::ArrayEq;
    use crate::ArrayHash;

    pub trait SealedHash {}
    impl<T: ArrayHash + ?Sized> SealedHash for T {}
    pub trait SealedEq {}
    impl<T: ArrayEq + ?Sized> SealedEq for T {}
}

impl<T: Hash> ArrayHash for Buffer<T> {
    fn array_hash<H: Hasher>(&self, state: &mut H, precision: Precision) {
        match precision {
            Precision::Ptr => {
                self.as_ptr().hash(state);
                self.len().hash(state);
            }
            Precision::Value => {
                self.as_ref().hash(state);
            }
        }
    }
}
impl<T: PartialEq> ArrayEq for Buffer<T> {
    fn array_eq(&self, other: &Self, precision: Precision) -> bool {
        match precision {
            Precision::Ptr => self.as_ptr() == other.as_ptr() && self.len() == other.len(),
            Precision::Value => self.as_ref() == other.as_ref(),
        }
    }
}

impl ArrayHash for BitBuffer {
    fn array_hash<H: Hasher>(&self, state: &mut H, precision: Precision) {
        match precision {
            Precision::Ptr => {
                self.inner().as_ptr().hash(state);
                self.offset().hash(state);
                self.len().hash(state);
            }
            Precision::Value => {
                // NOTE(ngates): this is really rather expensive...
                for chunk in self.chunks().iter_padded() {
                    chunk.hash(state);
                }
            }
        }
    }
}
impl ArrayEq for BitBuffer {
    fn array_eq(&self, other: &Self, precision: Precision) -> bool {
        match precision {
            Precision::Ptr => {
                self.inner().as_ptr() == other.inner().as_ptr()
                    && self.offset() == other.offset()
                    && self.len() == other.len()
            }
            Precision::Value => self.eq(other),
        }
    }
}

impl<T: ArrayHash> ArrayHash for Option<T> {
    fn array_hash<H: Hasher>(&self, state: &mut H, precision: Precision) {
        match self {
            Some(value) => {
                true.hash(state);
                value.array_hash(state, precision);
            }
            None => {
                false.hash(state);
            }
        }
    }
}

impl<T: ArrayEq> ArrayEq for Option<T> {
    fn array_eq(&self, other: &Self, precision: Precision) -> bool {
        match (self, other) {
            (Some(v1), Some(v2)) => v1.array_eq(v2, precision),
            (None, None) => true,
            _ => false,
        }
    }
}

impl ArrayHash for Mask {
    fn array_hash<H: Hasher>(&self, state: &mut H, precision: Precision) {
        std::mem::discriminant(self).hash(state);
        match self {
            Mask::AllTrue(len) => {
                len.hash(state);
            }
            Mask::AllFalse(len) => {
                len.hash(state);
            }
            Mask::Values(values) => {
                values.bit_buffer().array_hash(state, precision);
            }
        }
    }
}
impl ArrayEq for Mask {
    fn array_eq(&self, other: &Self, precision: Precision) -> bool {
        match (self, other) {
            (Mask::AllTrue(len1), Mask::AllTrue(len2)) => len1 == len2,
            (Mask::AllFalse(len1), Mask::AllFalse(len2)) => len1 == len2,
            (Mask::Values(buf1), Mask::Values(buf2)) => {
                buf1.bit_buffer().array_eq(buf2.bit_buffer(), precision)
            }
            _ => false,
        }
    }
}

impl ArrayHash for Validity {
    fn array_hash<H: Hasher>(&self, state: &mut H, precision: Precision) {
        std::mem::discriminant(self).hash(state);
        if let Validity::Array(array) = self {
            array.array_hash(state, precision);
        }
    }
}

impl ArrayEq for Validity {
    fn array_eq(&self, other: &Self, precision: Precision) -> bool {
        match (self, other) {
            (Validity::AllValid, Validity::AllValid) => true,
            (Validity::AllInvalid, Validity::AllInvalid) => true,
            (Validity::NonNullable, Validity::NonNullable) => true,
            (Validity::Array(arr1), Validity::Array(arr2)) => arr1.array_eq(arr2, precision),
            _ => false,
        }
    }
}

impl ArrayHash for Patches {
    fn array_hash<H: Hasher>(&self, state: &mut H, precision: Precision) {
        self.array_len().hash(state);
        self.offset().hash(state);
        self.indices().array_hash(state, precision);
        self.values().array_hash(state, precision);
    }
}

impl ArrayEq for Patches {
    fn array_eq(&self, other: &Self, precision: Precision) -> bool {
        self.array_len() == other.array_len()
            && self.offset() == other.offset()
            && self.indices().array_eq(other.indices(), precision)
            && self.values().array_eq(other.values(), precision)
    }
}
