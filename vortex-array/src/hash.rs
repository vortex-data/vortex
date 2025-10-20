// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use vortex_buffer::Buffer;
use vortex_mask::Mask;

use crate::validity::Validity;
use crate::{Array, ArrayRef};

/// A hash trait for arrays that loosens the semantics to permit pointer-based hashing for
/// data objects such as buffers.
///
/// Note that since this trait can use pointer hashing, the hash is only valid for the lifetime of
/// the object.
pub trait ArrayHash {
    fn array_hash<H: Hasher>(&self, state: &mut H);
}

/// A dynamic version of [`ArrayHash`].
pub trait DynArrayHash: private::SealedHash {
    fn dyn_array_hash(&self, state: &mut dyn Hasher);
}

impl<T: ArrayHash + ?Sized> DynArrayHash for T {
    fn dyn_array_hash(&self, mut state: &mut dyn Hasher) {
        ArrayHash::array_hash(self, &mut state);
    }
}

/// An equality trait for arrays that loosens the semantics to permit pointer-based equality
/// for data objects such as buffers.
///
/// Note that this still represents structural equality, not equality of the logical data.
pub trait ArrayEq {
    fn array_eq(&self, other: &Self) -> bool;
}

/// A dynamic version of [`ArrayEq`].
pub trait DynArrayEq: private::SealedEq {
    fn dyn_array_eq(&self, other: &dyn Any) -> bool;
}

impl<T: ArrayEq + 'static> DynArrayEq for T {
    fn dyn_array_eq(&self, other: &dyn Any) -> bool {
        other
            .downcast_ref::<Self>()
            .is_some_and(|other| ArrayEq::array_eq(self, other))
    }
}

mod private {
    use crate::{ArrayEq, ArrayHash};

    pub trait SealedHash {}
    impl<T: ArrayHash + ?Sized> SealedHash for T {}
    pub trait SealedEq {}
    impl<T: ArrayEq + ?Sized> SealedEq for T {}
}

impl ArrayHash for dyn Array + '_ {
    fn array_hash<H: Hasher>(&self, state: &mut H) {
        self.dyn_array_hash(state);
    }
}

impl ArrayEq for dyn Array + '_ {
    fn array_eq(&self, other: &Self) -> bool {
        self.dyn_array_eq(other.as_any())
    }
}

impl ArrayHash for ArrayRef {
    fn array_hash<H: Hasher>(&self, state: &mut H) {
        self.as_ref().array_hash(state);
    }
}

impl ArrayEq for ArrayRef {
    fn array_eq(&self, other: &Self) -> bool {
        self.as_ref().array_eq(other.as_ref())
    }
}

/// A wrapper type to implement [`Hash`], [`PartialEq`], and [`Eq`] using the semantics defined
/// by [`ArrayHash`] and [`ArrayEq`].
pub struct ArrayKey<T>(pub T);
impl<T: ArrayHash> Hash for ArrayKey<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.array_hash(state);
    }
}
impl<T: ArrayEq + Any> PartialEq for ArrayKey<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0.array_eq(&other.0)
    }
}
impl<T: ArrayEq + Any> Eq for ArrayKey<T> {}

impl<T> ArrayHash for Buffer<T> {
    fn array_hash<H: Hasher>(&self, state: &mut H) {
        self.as_ptr().hash(state);
        self.len().hash(state);
    }
}
impl<T> ArrayEq for Buffer<T> {
    fn array_eq(&self, other: &Self) -> bool {
        self.as_ptr() == other.as_ptr() && self.len() == other.len()
    }
}

impl ArrayHash for Mask {
    fn array_hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Mask::AllTrue(len) => {
                len.hash(state);
            }
            Mask::AllFalse(len) => {
                len.hash(state);
            }
            Mask::Values(values) => {
                let buffer = values.boolean_buffer();
                buffer.offset().hash(state);
                buffer.len().hash(state);
                buffer.inner().as_ptr().hash(state);
            }
        }
    }
}
impl ArrayEq for Mask {
    fn array_eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Mask::AllTrue(len1), Mask::AllTrue(len2)) => len1 == len2,
            (Mask::AllFalse(len1), Mask::AllFalse(len2)) => len1 == len2,
            (Mask::Values(buf1), Mask::Values(buf2)) => {
                let b1 = buf1.boolean_buffer();
                let b2 = buf2.boolean_buffer();
                b1.offset() == b2.offset()
                    && b1.len() == b2.len()
                    && b1.inner().as_ptr() == b2.inner().as_ptr()
            }
            _ => false,
        }
    }
}

impl ArrayHash for Validity {
    fn array_hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        if let Validity::Array(array) = self {
            Arc::as_ptr(array).hash(state);
        }
    }
}

impl ArrayEq for Validity {
    fn array_eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Validity::AllValid, Validity::AllValid) => true,
            (Validity::AllInvalid, Validity::AllInvalid) => true,
            (Validity::NonNullable, Validity::NonNullable) => true,
            (Validity::Array(arr1), Validity::Array(arr2)) => Arc::ptr_eq(arr1, arr2),
            _ => false,
        }
    }
}
