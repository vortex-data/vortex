// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use vortex_buffer::Buffer;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::operator::{Operator, OperatorRef};
use crate::validity::Validity;

/// A hash trait for operators that loosens the semantics to permit pointer-based hashing for
/// data objects such as buffers.
///
/// Note that since this trait can use pointer hashing, the hash is only valid for the lifetime of
/// the object.
pub trait OperatorHash {
    fn operator_hash<H: Hasher>(&self, state: &mut H);
}

pub trait DynOperatorHash: private::SealedHash {
    fn dyn_operator_hash(&self, state: &mut dyn Hasher);
}

impl<T: OperatorHash + ?Sized> DynOperatorHash for T {
    fn dyn_operator_hash(&self, mut state: &mut dyn Hasher) {
        OperatorHash::operator_hash(self, &mut state);
    }
}

/// An equality trait for operators that loosens the semantics to permit pointer-based equality
/// for data objects such as buffers.
pub trait OperatorEq {
    fn operator_eq(&self, other: &Self) -> bool;
}

pub trait DynOperatorEq: private::SealedEq {
    fn dyn_operator_eq(&self, other: &dyn Any) -> bool;
}

impl<T: OperatorEq + 'static> DynOperatorEq for T {
    fn dyn_operator_eq(&self, other: &dyn Any) -> bool {
        other
            .downcast_ref::<Self>()
            .is_some_and(|other| OperatorEq::operator_eq(self, other))
    }
}

mod private {
    use crate::operator::{OperatorEq, OperatorHash};

    pub trait SealedHash {}
    impl<T: OperatorHash + ?Sized> SealedHash for T {}
    pub trait SealedEq {}
    impl<T: OperatorEq + ?Sized> SealedEq for T {}
}

impl OperatorHash for dyn Operator + '_ {
    fn operator_hash<H: Hasher>(&self, state: &mut H) {
        self.dyn_operator_hash(state);
    }
}

impl OperatorEq for dyn Operator + '_ {
    fn operator_eq(&self, other: &Self) -> bool {
        self.dyn_operator_eq(other.as_any())
    }
}

impl OperatorHash for OperatorRef {
    fn operator_hash<H: Hasher>(&self, state: &mut H) {
        self.as_ref().operator_hash(state);
    }
}

impl OperatorEq for OperatorRef {
    fn operator_eq(&self, other: &Self) -> bool {
        self.as_ref().operator_eq(other.as_ref())
    }
}

/// A wrapper type to implement [`Hash`], [`PartialEq`], and [`Eq`] using the semantics defined
/// by [`OperatorHash`] and [`OperatorEq`].
pub struct OperatorKey<T>(pub T);
impl<T: OperatorHash> Hash for OperatorKey<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.operator_hash(state);
    }
}
impl<T: OperatorEq + Any> PartialEq for OperatorKey<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0.operator_eq(&other.0)
    }
}
impl<T: OperatorEq + Any> Eq for OperatorKey<T> {}

impl<T> OperatorHash for Buffer<T> {
    fn operator_hash<H: Hasher>(&self, state: &mut H) {
        self.as_ptr().hash(state);
        self.len().hash(state);
    }
}
impl<T> OperatorEq for Buffer<T> {
    fn operator_eq(&self, other: &Self) -> bool {
        self.as_ptr() == other.as_ptr() && self.len() == other.len()
    }
}

impl OperatorHash for Mask {
    fn operator_hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Mask::AllTrue(len) => {
                len.hash(state);
            }
            Mask::AllFalse(len) => {
                len.hash(state);
            }
            Mask::Values(values) => {
                let buffer = values.bit_buffer();
                buffer.offset().hash(state);
                buffer.len().hash(state);
                buffer.inner().as_ptr().hash(state);
            }
        }
    }
}
impl OperatorEq for Mask {
    fn operator_eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Mask::AllTrue(len1), Mask::AllTrue(len2)) => len1 == len2,
            (Mask::AllFalse(len1), Mask::AllFalse(len2)) => len1 == len2,
            (Mask::Values(buf1), Mask::Values(buf2)) => {
                let b1 = buf1.bit_buffer();
                let b2 = buf2.bit_buffer();
                b1.offset() == b2.offset()
                    && b1.len() == b2.len()
                    && b1.inner().as_ptr() == b2.inner().as_ptr()
            }
            _ => false,
        }
    }
}

impl OperatorHash for Validity {
    fn operator_hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        if let Validity::Array(array) = self {
            Arc::as_ptr(array).hash(state);
        }
    }
}
impl OperatorEq for Validity {
    fn operator_eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Validity::AllValid, Validity::AllValid) => true,
            (Validity::AllInvalid, Validity::AllInvalid) => true,
            (Validity::NonNullable, Validity::NonNullable) => true,
            (Validity::Array(arr1), Validity::Array(arr2)) => Arc::ptr_eq(arr1, arr2),
            _ => false,
        }
    }
}

impl OperatorHash for ArrayRef {
    fn operator_hash<H: Hasher>(&self, state: &mut H) {
        Arc::as_ptr(self).hash(state);
    }
}
impl OperatorEq for ArrayRef {
    fn operator_eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(self, other)
    }
}
