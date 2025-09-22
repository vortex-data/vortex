// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::{Hash, Hasher};
use std::sync::Arc;

use vortex_buffer::Buffer;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::validity::Validity;

/// A wrapper type to implement [`Hash`] using the semantics defined by
/// [`crate::operator::Operator`].
pub struct OperatorHash<T>(pub T);

impl<T> Hash for OperatorHash<&Buffer<T>> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.as_ptr().hash(state);
        self.0.len().hash(state);
    }
}
impl<T> PartialEq for OperatorHash<&Buffer<T>> {
    fn eq(&self, other: &Self) -> bool {
        self.0.as_ptr() == other.0.as_ptr() && self.0.len() == other.0.len()
    }
}
impl<T> Eq for OperatorHash<&Buffer<T>> {}

impl Hash for OperatorHash<&Mask> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self.0).hash(state);
        match &self.0 {
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
impl PartialEq for OperatorHash<&Mask> {
    fn eq(&self, other: &Self) -> bool {
        match (&self.0, &other.0) {
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
impl Eq for OperatorHash<&Mask> {}

impl Hash for OperatorHash<&Validity> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self.0).hash(state);
        if let Validity::Array(array) = &self.0 {
            Arc::as_ptr(array).hash(state);
        }
    }
}
impl PartialEq for OperatorHash<&Validity> {
    fn eq(&self, other: &Self) -> bool {
        match (&self.0, &other.0) {
            (Validity::AllValid, Validity::AllValid) => true,
            (Validity::AllInvalid, Validity::AllInvalid) => true,
            (Validity::NonNullable, Validity::NonNullable) => true,
            (Validity::Array(arr1), Validity::Array(arr2)) => Arc::ptr_eq(arr1, arr2),
            _ => false,
        }
    }
}
impl Eq for OperatorHash<&Validity> {}

impl Hash for OperatorHash<&ArrayRef> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Arc::as_ptr(self.0).hash(state);
    }
}
impl PartialEq for OperatorHash<&ArrayRef> {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(self.0, other.0)
    }
}
impl Eq for OperatorHash<&ArrayRef> {}
