// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Definition of the [`VectorMut`] type, which represents mutable and fully decompressed
//! (canonical) array data.
//!
//! [`VectorMut`] can be frozen into the [`Vector`] type.

use vortex_dtype::DType;
use vortex_error::vortex_panic;

use crate::{
    BoolVectorMut, NullVectorMut, PrimitiveVectorMut, Vector, VectorMutOps, match_each_vector_mut,
};

/// An enum over all kinds of mutable vectors, which represent fully decompressed (canonical) array
/// data.
///
/// Most of the behavior of `VectorMut` is described by the [`VectorMutOps`] trait. Note that
/// vectors are **always** considered as nullable, and it is the responsibility of the user to not
/// add any nullable data to a vector they want to keep as non-nullable.
///
/// The immutable equivalent of this type is [`Vector`], which implements the
/// [`VectorOps`](crate::VectorOps) trait.
#[derive(Debug, Clone)]
pub enum VectorMut {
    /// Null
    Null(NullVectorMut),
    /// Bool
    Bool(BoolVectorMut),
    /// Primitive
    Primitive(PrimitiveVectorMut),
}

impl VectorMut {
    /// Create a new mutable vector with the given capacity and dtype.
    pub fn with_capacity(capacity: usize, dtype: &DType) -> Self {
        match dtype {
            DType::Null => NullVectorMut::new(0).into(), // `NullVector` has `usize::MAX` capacity.
            DType::Bool(_) => BoolVectorMut::with_capacity(capacity).into(),
            DType::Primitive(ptype, _) => {
                PrimitiveVectorMut::with_capacity(*ptype, capacity).into()
            }
            _ => vortex_panic!("Unsupported dtype for VectorMut"),
        }
    }
}

impl VectorMutOps for VectorMut {
    type Immutable = Vector;

    fn len(&self) -> usize {
        match_each_vector_mut!(self, |v| { v.len() })
    }

    fn capacity(&self) -> usize {
        match_each_vector_mut!(self, |v| { v.capacity() })
    }

    fn reserve(&mut self, additional: usize) {
        match_each_vector_mut!(self, |v| { v.reserve(additional) })
    }

    fn extend_from_vector(&mut self, other: &Self::Immutable) {
        match (self, other) {
            (VectorMut::Null(a), Vector::Null(b)) => a.extend_from_vector(b),
            (VectorMut::Bool(a), Vector::Bool(b)) => a.extend_from_vector(b),
            (VectorMut::Primitive(a), Vector::Primitive(b)) => a.extend_from_vector(b),
            _ => vortex_panic!("Mismatched vector types"),
        }
    }

    fn append_nulls(&mut self, n: usize) {
        match_each_vector_mut!(self, |v| { v.append_nulls(n) })
    }

    fn freeze(self) -> Self::Immutable {
        match_each_vector_mut!(self, |v| { v.freeze().into() })
    }

    fn split_off(&mut self, at: usize) -> Self {
        match_each_vector_mut!(self, |v| { v.split_off(at).into() })
    }

    fn unsplit(&mut self, other: Self) {
        match (self, other) {
            (VectorMut::Null(a), VectorMut::Null(b)) => a.unsplit(b),
            (VectorMut::Bool(a), VectorMut::Bool(b)) => a.unsplit(b),
            (VectorMut::Primitive(a), VectorMut::Primitive(b)) => a.unsplit(b),
            _ => vortex_panic!("Mismatched vector types"),
        }
    }
}

impl VectorMut {
    /// Convert into NullVectorMut, panicking if the type does not match.
    pub fn into_null(self) -> NullVectorMut {
        if let VectorMut::Null(v) = self {
            return v;
        }
        vortex_panic!("Expected NullVectorMut, got {self:?}");
    }

    /// Convert into BoolVectorMut, panicking if the type does not match.
    pub fn into_bool(self) -> BoolVectorMut {
        if let VectorMut::Bool(v) = self {
            return v;
        }
        vortex_panic!("Expected BoolVectorMut, got {self:?}");
    }

    /// Convert into PrimitiveVectorMut, panicking if the type does not match.
    pub fn into_primitive(self) -> PrimitiveVectorMut {
        if let VectorMut::Primitive(v) = self {
            return v;
        }
        vortex_panic!("Expected PrimitiveVectorMut, got {self:?}");
    }
}

#[cfg(test)]
mod tests {
    use vortex_dtype::{Nullability, PType};

    use super::*;
    use crate::{PVectorMut, VectorOps};

    #[test]
    fn test_with_capacity() {
        // Test capacity allocation for different types.
        let null_vec = VectorMut::with_capacity(10, &DType::Null);
        assert_eq!(null_vec.capacity(), usize::MAX); // Null vectors have unlimited capacity.

        let bool_vec = VectorMut::with_capacity(100, &DType::Bool(Nullability::Nullable));
        assert!(bool_vec.capacity() >= 100);

        let prim_vec =
            VectorMut::with_capacity(50, &DType::Primitive(PType::I32, Nullability::Nullable));
        assert!(prim_vec.capacity() >= 50);
    }

    #[test]
    #[should_panic(expected = "Mismatched vector types")]
    fn test_type_mismatch_panics() {
        // Test that operations between mismatched types panic.
        let mut vec1 = VectorMut::with_capacity(10, &DType::Bool(Nullability::Nullable));
        let vec2 =
            VectorMut::with_capacity(10, &DType::Primitive(PType::I32, Nullability::Nullable));

        vec1.unsplit(vec2); // Should panic.
    }

    #[test]
    fn test_split_and_unsplit() {
        // Test split at various positions.
        let mut vec: VectorMut = BoolVectorMut::from_iter([true, false, true].map(Some)).into();

        // Split at beginning.
        let second = vec.split_off(0);
        assert_eq!(vec.len(), 0);
        assert_eq!(second.len(), 3);

        // Unsplit to restore.
        vec.unsplit(second);
        assert_eq!(vec.len(), 3);

        // Split at end.
        let second = vec.split_off(3);
        assert_eq!(vec.len(), 3);
        assert_eq!(second.len(), 0);
    }

    #[test]
    fn test_reserve_ensures_len_plus_additional() {
        // Test that reserve ensures capacity >= len + additional.
        // This specifically tests the fix for the BitBufferMut::reserve bug.
        let mut bool_vec: VectorMut = BoolVectorMut::with_capacity(10).into();
        let initial_len = bool_vec.len();
        assert_eq!(initial_len, 0);

        // Reserve 100 additional capacity.
        bool_vec.reserve(100);

        // Should have capacity for at least len + 100.
        assert!(bool_vec.capacity() >= initial_len + 100);
        assert!(bool_vec.capacity() >= 100); // Since len is 0.

        // Test with primitive vector too.
        let mut prim_vec: VectorMut = PVectorMut::<i32>::with_capacity(10).into();
        prim_vec.reserve(100);
        assert!(prim_vec.capacity() >= prim_vec.len() + 100);

        // Test with non-empty vector.
        let mut vec: VectorMut = BoolVectorMut::from_iter([true, false, true].map(Some)).into();
        let len = vec.len();
        assert_eq!(len, 3);
        vec.reserve(50);
        assert!(vec.capacity() >= len + 50);
        assert!(vec.capacity() >= 53);
    }

    #[test]
    fn test_append_nulls_preserves_validity() {
        // Test that appending nulls preserves existing validity.
        let mut vec: VectorMut = BoolVectorMut::from_iter([true].map(Some)).into();
        vec.append_nulls(2);
        assert_eq!(vec.len(), 3);

        let frozen = vec.freeze();
        assert_eq!(frozen.validity().true_count(), 1); // Only first element is valid.
    }

    #[test]
    fn test_extend_from_vector() {
        // Test extending a primitive vector with data from another vector.
        let mut vec: VectorMut = PVectorMut::<i32>::from_iter([1, 2, 3].map(Some)).into();
        assert_eq!(vec.len(), 3);

        // Create an immutable vector to extend from.
        let to_append: Vector = PVectorMut::<i32>::from_iter([4, 5, 6].map(Some))
            .freeze()
            .into();
        assert_eq!(to_append.len(), 3);

        // Extend the mutable vector.
        vec.extend_from_vector(&to_append);

        // Verify the length is the sum of both vectors.
        assert_eq!(vec.len(), 6);

        // Verify validity is preserved (all elements are valid).
        let frozen = vec.freeze();
        assert_eq!(frozen.validity().true_count(), 6);
    }
}
