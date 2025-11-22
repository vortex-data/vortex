// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Definition of the [`VectorMut`] type, which represents mutable and fully decompressed
//! (canonical) array data.
//!
//! [`VectorMut`] can be frozen into the [`Vector`] type.

use vortex_dtype::DType;
use vortex_error::vortex_panic;
use vortex_mask::MaskMut;

use crate::binaryview::{BinaryVectorMut, StringVectorMut};
use crate::bool::BoolVectorMut;
use crate::decimal::DecimalVectorMut;
use crate::fixed_size_list::FixedSizeListVectorMut;
use crate::listview::ListViewVectorMut;
use crate::null::NullVectorMut;
use crate::primitive::PrimitiveVectorMut;
use crate::struct_::StructVectorMut;
use crate::{Vector, VectorMutOps, VectorOps, match_each_vector_mut, match_vector_pair};

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
    /// Mutable Null vectors.
    Null(NullVectorMut),
    /// Mutable Boolean vectors.
    Bool(BoolVectorMut),
    /// Mutable Decimal vectors.
    ///
    /// Note that [`DecimalVectorMut`] is an enum over the different possible (generic)
    /// [`DVectorMut<D>`](crate::decimal::DVectorMut)s.
    ///
    /// See the [documentation](crate::decimal) for more information.
    Decimal(DecimalVectorMut),
    /// Mutable Primitive vectors.
    ///
    /// Note that [`PrimitiveVectorMut`] is an enum over the different possible (generic)
    /// [`PVectorMut<T>`](crate::primitive::PVectorMut)s.
    ///
    /// See the documentation for more information.
    Primitive(PrimitiveVectorMut),
    /// Mutable String vectors.
    String(StringVectorMut),
    /// Mutable Binary vectors.
    Binary(BinaryVectorMut),
    /// Mutable vectors of Lists with variable sizes.
    List(ListViewVectorMut),
    /// Mutable vectors of Lists with fixed sizes.
    FixedSizeList(FixedSizeListVectorMut),
    /// Mutable vectors of Struct elements.
    Struct(StructVectorMut),
}

impl VectorMut {
    /// Create a new mutable vector with the given capacity and dtype.
    pub fn with_capacity(dtype: &DType, capacity: usize) -> Self {
        match dtype {
            DType::Null => NullVectorMut::new(0).into(),
            DType::Bool(_) => BoolVectorMut::with_capacity(capacity).into(),
            DType::Primitive(ptype, _) => {
                PrimitiveVectorMut::with_capacity(*ptype, capacity).into()
            }
            DType::FixedSizeList(elem_dtype, list_size, _) => {
                FixedSizeListVectorMut::with_capacity(elem_dtype, *list_size, capacity).into()
            }
            DType::Struct(struct_fields, _) => {
                StructVectorMut::with_capacity(struct_fields, capacity).into()
            }
            DType::Decimal(decimal_dtype, _) => {
                DecimalVectorMut::with_capacity(decimal_dtype, capacity).into()
            }
            DType::Utf8(..) => StringVectorMut::with_capacity(capacity).into(),
            DType::Binary(..) => BinaryVectorMut::with_capacity(capacity).into(),
            DType::Extension(ext) => VectorMut::with_capacity(ext.storage_dtype(), capacity),
            DType::List(..) => ListViewVectorMut::with_capacity(dtype, capacity).into(),
        }
    }
}

impl VectorMutOps for VectorMut {
    type Immutable = Vector;

    fn len(&self) -> usize {
        match_each_vector_mut!(self, |v| { v.len() })
    }

    fn validity(&self) -> &MaskMut {
        match_each_vector_mut!(self, |v| { v.validity() })
    }

    fn capacity(&self) -> usize {
        match_each_vector_mut!(self, |v| { v.capacity() })
    }

    fn reserve(&mut self, additional: usize) {
        match_each_vector_mut!(self, |v| { v.reserve(additional) })
    }

    fn clear(&mut self) {
        match_each_vector_mut!(self, |v| { v.clear() })
    }

    fn truncate(&mut self, len: usize) {
        match_each_vector_mut!(self, |v| { v.truncate(len) })
    }

    fn extend_from_vector(&mut self, other: &Vector) {
        match_vector_pair!(self, other, |a: VectorMut, b: Vector| {
            a.extend_from_vector(b)
        })
    }

    fn append_nulls(&mut self, n: usize) {
        match_each_vector_mut!(self, |v| { v.append_nulls(n) })
    }

    fn append_zeros(&mut self, n: usize) {
        match_each_vector_mut!(self, |v| { v.append_zeros(n) })
    }

    fn append_scalars(&mut self, scalar: &<Self::Immutable as VectorOps>::Scalar, n: usize) {
        match_vector_pair!(self, scalar, |a: VectorMut, b: Scalar| {
            a.append_scalars(b, n)
        })
    }

    fn freeze(self) -> Vector {
        match_each_vector_mut!(self, |v| { v.freeze().into() })
    }

    fn split_off(&mut self, at: usize) -> Self {
        match_each_vector_mut!(self, |v| { v.split_off(at).into() })
    }

    fn unsplit(&mut self, other: Self) {
        match_vector_pair!(self, other, |a: VectorMut, b: VectorMut| a.unsplit(b))
    }
}

impl VectorMut {
    /// Returns a reference to the inner [`NullVectorMut`] if `self` is of that variant.
    pub fn as_null_mut(&mut self) -> &mut NullVectorMut {
        if let VectorMut::Null(v) = self {
            return v;
        }
        vortex_panic!("Expected NullVectorMut, got {self:?}");
    }

    /// Returns a reference to the inner [`BoolVectorMut`] if `self` is of that variant.
    pub fn as_bool_mut(&mut self) -> &mut BoolVectorMut {
        if let VectorMut::Bool(v) = self {
            return v;
        }
        vortex_panic!("Expected BoolVectorMut, got {self:?}");
    }

    /// Returns a reference to the inner [`PrimitiveVectorMut`] if `self` is of that variant.
    pub fn as_primitive_mut(&mut self) -> &mut PrimitiveVectorMut {
        if let VectorMut::Primitive(v) = self {
            return v;
        }
        vortex_panic!("Expected PrimitiveVectorMut, got {self:?}");
    }

    /// Returns a reference to the inner [`StringVectorMut`] if `self` is of that variant.
    pub fn as_string_mut(&mut self) -> &mut StringVectorMut {
        if let VectorMut::String(v) = self {
            return v;
        }
        vortex_panic!("Expected StringVectorMut, got {self:?}");
    }

    /// Returns a reference to the inner [`BinaryVectorMut`] if `self` is of that variant.
    pub fn as_binary_mut(&mut self) -> &mut BinaryVectorMut {
        if let VectorMut::Binary(v) = self {
            return v;
        }
        vortex_panic!("Expected BinaryVectorMut, got {self:?}");
    }

    /// Returns a reference to the inner [`ListViewVectorMut`] if `self` is of that variant.
    pub fn as_list_mut(&mut self) -> &mut ListViewVectorMut {
        if let VectorMut::List(v) = self {
            return v;
        }
        vortex_panic!("Expected ListViewVectorMut, got {self:?}");
    }

    /// Returns a reference to the inner [`FixedSizeListVectorMut`] if `self` is of that variant.
    pub fn as_fixed_size_list_mut(&mut self) -> &mut FixedSizeListVectorMut {
        if let VectorMut::FixedSizeList(v) = self {
            return v;
        }
        vortex_panic!("Expected FixedSizeListVectorMut, got {self:?}");
    }

    /// Returns a reference to the inner [`StructVectorMut`] if `self` is of that variant.
    pub fn as_struct_mut(&mut self) -> &mut StructVectorMut {
        if let VectorMut::Struct(v) = self {
            return v;
        }
        vortex_panic!("Expected StructVectorMut, got {self:?}");
    }

    /// Consumes `self` and returns the inner [`NullVectorMut`] if `self` is of that variant.
    pub fn into_null(self) -> NullVectorMut {
        if let VectorMut::Null(v) = self {
            return v;
        }
        vortex_panic!("Expected NullVectorMut, got {self:?}");
    }

    /// Consumes `self` and returns the inner [`BoolVectorMut`] if `self` is of that variant.
    pub fn into_bool(self) -> BoolVectorMut {
        if let VectorMut::Bool(v) = self {
            return v;
        }
        vortex_panic!("Expected BoolVectorMut, got {self:?}");
    }

    /// Consumes `self` and returns the inner [`PrimitiveVectorMut`] if `self` is of that variant.
    pub fn into_primitive(self) -> PrimitiveVectorMut {
        if let VectorMut::Primitive(v) = self {
            return v;
        }
        vortex_panic!("Expected PrimitiveVectorMut, got {self:?}");
    }

    /// Consumes `self` and returns the inner [`StringVectorMut`] if `self` is of that variant.
    #[allow(clippy::same_name_method)] // Same as VarBinTypeDowncast
    pub fn into_string(self) -> StringVectorMut {
        if let VectorMut::String(v) = self {
            return v;
        }
        vortex_panic!("Expected StringVectorMut, got {self:?}");
    }

    /// Consumes `self` and returns the inner [`BinaryVectorMut`] if `self` is of that variant.
    #[allow(clippy::same_name_method)] // Same as VarBinTypeDowncast
    pub fn into_binary(self) -> BinaryVectorMut {
        if let VectorMut::Binary(v) = self {
            return v;
        }
        vortex_panic!("Expected BinaryVectorMut, got {self:?}");
    }

    /// Consumes `self` and returns the inner [`ListViewVectorMut`] if `self` is of that variant.
    pub fn into_list(self) -> ListViewVectorMut {
        if let VectorMut::List(v) = self {
            return v;
        }
        vortex_panic!("Expected ListViewVectorMut, got {self:?}");
    }

    /// Consumes `self` and returns the inner [`FixedSizeListVectorMut`] if `self` is of that
    /// variant.
    pub fn into_fixed_size_list(self) -> FixedSizeListVectorMut {
        if let VectorMut::FixedSizeList(v) = self {
            return v;
        }
        vortex_panic!("Expected FixedSizeListVectorMut, got {self:?}");
    }

    /// Consumes `self` and returns the inner [`StructVectorMut`] if `self` is of that variant.
    pub fn into_struct(self) -> StructVectorMut {
        if let VectorMut::Struct(v) = self {
            return v;
        }
        vortex_panic!("Expected StructVectorMut, got {self:?}");
    }
}

#[cfg(test)]
mod tests {
    use vortex_dtype::{DecimalDType, Nullability, PType};

    use super::*;
    use crate::VectorOps;
    use crate::decimal::DecimalVectorMut;
    use crate::primitive::PVectorMut;

    #[test]
    fn test_with_capacity() {
        // Test capacity allocation for different types.
        let null_vec = VectorMut::with_capacity(&DType::Null, 10);
        assert_eq!(null_vec.capacity(), usize::MAX); // Null vectors have unlimited capacity.

        let bool_vec = VectorMut::with_capacity(&DType::Bool(Nullability::Nullable), 100);
        assert!(bool_vec.capacity() >= 100);

        let prim_vec =
            VectorMut::with_capacity(&DType::Primitive(PType::I32, Nullability::Nullable), 50);
        assert!(prim_vec.capacity() >= 50);
    }

    #[test]
    fn test_with_capacity_decimal() {
        // Test decimal vectors with different precisions that map to different internal types.
        // Precision 1-2 uses i8, 3-4 uses i16, 5-9 uses i32, 10-18 uses i64,
        // 19-38 uses i128, 39-76 uses i256.

        // Test precision 4 (uses i16 internally).
        let decimal_dtype = DType::Decimal(DecimalDType::new(4, 2), Nullability::Nullable);
        let decimal_vec = VectorMut::with_capacity(&decimal_dtype, 50);

        match decimal_vec {
            VectorMut::Decimal(dec_vec) => {
                assert_eq!(dec_vec.len(), 0, "New vector should be empty");
                assert!(dec_vec.capacity() >= 50, "Capacity should be at least 50");

                // Verify it's using D16 variant internally.
                assert!(
                    matches!(dec_vec, DecimalVectorMut::D16(_)),
                    "Precision 4 should use D16 variant"
                );
            }
            _ => panic!("Expected decimal vector for decimal dtype"),
        }

        // Test precision 9 (uses i32 internally).
        let decimal_dtype = DType::Decimal(DecimalDType::new(9, 0), Nullability::NonNullable);
        let decimal_vec = VectorMut::with_capacity(&decimal_dtype, 100);

        match decimal_vec {
            VectorMut::Decimal(dec_vec) => {
                assert_eq!(dec_vec.len(), 0, "New vector should be empty");
                assert!(dec_vec.capacity() >= 100, "Capacity should be at least 100");

                // Verify it's using D32 variant internally.
                assert!(
                    matches!(dec_vec, DecimalVectorMut::D32(_)),
                    "Precision 9 should use D32 variant"
                );
            }
            _ => panic!("Expected decimal vector for decimal dtype"),
        }

        // Test precision 18 (uses i64 internally).
        let decimal_dtype = DType::Decimal(DecimalDType::new(18, -2), Nullability::Nullable);
        let decimal_vec = VectorMut::with_capacity(&decimal_dtype, 75);

        match decimal_vec {
            VectorMut::Decimal(dec_vec) => {
                assert_eq!(dec_vec.len(), 0, "New vector should be empty");
                assert!(dec_vec.capacity() >= 75, "Capacity should be at least 75");

                // Verify it's using D64 variant internally.
                assert!(
                    matches!(dec_vec, DecimalVectorMut::D64(_)),
                    "Precision 18 should use D64 variant"
                );
            }
            _ => panic!("Expected decimal vector for decimal dtype"),
        }

        // Test precision 38 (uses i128 internally).
        let decimal_dtype = DType::Decimal(DecimalDType::new(38, 10), Nullability::NonNullable);
        let decimal_vec = VectorMut::with_capacity(&decimal_dtype, 25);

        match decimal_vec {
            VectorMut::Decimal(dec_vec) => {
                assert_eq!(dec_vec.len(), 0, "New vector should be empty");
                assert!(dec_vec.capacity() >= 25, "Capacity should be at least 25");

                // Verify it's using D128 variant internally.
                assert!(
                    matches!(dec_vec, DecimalVectorMut::D128(_)),
                    "Precision 38 should use D128 variant"
                );
            }
            _ => panic!("Expected decimal vector for decimal dtype"),
        }
    }

    #[test]
    #[should_panic(expected = "Mismatched vector types")]
    fn test_type_mismatch_panics() {
        // Test that operations between mismatched types panic.
        let mut vec1 = VectorMut::with_capacity(&DType::Bool(Nullability::Nullable), 10);
        let vec2 =
            VectorMut::with_capacity(&DType::Primitive(PType::I32, Nullability::Nullable), 10);

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
