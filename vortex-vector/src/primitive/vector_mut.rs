// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Definition and implementation of [`PrimitiveVectorMut`].

use vortex_dtype::NativePType;
use vortex_dtype::PType;
use vortex_dtype::PTypeDowncast;
use vortex_dtype::PTypeUpcast;
use vortex_dtype::half::f16;
use vortex_error::vortex_panic;
use vortex_mask::MaskMut;

use crate::VectorMutOps;
use crate::match_each_pvector_mut;
use crate::primitive::PVectorMut;
use crate::primitive::PrimitiveScalar;
use crate::primitive::PrimitiveVector;

/// A mutable vector of primitive values.
///
/// The immutable equivalent of this type is [`PrimitiveVector`].
///
/// `PrimitiveVector` is represented by an enum over all possible [`PVectorMut`] types (which are
/// templated by the types that implement [`NativePType`]).
///
/// See the documentation for [`PVectorMut`] for more information.
#[derive(Debug, Clone)]
pub enum PrimitiveVectorMut {
    /// U8
    U8(PVectorMut<u8>),
    /// U16
    U16(PVectorMut<u16>),
    /// U32
    U32(PVectorMut<u32>),
    /// U64
    U64(PVectorMut<u64>),
    /// I8
    I8(PVectorMut<i8>),
    /// I16
    I16(PVectorMut<i16>),
    /// I32
    I32(PVectorMut<i32>),
    /// I64
    I64(PVectorMut<i64>),
    /// F16
    F16(PVectorMut<f16>),
    /// F32
    F32(PVectorMut<f32>),
    /// F64
    F64(PVectorMut<f64>),
}

impl PrimitiveVectorMut {
    /// Returns the [`PType`] of this [`PrimitiveVectorMut`].
    pub fn ptype(&self) -> PType {
        match self {
            Self::U8(_) => PType::U8,
            Self::U16(_) => PType::U16,
            Self::U32(_) => PType::U32,
            Self::U64(_) => PType::U64,
            Self::I8(_) => PType::I8,
            Self::I16(_) => PType::I16,
            Self::I32(_) => PType::I32,
            Self::I64(_) => PType::I64,
            Self::F16(_) => PType::F16,
            Self::F32(_) => PType::F32,
            Self::F64(_) => PType::F64,
        }
    }

    /// Create a new mutable primitive vector with the given primitive type and capacity.
    pub fn with_capacity(ptype: PType, capacity: usize) -> Self {
        match ptype {
            PType::U8 => PVectorMut::<u8>::with_capacity(capacity).into(),
            PType::U16 => PVectorMut::<u16>::with_capacity(capacity).into(),
            PType::U32 => PVectorMut::<u32>::with_capacity(capacity).into(),
            PType::U64 => PVectorMut::<u64>::with_capacity(capacity).into(),
            PType::I8 => PVectorMut::<i8>::with_capacity(capacity).into(),
            PType::I16 => PVectorMut::<i16>::with_capacity(capacity).into(),
            PType::I32 => PVectorMut::<i32>::with_capacity(capacity).into(),
            PType::I64 => PVectorMut::<i64>::with_capacity(capacity).into(),
            PType::F16 => PVectorMut::<f16>::with_capacity(capacity).into(),
            PType::F32 => PVectorMut::<f32>::with_capacity(capacity).into(),
            PType::F64 => PVectorMut::<f64>::with_capacity(capacity).into(),
        }
    }
}

impl VectorMutOps for PrimitiveVectorMut {
    type Immutable = PrimitiveVector;

    fn len(&self) -> usize {
        match_each_pvector_mut!(self, |v| { v.len() })
    }

    fn validity(&self) -> &MaskMut {
        match_each_pvector_mut!(self, |v| { v.validity() })
    }

    fn capacity(&self) -> usize {
        match_each_pvector_mut!(self, |v| { v.capacity() })
    }

    fn reserve(&mut self, additional: usize) {
        match_each_pvector_mut!(self, |v| { v.reserve(additional) })
    }

    fn clear(&mut self) {
        match_each_pvector_mut!(self, |v| { v.clear() })
    }

    fn truncate(&mut self, len: usize) {
        match_each_pvector_mut!(self, |v| { v.truncate(len) })
    }

    fn extend_from_vector(&mut self, other: &PrimitiveVector) {
        match (self, other) {
            (Self::U8(a), PrimitiveVector::U8(b)) => a.extend_from_vector(b),
            (Self::U16(a), PrimitiveVector::U16(b)) => a.extend_from_vector(b),
            (Self::U32(a), PrimitiveVector::U32(b)) => a.extend_from_vector(b),
            (Self::U64(a), PrimitiveVector::U64(b)) => a.extend_from_vector(b),
            (Self::I8(a), PrimitiveVector::I8(b)) => a.extend_from_vector(b),
            (Self::I16(a), PrimitiveVector::I16(b)) => a.extend_from_vector(b),
            (Self::I32(a), PrimitiveVector::I32(b)) => a.extend_from_vector(b),
            (Self::I64(a), PrimitiveVector::I64(b)) => a.extend_from_vector(b),
            (Self::F16(a), PrimitiveVector::F16(b)) => a.extend_from_vector(b),
            (Self::F32(a), PrimitiveVector::F32(b)) => a.extend_from_vector(b),
            (Self::F64(a), PrimitiveVector::F64(b)) => a.extend_from_vector(b),
            _ => ::vortex_error::vortex_panic!("Mismatched primitive vector types"),
        }
    }

    fn append_nulls(&mut self, n: usize) {
        match_each_pvector_mut!(self, |v| { v.append_nulls(n) })
    }

    fn append_zeros(&mut self, n: usize) {
        match_each_pvector_mut!(self, |v| { v.append_zeros(n) })
    }

    #[expect(
        clippy::many_single_char_names,
        reason = "single-letter names a/b are clear for matching variants"
    )]
    fn append_scalars(&mut self, scalar: &PrimitiveScalar, n: usize) {
        match (self, scalar) {
            (Self::U8(a), PrimitiveScalar::U8(b)) => a.append_scalars(b, n),
            (Self::U16(a), PrimitiveScalar::U16(b)) => a.append_scalars(b, n),
            (Self::U32(a), PrimitiveScalar::U32(b)) => a.append_scalars(b, n),
            (Self::U64(a), PrimitiveScalar::U64(b)) => a.append_scalars(b, n),
            (Self::I8(a), PrimitiveScalar::I8(b)) => a.append_scalars(b, n),
            (Self::I16(a), PrimitiveScalar::I16(b)) => a.append_scalars(b, n),
            (Self::I32(a), PrimitiveScalar::I32(b)) => a.append_scalars(b, n),
            (Self::I64(a), PrimitiveScalar::I64(b)) => a.append_scalars(b, n),
            (Self::F16(a), PrimitiveScalar::F16(b)) => a.append_scalars(b, n),
            (Self::F32(a), PrimitiveScalar::F32(b)) => a.append_scalars(b, n),
            (Self::F64(a), PrimitiveScalar::F64(b)) => a.append_scalars(b, n),
            _ => vortex_panic!("Mismatched primitive vector and scalar types"),
        }
    }

    fn freeze(self) -> PrimitiveVector {
        match_each_pvector_mut!(self, |v| { v.freeze().into() })
    }

    fn split_off(&mut self, at: usize) -> Self {
        match_each_pvector_mut!(self, |v| { v.split_off(at).into() })
    }

    fn unsplit(&mut self, other: Self) {
        match (self, other) {
            (Self::U8(a), Self::U8(b)) => a.unsplit(b),
            (Self::U16(a), Self::U16(b)) => a.unsplit(b),
            (Self::U32(a), Self::U32(b)) => a.unsplit(b),
            (Self::U64(a), Self::U64(b)) => a.unsplit(b),
            (Self::I8(a), Self::I8(b)) => a.unsplit(b),
            (Self::I16(a), Self::I16(b)) => a.unsplit(b),
            (Self::I32(a), Self::I32(b)) => a.unsplit(b),
            (Self::I64(a), Self::I64(b)) => a.unsplit(b),
            (Self::F16(a), Self::F16(b)) => a.unsplit(b),
            (Self::F32(a), Self::F32(b)) => a.unsplit(b),
            (Self::F64(a), Self::F64(b)) => a.unsplit(b),
            _ => vortex_panic!("Mismatched primitive vector types"),
        }
    }
}

impl PTypeUpcast for PrimitiveVectorMut {
    type Input<T: NativePType> = PVectorMut<T>;

    fn from_u8(input: Self::Input<u8>) -> Self {
        Self::U8(input)
    }

    fn from_u16(input: Self::Input<u16>) -> Self {
        Self::U16(input)
    }

    fn from_u32(input: Self::Input<u32>) -> Self {
        Self::U32(input)
    }

    fn from_u64(input: Self::Input<u64>) -> Self {
        Self::U64(input)
    }

    fn from_i8(input: Self::Input<i8>) -> Self {
        Self::I8(input)
    }

    fn from_i16(input: Self::Input<i16>) -> Self {
        Self::I16(input)
    }

    fn from_i32(input: Self::Input<i32>) -> Self {
        Self::I32(input)
    }

    fn from_i64(input: Self::Input<i64>) -> Self {
        Self::I64(input)
    }

    fn from_f16(input: Self::Input<f16>) -> Self {
        Self::F16(input)
    }

    fn from_f32(input: Self::Input<f32>) -> Self {
        Self::F32(input)
    }

    fn from_f64(input: Self::Input<f64>) -> Self {
        Self::F64(input)
    }
}

impl PTypeDowncast for PrimitiveVectorMut {
    type Output<T: NativePType> = PVectorMut<T>;

    fn into_u8(self) -> Self::Output<u8> {
        if let Self::U8(v) = self {
            return v;
        }
        vortex_panic!("Expected PrimitiveVectorMut::U8, got {self:?}");
    }

    fn into_u16(self) -> Self::Output<u16> {
        if let Self::U16(v) = self {
            return v;
        }
        vortex_panic!("Expected PrimitiveVectorMut::U16, got {self:?}");
    }

    fn into_u32(self) -> Self::Output<u32> {
        if let Self::U32(v) = self {
            return v;
        }
        vortex_panic!("Expected PrimitiveVectorMut::U32, got {self:?}");
    }

    fn into_u64(self) -> Self::Output<u64> {
        if let Self::U64(v) = self {
            return v;
        }
        vortex_panic!("Expected PrimitiveVectorMut::U64, got {self:?}");
    }

    fn into_i8(self) -> Self::Output<i8> {
        if let Self::I8(v) = self {
            return v;
        }
        vortex_panic!("Expected PrimitiveVectorMut::I8, got {self:?}");
    }

    fn into_i16(self) -> Self::Output<i16> {
        if let Self::I16(v) = self {
            return v;
        }
        vortex_panic!("Expected PrimitiveVectorMut::I16, got {self:?}");
    }

    fn into_i32(self) -> Self::Output<i32> {
        if let Self::I32(v) = self {
            return v;
        }
        vortex_panic!("Expected PrimitiveVectorMut::I32, got {self:?}");
    }

    fn into_i64(self) -> Self::Output<i64> {
        if let Self::I64(v) = self {
            return v;
        }
        vortex_panic!("Expected PrimitiveVectorMut::I64, got {self:?}");
    }

    fn into_f16(self) -> Self::Output<f16> {
        if let Self::F16(v) = self {
            return v;
        }
        vortex_panic!("Expected PrimitiveVectorMut::F16, got {self:?}");
    }

    fn into_f32(self) -> Self::Output<f32> {
        if let Self::F32(v) = self {
            return v;
        }
        vortex_panic!("Expected PrimitiveVectorMut::F32, got {self:?}");
    }

    fn into_f64(self) -> Self::Output<f64> {
        if let Self::F64(v) = self {
            return v;
        }
        vortex_panic!("Expected PrimitiveVectorMut::F64, got {self:?}");
    }
}

impl<'a> PTypeDowncast for &'a mut PrimitiveVectorMut {
    type Output<T: NativePType> = &'a mut PVectorMut<T>;

    fn into_u8(self) -> Self::Output<u8> {
        match self {
            PrimitiveVectorMut::U8(v) => v,
            _ => vortex_panic!("Expected PrimitiveVectorMut::U8, got {self:?}"),
        }
    }

    fn into_u16(self) -> Self::Output<u16> {
        match self {
            PrimitiveVectorMut::U16(v) => v,
            _ => vortex_panic!("Expected PrimitiveVectorMut::U16, got {self:?}"),
        }
    }

    fn into_u32(self) -> Self::Output<u32> {
        match self {
            PrimitiveVectorMut::U32(v) => v,
            _ => vortex_panic!("Expected PrimitiveVectorMut::U32, got {self:?}"),
        }
    }

    fn into_u64(self) -> Self::Output<u64> {
        match self {
            PrimitiveVectorMut::U64(v) => v,
            _ => vortex_panic!("Expected PrimitiveVectorMut::U64, got {self:?}"),
        }
    }

    fn into_i8(self) -> Self::Output<i8> {
        match self {
            PrimitiveVectorMut::I8(v) => v,
            _ => vortex_panic!("Expected PrimitiveVectorMut::I8, got {self:?}"),
        }
    }

    fn into_i16(self) -> Self::Output<i16> {
        match self {
            PrimitiveVectorMut::I16(v) => v,
            _ => vortex_panic!("Expected PrimitiveVectorMut::I16, got {self:?}"),
        }
    }

    fn into_i32(self) -> Self::Output<i32> {
        match self {
            PrimitiveVectorMut::I32(v) => v,
            _ => vortex_panic!("Expected PrimitiveVectorMut::I32, got {self:?}"),
        }
    }

    fn into_i64(self) -> Self::Output<i64> {
        match self {
            PrimitiveVectorMut::I64(v) => v,
            _ => vortex_panic!("Expected PrimitiveVectorMut::I64, got {self:?}"),
        }
    }

    fn into_f16(self) -> Self::Output<f16> {
        match self {
            PrimitiveVectorMut::F16(v) => v,
            _ => vortex_panic!("Expected PrimitiveVectorMut::F16, got {self:?}"),
        }
    }

    fn into_f32(self) -> Self::Output<f32> {
        match self {
            PrimitiveVectorMut::F32(v) => v,
            _ => vortex_panic!("Expected PrimitiveVectorMut::F32, got {self:?}"),
        }
    }

    fn into_f64(self) -> Self::Output<f64> {
        match self {
            PrimitiveVectorMut::F64(v) => v,
            _ => vortex_panic!("Expected PrimitiveVectorMut::F64, got {self:?}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::VectorOps;

    #[test]
    fn test_from_iter_with_options() {
        // Test FromIterator<Option<T>> with different types.
        let vec_i32: PrimitiveVectorMut =
            PVectorMut::<i32>::from_iter(vec![Some(1), None, Some(3), None, Some(5)]).into();
        assert_eq!(vec_i32.len(), 5);
        let frozen = vec_i32.freeze();
        assert_eq!(frozen.validity().true_count(), 3);

        // Test empty iterator.
        let vec_empty: PrimitiveVectorMut =
            PVectorMut::<f64>::from_iter(std::iter::empty::<Option<f64>>()).into();
        assert_eq!(vec_empty.len(), 0);

        // Test that None values use T::default().
        let vec_nulls: PrimitiveVectorMut = PVectorMut::<i32>::from_iter([None, None, None]).into();
        // Check that validity is all false for nulls.
        let frozen = vec_nulls.freeze();
        assert_eq!(frozen.validity().true_count(), 0);
    }

    #[test]
    fn test_from_iter_non_null() {
        // Test FromIterator<T> for different primitive types.
        let vec_f64: PrimitiveVectorMut =
            PVectorMut::<f64>::from_iter([1.5, 2.5, 3.5, 4.5, 5.5]).into();
        assert_eq!(vec_f64.len(), 5);
        let frozen = vec_f64.freeze();
        assert_eq!(frozen.validity().true_count(), 5); // All valid.

        let vec_u16: PrimitiveVectorMut = PVectorMut::<u16>::from_iter([1u16, 2, 3, 4, 5]).into();
        assert_eq!(vec_u16.len(), 5);
        let frozen = vec_u16.freeze();
        assert_eq!(frozen.validity().true_count(), 5);
    }

    #[test]
    fn test_operations_preserve_validity() {
        // Test split/unsplit/extend with different primitive types.
        let mut vec: PrimitiveVectorMut =
            PVectorMut::<i64>::from_iter([Some(100), None, Some(300), None, Some(500)]).into();

        let second_half = vec.split_off(2);
        assert_eq!(vec.len(), 2);
        assert_eq!(second_half.len(), 3);

        let first_frozen = vec.freeze();
        let second_frozen = second_half.freeze();
        assert_eq!(first_frozen.validity().true_count(), 1);
        assert_eq!(second_frozen.validity().true_count(), 2);

        // Test unsplit.
        let mut vec1: PrimitiveVectorMut = PVectorMut::<u32>::from_iter([Some(1000), None]).into();
        let vec2: PrimitiveVectorMut = PVectorMut::<u32>::from_iter([None, Some(2000)]).into();
        vec1.unsplit(vec2);
        assert_eq!(vec1.len(), 4);
        let frozen = vec1.freeze();
        assert_eq!(frozen.validity().true_count(), 2);
    }
}
