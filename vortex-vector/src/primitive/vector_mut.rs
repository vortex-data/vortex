// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Definition and implementation of [`PrimitiveVectorMut`].

use vortex_dtype::half::f16;
use vortex_dtype::{DType, NativePType, Nullability, PType, PTypeUpcast};

use crate::{
    GenericPVectorMut, PrimitiveVector, VectorMutOps, match_each_pvector_mut,
    match_each_pvector_mut_immut_pair, match_each_pvector_mut_pair,
};

/// A mutable vector of primitive values.
///
/// `PrimitiveVector` is represented by an enum over all possible [`GenericPVectorMut`] types (which
/// are templated by the types that implement [`NativePType`]).
///
/// The immutable equivalent of this type is [`PrimitiveVector`].
#[derive(Debug, Clone)]
pub enum PrimitiveVectorMut {
    /// U8
    U8(GenericPVectorMut<u8>),
    /// U16
    U16(GenericPVectorMut<u16>),
    /// U32
    U32(GenericPVectorMut<u32>),
    /// U64
    U64(GenericPVectorMut<u64>),
    /// I8
    I8(GenericPVectorMut<i8>),
    /// I16
    I16(GenericPVectorMut<i16>),
    /// I32
    I32(GenericPVectorMut<i32>),
    /// I64
    I64(GenericPVectorMut<i64>),
    /// F16
    F16(GenericPVectorMut<f16>),
    /// F32
    F32(GenericPVectorMut<f32>),
    /// F64
    F64(GenericPVectorMut<f64>),
}

impl PrimitiveVectorMut {
    /// Create a new mutable primitive vector with the given capacity, primitive type, and nullability.
    pub fn with_capacity(capacity: usize, ptype: PType, nullability: Nullability) -> Self {
        match ptype {
            PType::U8 => GenericPVectorMut::<u8>::with_capacity(capacity, nullability).into(),
            PType::U16 => GenericPVectorMut::<u16>::with_capacity(capacity, nullability).into(),
            PType::U32 => GenericPVectorMut::<u32>::with_capacity(capacity, nullability).into(),
            PType::U64 => GenericPVectorMut::<u64>::with_capacity(capacity, nullability).into(),
            PType::I8 => GenericPVectorMut::<i8>::with_capacity(capacity, nullability).into(),
            PType::I16 => GenericPVectorMut::<i16>::with_capacity(capacity, nullability).into(),
            PType::I32 => GenericPVectorMut::<i32>::with_capacity(capacity, nullability).into(),
            PType::I64 => GenericPVectorMut::<i64>::with_capacity(capacity, nullability).into(),
            PType::F16 => GenericPVectorMut::<f16>::with_capacity(capacity, nullability).into(),
            PType::F32 => GenericPVectorMut::<f32>::with_capacity(capacity, nullability).into(),
            PType::F64 => GenericPVectorMut::<f64>::with_capacity(capacity, nullability).into(),
        }
    }
}

impl VectorMutOps for PrimitiveVectorMut {
    type Immutable = PrimitiveVector;

    fn nullability(&self) -> Nullability {
        match_each_pvector_mut!(self, |v| { v.nullability() })
    }

    fn dtype(&self) -> DType {
        match_each_pvector_mut!(self, |v| { v.dtype() })
    }

    fn len(&self) -> usize {
        match_each_pvector_mut!(self, |v| { v.len() })
    }

    fn capacity(&self) -> usize {
        match_each_pvector_mut!(self, |v| { v.capacity() })
    }

    fn reserve(&mut self, additional: usize) {
        match_each_pvector_mut!(self, |v| { v.reserve(additional) })
    }

    fn extend_from_vector(&mut self, other: &Self::Immutable) {
        match_each_pvector_mut_immut_pair!(self, other, |a, b| {
            a.extend_from_vector(b);
        });
    }

    fn append_nulls(&mut self, n: usize) {
        match_each_pvector_mut!(self, |v| { v.append_nulls(n) })
    }

    fn freeze(self) -> Self::Immutable {
        match_each_pvector_mut!(self, |v| { v.freeze().into() })
    }

    fn split_off(&mut self, at: usize) -> Self {
        match_each_pvector_mut!(self, |v| { v.split_off(at).into() })
    }

    fn unsplit(&mut self, other: Self) {
        match_each_pvector_mut_pair!(self, other, |a, b| {
            a.unsplit(b);
        });
    }
}

impl<T: NativePType> From<GenericPVectorMut<T>> for PrimitiveVectorMut {
    fn from(v: GenericPVectorMut<T>) -> Self {
        T::upcast(v)
    }
}

impl PTypeUpcast for PrimitiveVectorMut {
    type Input<T: NativePType> = GenericPVectorMut<T>;

    fn from_u8(input: Self::Input<u8>) -> Self {
        PrimitiveVectorMut::U8(input)
    }

    fn from_u16(input: Self::Input<u16>) -> Self {
        PrimitiveVectorMut::U16(input)
    }

    fn from_u32(input: Self::Input<u32>) -> Self {
        PrimitiveVectorMut::U32(input)
    }

    fn from_u64(input: Self::Input<u64>) -> Self {
        PrimitiveVectorMut::U64(input)
    }

    fn from_i8(input: Self::Input<i8>) -> Self {
        PrimitiveVectorMut::I8(input)
    }

    fn from_i16(input: Self::Input<i16>) -> Self {
        PrimitiveVectorMut::I16(input)
    }

    fn from_i32(input: Self::Input<i32>) -> Self {
        PrimitiveVectorMut::I32(input)
    }

    fn from_i64(input: Self::Input<i64>) -> Self {
        PrimitiveVectorMut::I64(input)
    }

    fn from_f16(input: Self::Input<f16>) -> Self {
        PrimitiveVectorMut::F16(input)
    }

    fn from_f32(input: Self::Input<f32>) -> Self {
        PrimitiveVectorMut::F32(input)
    }

    fn from_f64(input: Self::Input<f64>) -> Self {
        PrimitiveVectorMut::F64(input)
    }
}
