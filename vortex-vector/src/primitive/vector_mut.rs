// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::half::f16;
use vortex_dtype::{DType, Nullability, PType};

use crate::{
    GenericPVectorMut, PrimitiveVector, VectorMut, VectorMutOps, match_each_pvector_mut,
    match_each_pvector_mut_immut_pair, match_each_pvector_mut_pair,
};

/// An enum over all mutable primitive vector types.
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

impl From<PrimitiveVectorMut> for VectorMut {
    fn from(v: PrimitiveVectorMut) -> Self {
        Self::Primitive(v)
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
