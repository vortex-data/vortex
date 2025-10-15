// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::half::f16;
use vortex_dtype::{DType, NativePType, Nullability, PType, PTypeUpcast};
use vortex_error::vortex_panic;

use crate::ops::{VectorMutOps, VectorOps};
use crate::{PrimitiveVector, PrimitiveVectorMut, Vector};

/// An enum over all primitive vector types.
pub enum PVector {
    /// U8
    U8(PrimitiveVector<u8>),
    /// U16
    U16(PrimitiveVector<u16>),
    /// U32
    U32(PrimitiveVector<u32>),
    /// U64
    U64(PrimitiveVector<u64>),
    /// I8
    I8(PrimitiveVector<i8>),
    /// I16
    I16(PrimitiveVector<i16>),
    /// I32
    I32(PrimitiveVector<i32>),
    /// I64
    I64(PrimitiveVector<i64>),
    /// F16
    F16(PrimitiveVector<f16>),
    /// F32
    F32(PrimitiveVector<f32>),
    /// F64
    F64(PrimitiveVector<f64>),
}

/// An enum over all mutable primitive vector types.
pub enum PVectorMut {
    /// U8
    U8(PrimitiveVectorMut<u8>),
    /// U16
    U16(PrimitiveVectorMut<u16>),
    /// U32
    U32(PrimitiveVectorMut<u32>),
    /// U64
    U64(PrimitiveVectorMut<u64>),
    /// I8
    I8(PrimitiveVectorMut<i8>),
    /// I16
    I16(PrimitiveVectorMut<i16>),
    /// I32
    I32(PrimitiveVectorMut<i32>),
    /// I64
    I64(PrimitiveVectorMut<i64>),
    /// F16
    F16(PrimitiveVectorMut<f16>),
    /// F32
    F32(PrimitiveVectorMut<f32>),
    /// F64
    F64(PrimitiveVectorMut<f64>),
}

impl PVectorMut {
    /// Create a new mutable primitive vector with the given capacity, primitive type, and nullability.
    pub fn with_capacity(capacity: usize, ptype: PType, nullability: Nullability) -> Self {
        match ptype {
            PType::U8 => PrimitiveVectorMut::<u8>::with_capacity(capacity, nullability).into(),
            PType::U16 => PrimitiveVectorMut::<u16>::with_capacity(capacity, nullability).into(),
            PType::U32 => PrimitiveVectorMut::<u32>::with_capacity(capacity, nullability).into(),
            PType::U64 => PrimitiveVectorMut::<u64>::with_capacity(capacity, nullability).into(),
            PType::I8 => PrimitiveVectorMut::<i8>::with_capacity(capacity, nullability).into(),
            PType::I16 => PrimitiveVectorMut::<i16>::with_capacity(capacity, nullability).into(),
            PType::I32 => PrimitiveVectorMut::<i32>::with_capacity(capacity, nullability).into(),
            PType::I64 => PrimitiveVectorMut::<i64>::with_capacity(capacity, nullability).into(),
            PType::F16 => PrimitiveVectorMut::<f16>::with_capacity(capacity, nullability).into(),
            PType::F32 => PrimitiveVectorMut::<f32>::with_capacity(capacity, nullability).into(),
            PType::F64 => PrimitiveVectorMut::<f64>::with_capacity(capacity, nullability).into(),
        }
    }
}

macro_rules! match_each_pvector {
    ($self:expr, | $vec:ident | $body:block) => {{
        match $self {
            PVector::U8(v) => {
                let $vec = v;
                $body
            }
            PVector::U16(v) => {
                let $vec = v;
                $body
            }
            PVector::U32(v) => {
                let $vec = v;
                $body
            }
            PVector::U64(v) => {
                let $vec = v;
                $body
            }
            PVector::I8(v) => {
                let $vec = v;
                $body
            }
            PVector::I16(v) => {
                let $vec = v;
                $body
            }
            PVector::I32(v) => {
                let $vec = v;
                $body
            }
            PVector::I64(v) => {
                let $vec = v;
                $body
            }
            PVector::F16(v) => {
                let $vec = v;
                $body
            }
            PVector::F32(v) => {
                let $vec = v;
                $body
            }
            PVector::F64(v) => {
                let $vec = v;
                $body
            }
        }
    }};
}

macro_rules! match_each_pvector_mut {
    ($self:expr, | $vec:ident | $body:block) => {{
        match $self {
            PVectorMut::U8(v) => {
                let $vec = v;
                $body
            }
            PVectorMut::U16(v) => {
                let $vec = v;
                $body
            }
            PVectorMut::U32(v) => {
                let $vec = v;
                $body
            }
            PVectorMut::U64(v) => {
                let $vec = v;
                $body
            }
            PVectorMut::I8(v) => {
                let $vec = v;
                $body
            }
            PVectorMut::I16(v) => {
                let $vec = v;
                $body
            }
            PVectorMut::I32(v) => {
                let $vec = v;
                $body
            }
            PVectorMut::I64(v) => {
                let $vec = v;
                $body
            }
            PVectorMut::F16(v) => {
                let $vec = v;
                $body
            }
            PVectorMut::F32(v) => {
                let $vec = v;
                $body
            }
            PVectorMut::F64(v) => {
                let $vec = v;
                $body
            }
        }
    }};
}

macro_rules! match_each_pvector_mut_pair {
    ($self:expr, $other:expr, | $vec:ident, $vec_other:ident | $body:block) => {{
        match ($self, $other) {
            (PVectorMut::U8(a), PVectorMut::U8(b)) => {
                let $vec = a;
                let $vec_other = b;
                $body
            }
            (PVectorMut::U16(a), PVectorMut::U16(b)) => {
                let $vec = a;
                let $vec_other = b;
                $body
            }
            (PVectorMut::U32(a), PVectorMut::U32(b)) => {
                let $vec = a;
                let $vec_other = b;
                $body
            }
            (PVectorMut::U64(a), PVectorMut::U64(b)) => {
                let $vec = a;
                let $vec_other = b;
                $body
            }
            (PVectorMut::I8(a), PVectorMut::I8(b)) => {
                let $vec = a;
                let $vec_other = b;
                $body
            }
            (PVectorMut::I16(a), PVectorMut::I16(b)) => {
                let $vec = a;
                let $vec_other = b;
                $body
            }
            (PVectorMut::I32(a), PVectorMut::I32(b)) => {
                let $vec = a;
                let $vec_other = b;
                $body
            }
            (PVectorMut::I64(a), PVectorMut::I64(b)) => {
                let $vec = a;
                let $vec_other = b;
                $body
            }
            (PVectorMut::F16(a), PVectorMut::F16(b)) => {
                let $vec = a;
                let $vec_other = b;
                $body
            }
            (PVectorMut::F32(a), PVectorMut::F32(b)) => {
                let $vec = a;
                let $vec_other = b;
                $body
            }
            (PVectorMut::F64(a), PVectorMut::F64(b)) => {
                let $vec = a;
                let $vec_other = b;
                $body
            }
            _ => vortex_panic!("Mismatched primitive vector types"),
        }
    }};
}

macro_rules! match_each_pvector_mut_immut_pair {
    ($self:expr, $other:expr, | $vec:ident, $vec_other:ident | $body:block) => {{
        match ($self, $other) {
            (PVectorMut::U8(a), PVector::U8(b)) => {
                let $vec = a;
                let $vec_other = b;
                $body
            }
            (PVectorMut::U16(a), PVector::U16(b)) => {
                let $vec = a;
                let $vec_other = b;
                $body
            }
            (PVectorMut::U32(a), PVector::U32(b)) => {
                let $vec = a;
                let $vec_other = b;
                $body
            }
            (PVectorMut::U64(a), PVector::U64(b)) => {
                let $vec = a;
                let $vec_other = b;
                $body
            }
            (PVectorMut::I8(a), PVector::I8(b)) => {
                let $vec = a;
                let $vec_other = b;
                $body
            }
            (PVectorMut::I16(a), PVector::I16(b)) => {
                let $vec = a;
                let $vec_other = b;
                $body
            }
            (PVectorMut::I32(a), PVector::I32(b)) => {
                let $vec = a;
                let $vec_other = b;
                $body
            }
            (PVectorMut::I64(a), PVector::I64(b)) => {
                let $vec = a;
                let $vec_other = b;
                $body
            }
            (PVectorMut::F16(a), PVector::F16(b)) => {
                let $vec = a;
                let $vec_other = b;
                $body
            }
            (PVectorMut::F32(a), PVector::F32(b)) => {
                let $vec = a;
                let $vec_other = b;
                $body
            }
            (PVectorMut::F64(a), PVector::F64(b)) => {
                let $vec = a;
                let $vec_other = b;
                $body
            }
            _ => vortex_panic!("Mismatched primitive vector types"),
        }
    }};
}

impl From<PVector> for Vector {
    fn from(v: PVector) -> Self {
        Self::Primitive(v)
    }
}

impl VectorOps for PVector {
    type Mutable = PVectorMut;

    fn len(&self) -> usize {
        match_each_pvector!(self, |v| { v.len() })
    }

    fn dtype(&self) -> &DType {
        match_each_pvector!(self, |v| { v.dtype() })
    }

    fn try_into_mut(self) -> Result<Self::Mutable, Self>
    where
        Self: Sized,
    {
        match_each_pvector!(self, |v| {
            v.try_into_mut()
                .map(PVectorMut::from)
                .map_err(PVector::from)
        })
    }
}

impl From<PVectorMut> for crate::VectorMut {
    fn from(v: PVectorMut) -> Self {
        Self::Primitive(v)
    }
}

impl VectorMutOps for PVectorMut {
    type Immutable = PVector;

    fn len(&self) -> usize {
        match_each_pvector_mut!(self, |v| { v.len() })
    }

    fn dtype(&self) -> &DType {
        match_each_pvector_mut!(self, |v| { v.dtype() })
    }

    fn capacity(&self) -> usize {
        match_each_pvector_mut!(self, |v| { v.capacity() })
    }

    fn reserve(&mut self, additional: usize) {
        match_each_pvector_mut!(self, |v| { v.reserve(additional) })
    }

    fn split_off(&mut self, at: usize) -> Self {
        match_each_pvector_mut!(self, |v| { v.split_off(at).into() })
    }

    fn unsplit(&mut self, other: Self) {
        match_each_pvector_mut_pair!(self, other, |a, b| {
            a.unsplit(b);
        });
    }

    fn extend_from_vector(&mut self, other: &Self::Immutable) {
        match_each_pvector_mut_immut_pair!(self, other, |a, b| {
            a.extend_from_vector(b);
        });
    }

    fn freeze(self) -> Self::Immutable {
        match_each_pvector_mut!(self, |v| { v.freeze().into() })
    }
}

// From impls for PVector variants
impl<T: NativePType> From<PrimitiveVector<T>> for PVector {
    fn from(v: PrimitiveVector<T>) -> Self {
        T::upcast(v)
    }
}

impl PTypeUpcast for PVector {
    type Input<T: NativePType> = PrimitiveVector<T>;

    fn from_u8(input: Self::Input<u8>) -> Self {
        PVector::U8(input)
    }

    fn from_u16(input: Self::Input<u16>) -> Self {
        PVector::U16(input)
    }

    fn from_u32(input: Self::Input<u32>) -> Self {
        PVector::U32(input)
    }

    fn from_u64(input: Self::Input<u64>) -> Self {
        PVector::U64(input)
    }

    fn from_i8(input: Self::Input<i8>) -> Self {
        PVector::I8(input)
    }

    fn from_i16(input: Self::Input<i16>) -> Self {
        PVector::I16(input)
    }

    fn from_i32(input: Self::Input<i32>) -> Self {
        PVector::I32(input)
    }

    fn from_i64(input: Self::Input<i64>) -> Self {
        PVector::I64(input)
    }

    fn from_f16(input: Self::Input<f16>) -> Self {
        PVector::F16(input)
    }

    fn from_f32(input: Self::Input<f32>) -> Self {
        PVector::F32(input)
    }

    fn from_f64(input: Self::Input<f64>) -> Self {
        PVector::F64(input)
    }
}

// From impls for PVectorMut variants
impl<T: NativePType> From<PrimitiveVectorMut<T>> for PVectorMut {
    fn from(v: PrimitiveVectorMut<T>) -> Self {
        T::upcast(v)
    }
}

impl PTypeUpcast for PVectorMut {
    type Input<T: NativePType> = PrimitiveVectorMut<T>;

    fn from_u8(input: Self::Input<u8>) -> Self {
        PVectorMut::U8(input)
    }

    fn from_u16(input: Self::Input<u16>) -> Self {
        PVectorMut::U16(input)
    }

    fn from_u32(input: Self::Input<u32>) -> Self {
        PVectorMut::U32(input)
    }

    fn from_u64(input: Self::Input<u64>) -> Self {
        PVectorMut::U64(input)
    }

    fn from_i8(input: Self::Input<i8>) -> Self {
        PVectorMut::I8(input)
    }

    fn from_i16(input: Self::Input<i16>) -> Self {
        PVectorMut::I16(input)
    }

    fn from_i32(input: Self::Input<i32>) -> Self {
        PVectorMut::I32(input)
    }

    fn from_i64(input: Self::Input<i64>) -> Self {
        PVectorMut::I64(input)
    }

    fn from_f16(input: Self::Input<f16>) -> Self {
        PVectorMut::F16(input)
    }

    fn from_f32(input: Self::Input<f32>) -> Self {
        PVectorMut::F32(input)
    }

    fn from_f64(input: Self::Input<f64>) -> Self {
        PVectorMut::F64(input)
    }
}
