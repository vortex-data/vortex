// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Definition and implementation of [`PrimitiveVector`].

use vortex_dtype::half::f16;
use vortex_dtype::{NativePType, PTypeUpcast};

use super::{PVector, PrimitiveVectorMut};
use crate::{VectorOps, match_each_pvector};

/// An immutable vector of primitive values.
///
/// The mutable equivalent of this type is [`PrimitiveVectorMut`].
///
/// `PrimitiveVector` is represented by an enum over all possible [`PVector`] types (which are
/// templated by the types that implement [`NativePType`]).
///
/// See the documentation for [`PVector`] for more information.
#[derive(Debug, Clone)]
pub enum PrimitiveVector {
    /// U8
    U8(PVector<u8>),
    /// U16
    U16(PVector<u16>),
    /// U32
    U32(PVector<u32>),
    /// U64
    U64(PVector<u64>),
    /// I8
    I8(PVector<i8>),
    /// I16
    I16(PVector<i16>),
    /// I32
    I32(PVector<i32>),
    /// I64
    I64(PVector<i64>),
    /// F16
    F16(PVector<f16>),
    /// F32
    F32(PVector<f32>),
    /// F64
    F64(PVector<f64>),
}

impl VectorOps for PrimitiveVector {
    type Mutable = PrimitiveVectorMut;

    fn len(&self) -> usize {
        match_each_pvector!(self, |v| { v.len() })
    }

    fn validity(&self) -> &vortex_mask::Mask {
        match_each_pvector!(self, |v| { v.validity() })
    }

    fn try_into_mut(self) -> Result<Self::Mutable, Self>
    where
        Self: Sized,
    {
        match_each_pvector!(self, |v| {
            v.try_into_mut()
                .map(PrimitiveVectorMut::from)
                .map_err(PrimitiveVector::from)
        })
    }
}

impl<T: NativePType> From<PVector<T>> for PrimitiveVector {
    fn from(v: PVector<T>) -> Self {
        T::upcast(v)
    }
}

impl PTypeUpcast for PrimitiveVector {
    type Input<T: NativePType> = PVector<T>;

    fn from_u8(input: Self::Input<u8>) -> Self {
        PrimitiveVector::U8(input)
    }

    fn from_u16(input: Self::Input<u16>) -> Self {
        PrimitiveVector::U16(input)
    }

    fn from_u32(input: Self::Input<u32>) -> Self {
        PrimitiveVector::U32(input)
    }

    fn from_u64(input: Self::Input<u64>) -> Self {
        PrimitiveVector::U64(input)
    }

    fn from_i8(input: Self::Input<i8>) -> Self {
        PrimitiveVector::I8(input)
    }

    fn from_i16(input: Self::Input<i16>) -> Self {
        PrimitiveVector::I16(input)
    }

    fn from_i32(input: Self::Input<i32>) -> Self {
        PrimitiveVector::I32(input)
    }

    fn from_i64(input: Self::Input<i64>) -> Self {
        PrimitiveVector::I64(input)
    }

    fn from_f16(input: Self::Input<f16>) -> Self {
        PrimitiveVector::F16(input)
    }

    fn from_f32(input: Self::Input<f32>) -> Self {
        PrimitiveVector::F32(input)
    }

    fn from_f64(input: Self::Input<f64>) -> Self {
        PrimitiveVector::F64(input)
    }
}
