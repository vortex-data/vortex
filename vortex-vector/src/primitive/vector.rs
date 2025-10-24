// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Definition and implementation of [`PrimitiveVector`].

use vortex_dtype::half::f16;
use vortex_dtype::{NativePType, PTypeDowncast, PTypeUpcast};
use vortex_error::vortex_panic;

use super::macros::match_each_pvec;
use crate::{PVec, PrimitiveVectorMut, VectorOps};

/// An immutable vector of primitive values.
///
/// The mutable equivalent of this type is [`PrimitiveVectorMut`].
///
/// `PrimitiveVector` is represented by an enum over all possible [`PVec`] types (which are
/// templated by the types that implement [`NativePType`]).
///
/// See the documentation for [`PVec`] for more information.
#[derive(Debug, Clone)]
pub enum PrimitiveVector {
    /// U8
    U8(PVec<u8>),
    /// U16
    U16(PVec<u16>),
    /// U32
    U32(PVec<u32>),
    /// U64
    U64(PVec<u64>),
    /// I8
    I8(PVec<i8>),
    /// I16
    I16(PVec<i16>),
    /// I32
    I32(PVec<i32>),
    /// I64
    I64(PVec<i64>),
    /// F16
    F16(PVec<f16>),
    /// F32
    F32(PVec<f32>),
    /// F64
    F64(PVec<f64>),
}

impl VectorOps for PrimitiveVector {
    type Mutable = PrimitiveVectorMut;

    fn len(&self) -> usize {
        match_each_pvec!(self, |v| { v.len() })
    }

    fn validity(&self) -> &vortex_mask::Mask {
        match_each_pvec!(self, |v| { v.validity() })
    }

    fn try_into_mut(self) -> Result<Self::Mutable, Self>
    where
        Self: Sized,
    {
        match_each_pvec!(self, |v| {
            v.try_into_mut()
                .map(PrimitiveVectorMut::from)
                .map_err(PrimitiveVector::from)
        })
    }
}

impl<T: NativePType> From<PVec<T>> for PrimitiveVector {
    fn from(v: PVec<T>) -> Self {
        T::upcast(v)
    }
}

impl PTypeUpcast for PrimitiveVector {
    type Input<T: NativePType> = PVec<T>;

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

impl PTypeDowncast for PrimitiveVector {
    type Output<T: NativePType> = PVec<T>;

    fn into_u8(self) -> Self::Output<u8> {
        if let PrimitiveVector::U8(v) = self {
            return v;
        }
        vortex_panic!("Expected PrimitiveVector::U8, got {self:?}");
    }

    fn into_u16(self) -> Self::Output<u16> {
        if let PrimitiveVector::U16(v) = self {
            return v;
        }
        vortex_panic!("Expected PrimitiveVector::U16, got {self:?}");
    }

    fn into_u32(self) -> Self::Output<u32> {
        if let PrimitiveVector::U32(v) = self {
            return v;
        }
        vortex_panic!("Expected PrimitiveVector::U32, got {self:?}");
    }

    fn into_u64(self) -> Self::Output<u64> {
        if let PrimitiveVector::U64(v) = self {
            return v;
        }
        vortex_panic!("Expected PrimitiveVector::U64, got {self:?}");
    }

    fn into_i8(self) -> Self::Output<i8> {
        if let PrimitiveVector::I8(v) = self {
            return v;
        }
        vortex_panic!("Expected PrimitiveVector::I8, got {self:?}");
    }

    fn into_i16(self) -> Self::Output<i16> {
        if let PrimitiveVector::I16(v) = self {
            return v;
        }
        vortex_panic!("Expected PrimitiveVector::I16, got {self:?}");
    }

    fn into_i32(self) -> Self::Output<i32> {
        if let PrimitiveVector::I32(v) = self {
            return v;
        }
        vortex_panic!("Expected PrimitiveVector::I32, got {self:?}");
    }

    fn into_i64(self) -> Self::Output<i64> {
        if let PrimitiveVector::I64(v) = self {
            return v;
        }
        vortex_panic!("Expected PrimitiveVector::I64, got {self:?}");
    }

    fn into_f16(self) -> Self::Output<f16> {
        if let PrimitiveVector::F16(v) = self {
            return v;
        }
        vortex_panic!("Expected PrimitiveVector::F16, got {self:?}");
    }

    fn into_f32(self) -> Self::Output<f32> {
        if let PrimitiveVector::F32(v) = self {
            return v;
        }
        vortex_panic!("Expected PrimitiveVector::F32, got {self:?}");
    }

    fn into_f64(self) -> Self::Output<f64> {
        if let PrimitiveVector::F64(v) = self {
            return v;
        }
        vortex_panic!("Expected PrimitiveVector::F64, got {self:?}");
    }
}
