// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Definition and implementation of [`PrimitiveVector`].

use std::fmt::Debug;
use std::ops::RangeBounds;

use vortex_dtype::NativePType;
use vortex_dtype::PType;
use vortex_dtype::PTypeDowncast;
use vortex_dtype::PTypeUpcast;
use vortex_dtype::half::f16;
use vortex_error::vortex_panic;
use vortex_mask::Mask;

use crate::VectorOps;
use crate::match_each_pvector;
use crate::primitive::PVector;
use crate::primitive::PrimitiveScalar;
use crate::primitive::PrimitiveVectorMut;

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

impl PrimitiveVector {
    /// Returns the [`PType`] of this [`PrimitiveVector`].
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
}

impl VectorOps for PrimitiveVector {
    type Mutable = PrimitiveVectorMut;
    type Scalar = PrimitiveScalar;

    fn len(&self) -> usize {
        match_each_pvector!(self, |v| { v.len() })
    }

    fn validity(&self) -> &Mask {
        match_each_pvector!(self, |v| { v.validity() })
    }

    fn scalar_at(&self, index: usize) -> PrimitiveScalar {
        match_each_pvector!(self, |v| { v.scalar_at(index).into() })
    }

    fn slice(&self, range: impl RangeBounds<usize> + Clone + Debug) -> Self {
        match_each_pvector!(self, |v| { v.slice(range).into() })
    }

    fn clear(&mut self) {
        match_each_pvector!(self, |v| { v.clear() })
    }

    fn try_into_mut(self) -> Result<PrimitiveVectorMut, Self> {
        match_each_pvector!(self, |v| {
            v.try_into_mut()
                .map(PrimitiveVectorMut::from)
                .map_err(Self::from)
        })
    }

    fn into_mut(self) -> PrimitiveVectorMut {
        match_each_pvector!(self, |v| { v.into_mut().into() })
    }
}

impl PTypeUpcast for PrimitiveVector {
    type Input<T: NativePType> = PVector<T>;

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

impl PTypeDowncast for PrimitiveVector {
    type Output<T: NativePType> = PVector<T>;

    fn into_u8(self) -> Self::Output<u8> {
        if let Self::U8(v) = self {
            return v;
        }
        vortex_panic!("Expected PrimitiveVector::U8, got {self:?}");
    }

    fn into_u16(self) -> Self::Output<u16> {
        if let Self::U16(v) = self {
            return v;
        }
        vortex_panic!("Expected PrimitiveVector::U16, got {self:?}");
    }

    fn into_u32(self) -> Self::Output<u32> {
        if let Self::U32(v) = self {
            return v;
        }
        vortex_panic!("Expected PrimitiveVector::U32, got {self:?}");
    }

    fn into_u64(self) -> Self::Output<u64> {
        if let Self::U64(v) = self {
            return v;
        }
        vortex_panic!("Expected PrimitiveVector::U64, got {self:?}");
    }

    fn into_i8(self) -> Self::Output<i8> {
        if let Self::I8(v) = self {
            return v;
        }
        vortex_panic!("Expected PrimitiveVector::I8, got {self:?}");
    }

    fn into_i16(self) -> Self::Output<i16> {
        if let Self::I16(v) = self {
            return v;
        }
        vortex_panic!("Expected PrimitiveVector::I16, got {self:?}");
    }

    fn into_i32(self) -> Self::Output<i32> {
        if let Self::I32(v) = self {
            return v;
        }
        vortex_panic!("Expected PrimitiveVector::I32, got {self:?}");
    }

    fn into_i64(self) -> Self::Output<i64> {
        if let Self::I64(v) = self {
            return v;
        }
        vortex_panic!("Expected PrimitiveVector::I64, got {self:?}");
    }

    fn into_f16(self) -> Self::Output<f16> {
        if let Self::F16(v) = self {
            return v;
        }
        vortex_panic!("Expected PrimitiveVector::F16, got {self:?}");
    }

    fn into_f32(self) -> Self::Output<f32> {
        if let Self::F32(v) = self {
            return v;
        }
        vortex_panic!("Expected PrimitiveVector::F32, got {self:?}");
    }

    fn into_f64(self) -> Self::Output<f64> {
        if let Self::F64(v) = self {
            return v;
        }
        vortex_panic!("Expected PrimitiveVector::F64, got {self:?}");
    }
}

impl<'a> PTypeDowncast for &'a PrimitiveVector {
    type Output<T: NativePType> = &'a PVector<T>;

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

impl<'a> PTypeDowncast for &'a mut PrimitiveVector {
    type Output<T: NativePType> = &'a mut PVector<T>;

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
