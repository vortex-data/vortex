// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::half::f16;
use vortex_dtype::{NativePType, PTypeUpcast};

use crate::{Scalar, ScalarOps, VectorMut};

/// Represents a primitive scalar value.
pub enum PrimitiveScalar {
    /// 8-bit signed integer scalar
    I8(Option<i8>),
    /// 16-bit signed integer scalar
    I16(Option<i16>),
    /// 32-bit signed integer scalar
    I32(Option<i32>),
    /// 64-bit signed integer scalar
    I64(Option<i64>),
    /// 8-bit unsigned integer scalar
    U8(Option<u8>),
    /// 16-bit unsigned integer scalar
    U16(Option<u16>),
    /// 32-bit unsigned integer scalar
    U32(Option<u32>),
    /// 64-bit unsigned integer scalar
    U64(Option<u64>),
    /// 16-bit floating point scalar
    F16(Option<f16>),
    /// 32-bit floating point scalar
    F32(Option<f32>),
    /// 64-bit floating point scalar
    F64(Option<f64>),
}

impl ScalarOps for PrimitiveScalar {
    fn is_valid(&self) -> bool {
        match self {
            PrimitiveScalar::I8(v) => v.is_some(),
            PrimitiveScalar::I16(v) => v.is_some(),
            PrimitiveScalar::I32(v) => v.is_some(),
            PrimitiveScalar::I64(v) => v.is_some(),
            PrimitiveScalar::U8(v) => v.is_some(),
            PrimitiveScalar::U16(v) => v.is_some(),
            PrimitiveScalar::U32(v) => v.is_some(),
            PrimitiveScalar::U64(v) => v.is_some(),
            PrimitiveScalar::F16(v) => v.is_some(),
            PrimitiveScalar::F32(v) => v.is_some(),
            PrimitiveScalar::F64(v) => v.is_some(),
        }
    }

    fn repeat(&self, _n: usize) -> VectorMut {
        todo!()
    }
}

impl Into<Scalar> for PrimitiveScalar {
    fn into(self) -> Scalar {
        Scalar::Primitive(self)
    }
}

impl<T: NativePType> From<Option<T>> for PrimitiveScalar {
    fn from(value: Option<T>) -> Self {
        T::upcast(value)
    }
}

impl PTypeUpcast for PrimitiveScalar {
    type Input<T: NativePType> = Option<T>;

    fn from_u8(input: Self::Input<u8>) -> Self {
        PrimitiveScalar::U8(input)
    }

    fn from_u16(input: Self::Input<u16>) -> Self {
        PrimitiveScalar::U16(input)
    }

    fn from_u32(input: Self::Input<u32>) -> Self {
        PrimitiveScalar::U32(input)
    }

    fn from_u64(input: Self::Input<u64>) -> Self {
        PrimitiveScalar::U64(input)
    }

    fn from_i8(input: Self::Input<i8>) -> Self {
        PrimitiveScalar::I8(input)
    }

    fn from_i16(input: Self::Input<i16>) -> Self {
        PrimitiveScalar::I16(input)
    }

    fn from_i32(input: Self::Input<i32>) -> Self {
        PrimitiveScalar::I32(input)
    }

    fn from_i64(input: Self::Input<i64>) -> Self {
        PrimitiveScalar::I64(input)
    }

    fn from_f16(input: Self::Input<f16>) -> Self {
        PrimitiveScalar::F16(input)
    }

    fn from_f32(input: Self::Input<f32>) -> Self {
        PrimitiveScalar::F32(input)
    }

    fn from_f64(input: Self::Input<f64>) -> Self {
        PrimitiveScalar::F64(input)
    }
}
