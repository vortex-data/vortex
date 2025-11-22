// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Deref;

use vortex_dtype::half::f16;
use vortex_dtype::{NativePType, PTypeUpcast};
use vortex_error::VortexExpect;

use crate::primitive::{PVectorMut, PrimitiveVectorMut};
use crate::{Scalar, ScalarOps, VectorMut, VectorMutOps};

/// Represents a primitive scalar value.
#[derive(Debug)]
pub enum PrimitiveScalar {
    /// 8-bit signed integer scalar
    I8(PScalar<i8>),
    /// 16-bit signed integer scalar
    I16(PScalar<i16>),
    /// 32-bit signed integer scalar
    I32(PScalar<i32>),
    /// 64-bit signed integer scalar
    I64(PScalar<i64>),
    /// 8-bit unsigned integer scalar
    U8(PScalar<u8>),
    /// 16-bit unsigned integer scalar
    U16(PScalar<u16>),
    /// 32-bit unsigned integer scalar
    U32(PScalar<u32>),
    /// 64-bit unsigned integer scalar
    U64(PScalar<u64>),
    /// 16-bit floating point scalar
    F16(PScalar<f16>),
    /// 32-bit floating point scalar
    F32(PScalar<f32>),
    /// 64-bit floating point scalar
    F64(PScalar<f64>),
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

impl From<PrimitiveScalar> for Scalar {
    fn from(val: PrimitiveScalar) -> Self {
        Scalar::Primitive(val)
    }
}

/// Represents a primitive scalar value with a specific native primitive type.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PScalar<T>(Option<T>);

impl<T: NativePType> PScalar<T> {
    /// Creates a new primitive scalar with the given value.
    pub fn new(value: Option<T>) -> Self {
        Self(value)
    }

    /// Returns the value of the primitive scalar, or `None` if the scalar is null.
    pub fn value(&self) -> Option<T> {
        self.0
    }
}

impl<T: NativePType> From<PScalar<T>> for PrimitiveScalar {
    fn from(value: PScalar<T>) -> Self {
        T::upcast(value)
    }
}

impl<T: NativePType> ScalarOps for PScalar<T> {
    fn is_valid(&self) -> bool {
        self.0.is_some()
    }

    fn repeat(&self, n: usize) -> VectorMut {
        let mut vec = PVectorMut::<T>::with_capacity(n);
        match self.0 {
            None => vec.append_nulls(n),
            Some(v) => vec.append_values(v, n),
        }
        PrimitiveVectorMut::from(vec).into()
    }
}

impl<T: NativePType> From<PScalar<T>> for Scalar {
    fn from(value: PScalar<T>) -> Self {
        Scalar::Primitive(T::upcast(value))
    }
}

impl PTypeUpcast for PrimitiveScalar {
    type Input<T: NativePType> = PScalar<T>;

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

impl<T: NativePType> Deref for PScalar<T> {
    type Target = Option<T>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl PrimitiveScalar {
    /// Returns the scalar value as `usize` if possible.
    ///
    /// Returns `None` if the scalar cannot be cast to a usize.
    ///
    /// # Panics
    ///
    /// If the scalar is null.
    pub fn to_usize(&self) -> Option<usize> {
        match self {
            PrimitiveScalar::I8(v) => usize::try_from(v.vortex_expect("null scalar")).ok(),
            PrimitiveScalar::I16(v) => usize::try_from(v.vortex_expect("null scalar")).ok(),
            PrimitiveScalar::I32(v) => usize::try_from(v.vortex_expect("null scalar")).ok(),
            PrimitiveScalar::I64(v) => usize::try_from(v.vortex_expect("null scalar")).ok(),
            PrimitiveScalar::U8(v) => Some(v.vortex_expect("null scalar") as usize),
            PrimitiveScalar::U16(v) => Some(v.vortex_expect("null scalar") as usize),
            PrimitiveScalar::U32(v) => Some(v.vortex_expect("null scalar") as usize),
            PrimitiveScalar::U64(v) => usize::try_from(v.vortex_expect("null scalar")).ok(),
            PrimitiveScalar::F16(_) => None,
            PrimitiveScalar::F32(_) => None,
            PrimitiveScalar::F64(_) => None,
        }
    }
}
