// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::{DecimalTypeUpcast, NativeDecimalType, PrecisionScale, i256};
use vortex_error::VortexExpect;

use crate::decimal::DVectorMut;
use crate::{Scalar, ScalarOps, VectorMut, VectorMutOps};

/// Represents a decimal scalar value.
#[derive(Debug)]
pub enum DecimalScalar {
    /// 8-bit decimal scalar.
    D8(DScalar<i8>),
    /// 16-bit decimal scalar.
    D16(DScalar<i16>),
    /// 32-bit decimal scalar.
    D32(DScalar<i32>),
    /// 64-bit decimal scalar.
    D64(DScalar<i64>),
    /// 128-bit decimal scalar.
    D128(DScalar<i128>),
    /// 256-bit decimal scalar.
    D256(DScalar<i256>),
}

impl ScalarOps for DecimalScalar {
    fn is_valid(&self) -> bool {
        match self {
            DecimalScalar::D8(v) => v.is_valid(),
            DecimalScalar::D16(v) => v.is_valid(),
            DecimalScalar::D32(v) => v.is_valid(),
            DecimalScalar::D64(v) => v.is_valid(),
            DecimalScalar::D128(v) => v.is_valid(),
            DecimalScalar::D256(v) => v.is_valid(),
        }
    }

    fn repeat(&self, n: usize) -> VectorMut {
        match self {
            DecimalScalar::D8(v) => v.repeat(n),
            DecimalScalar::D16(v) => v.repeat(n),
            DecimalScalar::D32(v) => v.repeat(n),
            DecimalScalar::D64(v) => v.repeat(n),
            DecimalScalar::D128(v) => v.repeat(n),
            DecimalScalar::D256(v) => v.repeat(n),
        }
    }
}

impl From<DecimalScalar> for Scalar {
    fn from(val: DecimalScalar) -> Self {
        Scalar::Decimal(val)
    }
}

/// Represents a decimal scalar value with a specific native decimal type.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DScalar<D> {
    ps: PrecisionScale<D>,
    value: Option<D>,
}

impl<D: NativeDecimalType> DScalar<D> {
    /// Creates a new decimal scalar with the given precision/scale and value.
    ///
    /// Returns `None` if the value is not valid for the given precision/scale.
    pub fn maybe_new(ps: PrecisionScale<D>, value: Option<D>) -> Option<Self> {
        Some(match value {
            None => Self { ps, value: None },
            Some(v) => {
                if !ps.is_valid(v) {
                    return None;
                }
                Self { ps, value: Some(v) }
            }
        })
    }

    /// Creates a new decimal scalar with the given precision/scale and value without validation.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the value is valid for the given precision/scale.
    pub unsafe fn new_unchecked(ps: PrecisionScale<D>, value: Option<D>) -> Self {
        Self { ps, value }
    }

    /// Returns the value of the decimal scalar, or `None` if the scalar is null.
    pub fn value(&self) -> Option<D> {
        self.value
    }
}

impl<D: NativeDecimalType> ScalarOps for DScalar<D> {
    fn is_valid(&self) -> bool {
        self.value.is_some()
    }

    fn repeat(&self, n: usize) -> VectorMut {
        let mut vec = DVectorMut::with_capacity(self.ps, n);
        match &self.value {
            None => vec.append_nulls(n),
            Some(v) => vec.try_append_n(*v, n).vortex_expect("known to fit"),
        }
        vec.into()
    }
}

impl<D: NativeDecimalType> From<DScalar<D>> for Scalar {
    fn from(value: DScalar<D>) -> Self {
        Scalar::Decimal(D::upcast(value))
    }
}

impl<D: NativeDecimalType> From<DScalar<D>> for DecimalScalar {
    fn from(value: DScalar<D>) -> Self {
        D::upcast(value)
    }
}

impl DecimalTypeUpcast for DecimalScalar {
    type Input<T: NativeDecimalType> = DScalar<T>;

    fn from_i8(input: Self::Input<i8>) -> Self {
        DecimalScalar::D8(input)
    }

    fn from_i16(input: Self::Input<i16>) -> Self {
        DecimalScalar::D16(input)
    }

    fn from_i32(input: Self::Input<i32>) -> Self {
        DecimalScalar::D32(input)
    }

    fn from_i64(input: Self::Input<i64>) -> Self {
        DecimalScalar::D64(input)
    }

    fn from_i128(input: Self::Input<i128>) -> Self {
        DecimalScalar::D128(input)
    }

    fn from_i256(input: Self::Input<i256>) -> Self {
        DecimalScalar::D256(input)
    }
}
