// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Definition and implementation of [`DecimalVector`].

use std::fmt::Debug;
use std::ops::RangeBounds;

use vortex_dtype::DecimalType;
use vortex_dtype::DecimalTypeDowncast;
use vortex_dtype::DecimalTypeUpcast;
use vortex_dtype::NativeDecimalType;
use vortex_dtype::i256;
use vortex_error::vortex_panic;
use vortex_mask::Mask;

use crate::VectorOps;
use crate::decimal::DVector;
use crate::decimal::DecimalScalar;
use crate::decimal::DecimalVectorMut;
use crate::match_each_dvector;

/// An enum over all supported decimal mutable vector types.
#[derive(Clone, Debug)]
pub enum DecimalVector {
    /// A decimal vector with 8-bit integer representation.
    D8(DVector<i8>),
    /// A decimal vector with 16-bit integer representation.
    D16(DVector<i16>),
    /// A decimal vector with 32-bit integer representation.
    D32(DVector<i32>),
    /// A decimal vector with 64-bit integer representation.
    D64(DVector<i64>),
    /// A decimal vector with 128-bit integer representation.
    D128(DVector<i128>),
    /// A decimal vector with 256-bit integer representation.
    D256(DVector<i256>),
}

impl DecimalVector {
    /// Returns the precision of the decimal vector.
    pub fn precision(&self) -> u8 {
        match_each_dvector!(self, |v| { v.precision() })
    }

    /// Returns the scale of the decimal vector.
    pub fn scale(&self) -> i8 {
        match_each_dvector!(self, |v| { v.scale() })
    }

    /// Returns the physical [`DecimalType`] of the decimal vector.
    pub fn decimal_type(&self) -> DecimalType {
        match self {
            Self::D8(_) => DecimalType::I8,
            Self::D16(_) => DecimalType::I16,
            Self::D32(_) => DecimalType::I32,
            Self::D64(_) => DecimalType::I64,
            Self::D128(_) => DecimalType::I128,
            Self::D256(_) => DecimalType::I256,
        }
    }
}

impl VectorOps for DecimalVector {
    type Mutable = DecimalVectorMut;
    type Scalar = DecimalScalar;

    fn len(&self) -> usize {
        match_each_dvector!(self, |v| { v.len() })
    }

    fn validity(&self) -> &Mask {
        match_each_dvector!(self, |v| { v.validity() })
    }

    fn scalar_at(&self, index: usize) -> DecimalScalar {
        match_each_dvector!(self, |v| { v.scalar_at(index).into() })
    }

    fn slice(&self, range: impl RangeBounds<usize> + Clone + Debug) -> Self {
        match_each_dvector!(self, |v| { DecimalVector::from(v.slice(range)) })
    }

    fn clear(&mut self) {
        match_each_dvector!(self, |v| { v.clear() })
    }

    fn try_into_mut(self) -> Result<DecimalVectorMut, Self> {
        match_each_dvector!(self, |v| {
            v.try_into_mut()
                .map(DecimalVectorMut::from)
                .map_err(Self::from)
        })
    }

    fn into_mut(self) -> DecimalVectorMut {
        match_each_dvector!(self, |v| { DecimalVectorMut::from(v.into_mut()) })
    }
}

impl DecimalTypeDowncast for DecimalVector {
    type Output<T: NativeDecimalType> = DVector<T>;

    fn into_i8(self) -> Self::Output<i8> {
        if let Self::D8(vec) = self {
            return vec;
        }
        vortex_panic!("DecimalVector is not of type D8");
    }

    fn into_i16(self) -> Self::Output<i16> {
        if let Self::D16(vec) = self {
            return vec;
        }
        vortex_panic!("DecimalVector is not of type D16");
    }

    fn into_i32(self) -> Self::Output<i32> {
        if let Self::D32(vec) = self {
            return vec;
        }
        vortex_panic!("DecimalVector is not of type D32");
    }

    fn into_i64(self) -> Self::Output<i64> {
        if let Self::D64(vec) = self {
            return vec;
        }
        vortex_panic!("DecimalVector is not of type D64");
    }

    fn into_i128(self) -> Self::Output<i128> {
        if let Self::D128(vec) = self {
            return vec;
        }
        vortex_panic!("DecimalVector is not of type D128");
    }

    fn into_i256(self) -> Self::Output<i256> {
        if let Self::D256(vec) = self {
            return vec;
        }
        vortex_panic!("DecimalVector is not of type D256");
    }
}

impl DecimalTypeUpcast for DecimalVector {
    type Input<T: NativeDecimalType> = DVector<T>;

    fn from_i8(input: Self::Input<i8>) -> Self {
        Self::D8(input)
    }

    fn from_i16(input: Self::Input<i16>) -> Self {
        Self::D16(input)
    }

    fn from_i32(input: Self::Input<i32>) -> Self {
        Self::D32(input)
    }

    fn from_i64(input: Self::Input<i64>) -> Self {
        Self::D64(input)
    }

    fn from_i128(input: Self::Input<i128>) -> Self {
        Self::D128(input)
    }

    fn from_i256(input: Self::Input<i256>) -> Self {
        Self::D256(input)
    }
}
