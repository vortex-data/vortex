// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::{DecimalTypeDowncast, DecimalTypeUpcast, NativeDecimalType, i256};
use vortex_error::vortex_panic;
use vortex_mask::Mask;

use crate::decimal::DVector;
use crate::{DecimalVectorMut, VectorOps, match_each_dvector};

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

impl VectorOps for DecimalVector {
    type Mutable = DecimalVectorMut;

    fn len(&self) -> usize {
        match_each_dvector!(self, |v| { v.len() })
    }

    fn validity(&self) -> &Mask {
        match_each_dvector!(self, |v| { v.validity() })
    }

    fn try_into_mut(self) -> Result<DecimalVectorMut, Self>
    where
        Self: Sized,
    {
        match_each_dvector!(self, |v| {
            v.try_into_mut()
                .map(DecimalVectorMut::from)
                .map_err(DecimalVector::from)
        })
    }
}

impl DecimalTypeDowncast for DecimalVector {
    type Output<T: NativeDecimalType> = DVector<T>;

    fn into_i8(self) -> Self::Output<i8> {
        if let DecimalVector::D8(vec) = self {
            return vec;
        }
        vortex_panic!("DecimalVector is not of type D8");
    }

    fn into_i16(self) -> Self::Output<i16> {
        if let DecimalVector::D16(vec) = self {
            return vec;
        }
        vortex_panic!("DecimalVector is not of type D16");
    }

    fn into_i32(self) -> Self::Output<i32> {
        if let DecimalVector::D32(vec) = self {
            return vec;
        }
        vortex_panic!("DecimalVector is not of type D32");
    }

    fn into_i64(self) -> Self::Output<i64> {
        if let DecimalVector::D64(vec) = self {
            return vec;
        }
        vortex_panic!("DecimalVector is not of type D64");
    }

    fn into_i128(self) -> Self::Output<i128> {
        if let DecimalVector::D128(vec) = self {
            return vec;
        }
        vortex_panic!("DecimalVector is not of type D128");
    }

    fn into_i256(self) -> Self::Output<i256> {
        if let DecimalVector::D256(vec) = self {
            return vec;
        }
        vortex_panic!("DecimalVector is not of type D256");
    }
}

impl DecimalTypeUpcast for DecimalVector {
    type Input<T: NativeDecimalType> = DVector<T>;

    fn from_i8(input: Self::Input<i8>) -> Self {
        DecimalVector::D8(input)
    }

    fn from_i16(input: Self::Input<i16>) -> Self {
        DecimalVector::D16(input)
    }

    fn from_i32(input: Self::Input<i32>) -> Self {
        DecimalVector::D32(input)
    }

    fn from_i64(input: Self::Input<i64>) -> Self {
        DecimalVector::D64(input)
    }

    fn from_i128(input: Self::Input<i128>) -> Self {
        DecimalVector::D128(input)
    }

    fn from_i256(input: Self::Input<i256>) -> Self {
        DecimalVector::D256(input)
    }
}
