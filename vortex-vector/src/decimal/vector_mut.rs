// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::decimal::DVectorMut;
use vortex_dtype::{i256, DecimalTypeDowncast, DecimalTypeUpcast, NativeDecimalType};
use vortex_error::vortex_panic;

/// An enum over all supported decimal mutable vector types.
#[derive(Clone, Debug)]
pub enum DecimalVectorMut {
    /// A decimal vector with 8-bit integer representation.
    D8(DVectorMut<i8>),
    /// A decimal vector with 16-bit integer representation.
    D16(DVectorMut<i16>),
    /// A decimal vector with 32-bit integer representation.
    D32(DVectorMut<i32>),
    /// A decimal vector with 64-bit integer representation.
    D64(DVectorMut<i64>),
    /// A decimal vector with 128-bit integer representation.
    D128(DVectorMut<i128>),
    /// A decimal vector with 256-bit integer representation.
    D256(DVectorMut<i256>),
}

impl DecimalTypeDowncast for DecimalVectorMut {
    type Output<T: NativeDecimalType> = DVectorMut<T>;

    fn into_i8(self) -> Self::Output<i8> {
        if let DecimalVectorMut::D8(vec) = self {
            return vec;
        }
        vortex_panic!("DecimalVectorMut is not of type D8");
    }

    fn into_i16(self) -> Self::Output<i16> {
        if let DecimalVectorMut::D16(vec) = self {
            return vec;
        }
        vortex_panic!("DecimalVectorMut is not of type D16");
    }

    fn into_i32(self) -> Self::Output<i32> {
        if let DecimalVectorMut::D32(vec) = self {
            return vec;
        }
        vortex_panic!("DecimalVectorMut is not of type D32");
    }

    fn into_i64(self) -> Self::Output<i64> {
        if let DecimalVectorMut::D64(vec) = self {
            return vec;
        }
        vortex_panic!("DecimalVectorMut is not of type D64");
    }

    fn into_i128(self) -> Self::Output<i128> {
        if let DecimalVectorMut::D128(vec) = self {
            return vec;
        }
        vortex_panic!("DecimalVectorMut is not of type D128");
    }

    fn into_i256(self) -> Self::Output<i256> {
        if let DecimalVectorMut::D256(vec) = self {
            return vec;
        }
        vortex_panic!("DecimalVectorMut is not of type D256");
    }
}

impl DecimalTypeUpcast for DecimalVectorMut {
    type Input<T: NativeDecimalType> = DVectorMut<T>;

    fn from_i8(input: Self::Input<i8>) -> Self {
        DecimalVectorMut::D8(input)
    }

    fn from_i16(input: Self::Input<i16>) -> Self {
        DecimalVectorMut::D16(input)
    }

    fn from_i32(input: Self::Input<i32>) -> Self {
        DecimalVectorMut::D32(input)
    }

    fn from_i64(input: Self::Input<i64>) -> Self {
        DecimalVectorMut::D64(input)
    }

    fn from_i128(input: Self::Input<i128>) -> Self {
        DecimalVectorMut::D128(input)
    }

    fn from_i256(input: Self::Input<i256>) -> Self {
        DecimalVectorMut::D256(input)
    }
}
