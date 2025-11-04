// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Definition and implementation of [`DecimalVectorMut`].

use vortex_dtype::{
    DecimalDType, DecimalType, DecimalTypeDowncast, DecimalTypeUpcast, NativeDecimalType, i256,
};
use vortex_error::vortex_panic;

use crate::decimal::DVectorMut;
use crate::{DecimalVector, VectorMutOps, match_each_dvector_mut};

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

impl DecimalVectorMut {
    /// Create a new mutable decimal vector with the given primitive type and capacity.
    pub fn with_capacity(decimal_dtype: &DecimalDType, capacity: usize) -> Self {
        let decimal_kind = DecimalType::smallest_decimal_value_type(decimal_dtype);

        match decimal_kind {
            DecimalType::I8 => {
                DecimalVectorMut::D8(DVectorMut::<i8>::with_capacity(decimal_dtype, capacity))
            }
            DecimalType::I16 => {
                DecimalVectorMut::D16(DVectorMut::<i16>::with_capacity(decimal_dtype, capacity))
            }
            DecimalType::I32 => {
                DecimalVectorMut::D32(DVectorMut::<i32>::with_capacity(decimal_dtype, capacity))
            }
            DecimalType::I64 => {
                DecimalVectorMut::D64(DVectorMut::<i64>::with_capacity(decimal_dtype, capacity))
            }
            DecimalType::I128 => {
                DecimalVectorMut::D128(DVectorMut::<i128>::with_capacity(decimal_dtype, capacity))
            }
            DecimalType::I256 => {
                DecimalVectorMut::D256(DVectorMut::<i256>::with_capacity(decimal_dtype, capacity))
            }
        }
    }
}

impl VectorMutOps for DecimalVectorMut {
    type Immutable = DecimalVector;

    fn len(&self) -> usize {
        match_each_dvector_mut!(self, |d| { d.len() })
    }

    fn capacity(&self) -> usize {
        match_each_dvector_mut!(self, |d| { d.capacity() })
    }

    fn reserve(&mut self, additional: usize) {
        match_each_dvector_mut!(self, |d| { d.reserve(additional) })
    }

    fn extend_from_vector(&mut self, other: &DecimalVector) {
        match (self, other) {
            (DecimalVectorMut::D8(s), DecimalVector::D8(o)) => s.extend_from_vector(o),
            (DecimalVectorMut::D16(s), DecimalVector::D16(o)) => s.extend_from_vector(o),
            (DecimalVectorMut::D32(s), DecimalVector::D32(o)) => s.extend_from_vector(o),
            (DecimalVectorMut::D64(s), DecimalVector::D64(o)) => s.extend_from_vector(o),
            (DecimalVectorMut::D128(s), DecimalVector::D128(o)) => s.extend_from_vector(o),
            (DecimalVectorMut::D256(s), DecimalVector::D256(o)) => s.extend_from_vector(o),
            _ => vortex_panic!("Mismatched decimal vector types in extend_from_vector"),
        }
    }

    fn append_nulls(&mut self, n: usize) {
        match_each_dvector_mut!(self, |d| { d.append_nulls(n) })
    }

    fn freeze(self) -> DecimalVector {
        match_each_dvector_mut!(self, |d| { d.freeze().into() })
    }

    fn split_off(&mut self, at: usize) -> Self {
        match_each_dvector_mut!(self, |d| { d.split_off(at).into() })
    }

    fn unsplit(&mut self, other: Self) {
        match (self, other) {
            (DecimalVectorMut::D8(s), DecimalVectorMut::D8(o)) => s.unsplit(o),
            (DecimalVectorMut::D16(s), DecimalVectorMut::D16(o)) => s.unsplit(o),
            (DecimalVectorMut::D32(s), DecimalVectorMut::D32(o)) => s.unsplit(o),
            (DecimalVectorMut::D64(s), DecimalVectorMut::D64(o)) => s.unsplit(o),
            (DecimalVectorMut::D128(s), DecimalVectorMut::D128(o)) => s.unsplit(o),
            (DecimalVectorMut::D256(s), DecimalVectorMut::D256(o)) => s.unsplit(o),
            _ => vortex_panic!("Mismatched decimal vector types in unsplit"),
        }
    }
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
