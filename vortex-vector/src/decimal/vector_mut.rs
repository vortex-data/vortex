// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Definition and implementation of [`DecimalVectorMut`].

use vortex_dtype::DecimalDType;
use vortex_dtype::DecimalType;
use vortex_dtype::DecimalTypeDowncast;
use vortex_dtype::DecimalTypeUpcast;
use vortex_dtype::NativeDecimalType;
use vortex_dtype::PrecisionScale;
use vortex_dtype::i256;
use vortex_dtype::match_each_decimal_value_type;
use vortex_error::vortex_panic;
use vortex_mask::MaskMut;

use crate::VectorMutOps;
use crate::decimal::DVectorMut;
use crate::decimal::DecimalScalar;
use crate::decimal::DecimalVector;
use crate::match_each_dvector_mut;

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
    /// Returns the [`DecimalType`] of the decimal vector.
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

    /// Create a new mutable decimal vector with the given primitive type and capacity.
    pub fn with_capacity(decimal_dtype: &DecimalDType, capacity: usize) -> Self {
        let decimal_type = DecimalType::smallest_decimal_value_type(decimal_dtype);
        match_each_decimal_value_type!(decimal_type, |D| {
            let ps = PrecisionScale::<D>::new(decimal_dtype.precision(), decimal_dtype.scale());
            DVectorMut::<D>::with_capacity(ps, capacity).into()
        })
    }
}

impl VectorMutOps for DecimalVectorMut {
    type Immutable = DecimalVector;

    fn len(&self) -> usize {
        match_each_dvector_mut!(self, |d| { d.len() })
    }

    fn validity(&self) -> &MaskMut {
        match_each_dvector_mut!(self, |d| { d.validity() })
    }

    fn capacity(&self) -> usize {
        match_each_dvector_mut!(self, |d| { d.capacity() })
    }

    fn reserve(&mut self, additional: usize) {
        match_each_dvector_mut!(self, |d| { d.reserve(additional) })
    }

    fn clear(&mut self) {
        match_each_dvector_mut!(self, |d| { d.clear() })
    }

    fn truncate(&mut self, len: usize) {
        match_each_dvector_mut!(self, |d| { d.truncate(len) })
    }

    fn extend_from_vector(&mut self, other: &DecimalVector) {
        match (self, other) {
            (Self::D8(s), DecimalVector::D8(o)) => s.extend_from_vector(o),
            (Self::D16(s), DecimalVector::D16(o)) => s.extend_from_vector(o),
            (Self::D32(s), DecimalVector::D32(o)) => s.extend_from_vector(o),
            (Self::D64(s), DecimalVector::D64(o)) => s.extend_from_vector(o),
            (Self::D128(s), DecimalVector::D128(o)) => s.extend_from_vector(o),
            (Self::D256(s), DecimalVector::D256(o)) => s.extend_from_vector(o),
            _ => vortex_panic!("Mismatched decimal vector types in extend_from_vector"),
        }
    }

    fn append_nulls(&mut self, n: usize) {
        match_each_dvector_mut!(self, |d| { d.append_nulls(n) })
    }

    fn append_zeros(&mut self, n: usize) {
        match_each_dvector_mut!(self, |d| { d.append_zeros(n) })
    }

    #[allow(clippy::many_single_char_names)]
    fn append_scalars(&mut self, scalar: &DecimalScalar, n: usize) {
        match (self, scalar) {
            (Self::D8(s), DecimalScalar::D8(o)) => s.append_scalars(o, n),
            (Self::D16(s), DecimalScalar::D16(o)) => s.append_scalars(o, n),
            (Self::D32(s), DecimalScalar::D32(o)) => s.append_scalars(o, n),
            (Self::D64(s), DecimalScalar::D64(o)) => s.append_scalars(o, n),
            (Self::D128(s), DecimalScalar::D128(o)) => s.append_scalars(o, n),
            (Self::D256(s), DecimalScalar::D256(o)) => s.append_scalars(o, n),
            _ => vortex_panic!("Mismatched decimal vector and scalar types in append_scalar"),
        }
    }

    fn freeze(self) -> DecimalVector {
        match_each_dvector_mut!(self, |d| { d.freeze().into() })
    }

    fn split_off(&mut self, at: usize) -> Self {
        match_each_dvector_mut!(self, |d| { d.split_off(at).into() })
    }

    fn unsplit(&mut self, other: Self) {
        match (self, other) {
            (Self::D8(s), Self::D8(o)) => s.unsplit(o),
            (Self::D16(s), Self::D16(o)) => s.unsplit(o),
            (Self::D32(s), Self::D32(o)) => s.unsplit(o),
            (Self::D64(s), Self::D64(o)) => s.unsplit(o),
            (Self::D128(s), Self::D128(o)) => s.unsplit(o),
            (Self::D256(s), Self::D256(o)) => s.unsplit(o),
            _ => vortex_panic!("Mismatched decimal vector types in unsplit"),
        }
    }
}

impl DecimalTypeDowncast for DecimalVectorMut {
    type Output<T: NativeDecimalType> = DVectorMut<T>;

    fn into_i8(self) -> Self::Output<i8> {
        if let Self::D8(vec) = self {
            return vec;
        }
        vortex_panic!("DecimalVectorMut is not of type D8");
    }

    fn into_i16(self) -> Self::Output<i16> {
        if let Self::D16(vec) = self {
            return vec;
        }
        vortex_panic!("DecimalVectorMut is not of type D16");
    }

    fn into_i32(self) -> Self::Output<i32> {
        if let Self::D32(vec) = self {
            return vec;
        }
        vortex_panic!("DecimalVectorMut is not of type D32");
    }

    fn into_i64(self) -> Self::Output<i64> {
        if let Self::D64(vec) = self {
            return vec;
        }
        vortex_panic!("DecimalVectorMut is not of type D64");
    }

    fn into_i128(self) -> Self::Output<i128> {
        if let Self::D128(vec) = self {
            return vec;
        }
        vortex_panic!("DecimalVectorMut is not of type D128");
    }

    fn into_i256(self) -> Self::Output<i256> {
        if let Self::D256(vec) = self {
            return vec;
        }
        vortex_panic!("DecimalVectorMut is not of type D256");
    }
}

impl DecimalTypeUpcast for DecimalVectorMut {
    type Input<T: NativeDecimalType> = DVectorMut<T>;

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
