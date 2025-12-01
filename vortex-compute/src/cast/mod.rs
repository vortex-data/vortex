// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Lossless casting of Vortex vectors for different logical data types.

mod bool;
mod null;
mod pvector;

use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_vector::Datum;
use vortex_vector::Scalar;
use vortex_vector::Vector;

/// Trait for casting vectors to different data types.
pub trait Cast {
    /// The output type after casting.
    type Output;

    /// Cast the vector to the specified data type.
    fn cast(&self, dtype: &DType) -> VortexResult<Self::Output>;
}

impl Cast for Datum {
    type Output = Datum;

    fn cast(&self, dtype: &DType) -> VortexResult<Datum> {
        Ok(match self {
            Datum::Scalar(scalar) => scalar.cast(dtype)?.into(),
            Datum::Vector(vector) => vector.cast(dtype)?.into(),
        })
    }
}

impl Cast for Scalar {
    type Output = Scalar;

    fn cast(&self, _dtype: &DType) -> VortexResult<Scalar> {
        vortex_bail!("Casting not implemented for scalar type {:?}", self);
    }
}

impl Cast for Vector {
    type Output = Vector;

    fn cast(&self, dtype: &DType) -> VortexResult<Vector> {
        // Switch to macro once all vector types implement Cast
        // match_each_vector!(self, |v| { Cast::cast(v, dtype) })
        match self {
            Vector::Null(v) => Cast::cast(v, dtype),
            Vector::Bool(v) => Cast::cast(v, dtype),
            Vector::Primitive(v) => Cast::cast(v, dtype),
            _ => {
                vortex_bail!("Casting not implemented for vector type {:?}", self);
            }
        }
    }
}
