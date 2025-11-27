// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Lossless casting of Vortex vectors for different logical data types.

mod bool;
mod null;
mod pvector;

use vortex_dtype::DType;
use vortex_error::vortex_bail;
use vortex_error::VortexResult;
use vortex_vector::Datum;

/// Trait for casting vectors to different data types.
pub trait Cast {
    /// Cast the vector to the specified data type.
    fn cast(&self, dtype: &DType) -> VortexResult<Datum>;
}

impl Cast for Datum {
    fn cast(&self, dtype: &DType) -> VortexResult<Datum> {
        // Switch to macro once all vector types implement Cast
        // match_each_vector!(self, |v| { Cast::cast(v, dtype) })
        match self {
            Datum::Null(v) => Cast::cast(v, dtype),
            Datum::Bool(v) => Cast::cast(v, dtype),
            Datum::Primitive(v) => Cast::cast(v, dtype),
            _ => {
                vortex_bail!("Casting not implemented for vector type {:?}", self);
            }
        }
    }
}
