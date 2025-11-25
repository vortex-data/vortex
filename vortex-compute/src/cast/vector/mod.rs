// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Lossless casting of Vortex vectors for different logical data types.

use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_vector::Vector;

use crate::cast::Cast;

mod bool;
mod null;
mod primitive;

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
