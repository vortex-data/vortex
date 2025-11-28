// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_vector::Vector;
use vortex_vector::VectorMut;
use vortex_vector::VectorMutOps;
use vortex_vector::VectorOps;
use vortex_vector::null::NullVector;

use crate::cast::Cast;

impl Cast for NullVector {
    fn cast(&self, dtype: &DType) -> VortexResult<Vector> {
        if dtype.is_nullable() {
            // We can create an all-null vector of _any_ type.
            let mut vec = VectorMut::with_capacity(dtype, self.len());
            vec.append_nulls(self.len());
            Ok(vec.freeze())
        } else {
            vortex_bail!("Cannot cast NullVector to non-nullable type {}", dtype);
        }
    }
}
