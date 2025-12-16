// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_vector::Scalar;
use vortex_vector::Vector;
use vortex_vector::VectorMut;
use vortex_vector::VectorMutOps;
use vortex_vector::VectorOps;
use vortex_vector::null::NullScalar;
use vortex_vector::null::NullVector;

use crate::cast::Cast;

impl Cast for NullVector {
    type Output = Vector;

    /// Casts to any nullable target type by creating an all-null vector.
    fn cast(&self, target_dtype: &DType) -> VortexResult<Vector> {
        if target_dtype.is_nullable() {
            // We can create an all-null vector of _any_ type.
            let mut vec = VectorMut::with_capacity(target_dtype, self.len());
            vec.append_nulls(self.len());
            Ok(vec.freeze())
        } else {
            vortex_bail!(
                "Cannot cast NullVector to non-nullable type {}",
                target_dtype
            );
        }
    }
}

impl Cast for NullScalar {
    type Output = Scalar;

    /// Casts to any nullable target type by creating a null scalar.
    fn cast(&self, target_dtype: &DType) -> VortexResult<Scalar> {
        if target_dtype.is_nullable() {
            Ok(Scalar::null(target_dtype))
        } else {
            vortex_bail!(
                "Cannot cast NullScalar to non-nullable type {}",
                target_dtype
            );
        }
    }
}
