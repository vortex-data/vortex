// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_vector::Scalar;
use vortex_vector::Vector;
use vortex_vector::match_each_pscalar;
use vortex_vector::match_each_pvector;
use vortex_vector::primitive::PrimitiveScalar;
use vortex_vector::primitive::PrimitiveVector;

use crate::cast::Cast;

impl Cast for PrimitiveVector {
    type Output = Vector;

    /// Dispatches to the underlying [`PVector<T>`](vortex_vector::primitive::PVector)
    /// implementation.
    fn cast(&self, target_dtype: &DType) -> VortexResult<Vector> {
        match_each_pvector!(self, |v| { Cast::cast(v, target_dtype) })
    }
}

impl Cast for PrimitiveScalar {
    type Output = Scalar;

    /// Dispatches to the underlying [`PScalar<T>`](vortex_vector::primitive::PScalar)
    /// implementation.
    fn cast(&self, target_dtype: &DType) -> VortexResult<Scalar> {
        match_each_pscalar!(self, |s| { Cast::cast(s, target_dtype) })
    }
}
