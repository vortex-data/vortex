// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_vector::Scalar;
use vortex_vector::Vector;
use vortex_vector::decimal::DecimalScalar;
use vortex_vector::decimal::DecimalVector;
use vortex_vector::match_each_dscalar;
use vortex_vector::match_each_dvector;

use crate::cast::Cast;

impl Cast for DecimalVector {
    type Output = Vector;

    /// Dispatches to the underlying [`DVector<D>`](vortex_vector::decimal::DVector) implementation.
    fn cast(&self, target_dtype: &DType) -> VortexResult<Vector> {
        match_each_dvector!(self, |v| { Cast::cast(v, target_dtype) })
    }
}

impl Cast for DecimalScalar {
    type Output = Scalar;

    /// Dispatches to the underlying [`DScalar<D>`](vortex_vector::decimal::DScalar) implementation.
    fn cast(&self, target_dtype: &DType) -> VortexResult<Scalar> {
        match_each_dscalar!(self, |s| { Cast::cast(s, target_dtype) })
    }
}
