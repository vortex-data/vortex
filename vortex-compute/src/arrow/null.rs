// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arrow_array::Array;
use arrow_array::NullArray;
use vortex_error::VortexResult;
use vortex_vector::VectorOps;
use vortex_vector::null::NullVector;

use crate::arrow::IntoArrow;
use crate::arrow::IntoVector;

impl IntoArrow for NullVector {
    type Output = NullArray;

    fn into_arrow(self) -> VortexResult<Self::Output> {
        Ok(NullArray::new(self.len()))
    }
}

impl IntoVector for &NullArray {
    type Output = NullVector;

    fn into_vector(self) -> VortexResult<Self::Output> {
        Ok(NullVector::new(self.len()))
    }
}
