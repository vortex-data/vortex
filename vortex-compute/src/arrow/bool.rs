// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::Array;
use arrow_array::ArrayRef;
use arrow_array::BooleanArray;
use vortex_buffer::BitBuffer;
use vortex_error::VortexResult;
use vortex_vector::bool::BoolVector;

use crate::arrow::IntoArrow;
use crate::arrow::IntoVector;
use crate::arrow::nulls_to_mask;

impl IntoArrow for BoolVector {
    type Output = ArrayRef;

    fn into_arrow(self) -> VortexResult<Self::Output> {
        let (bits, validity) = self.into_parts();
        Ok(Arc::new(BooleanArray::new(bits.into(), validity.into())))
    }
}

impl IntoVector for &BooleanArray {
    type Output = BoolVector;

    fn into_vector(self) -> VortexResult<Self::Output> {
        let bits = BitBuffer::from(self.values().clone());
        let validity = nulls_to_mask(self.nulls(), self.len());
        Ok(BoolVector::new(bits, validity))
    }
}
