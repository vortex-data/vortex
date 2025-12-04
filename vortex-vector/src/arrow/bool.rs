// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use crate::arrow::nulls_to_mask;
use crate::bool::BoolVector;
use arrow_array::Array;
use arrow_array::ArrayRef;
use arrow_array::BooleanArray;
use vortex_buffer::BitBuffer;
use vortex_error::VortexError;
use vortex_error::vortex_err;

impl TryFrom<BoolVector> for ArrayRef {
    type Error = VortexError;

    fn try_from(value: BoolVector) -> Result<Self, Self::Error> {
        let (bits, validity) = value.into_parts();
        Ok(Arc::new(BooleanArray::new(bits.into(), validity.into())))
    }
}

impl From<&BooleanArray> for BoolVector {
    fn from(value: &BooleanArray) -> Self {
        let bits = BitBuffer::from(value.values().clone());
        let validity = nulls_to_mask(value.nulls(), value.len());
        BoolVector::new(bits, validity)
    }
}

impl TryFrom<&dyn Array> for BoolVector {
    type Error = VortexError;

    fn try_from(value: &dyn Array) -> Result<Self, Self::Error> {
        let array = value
            .as_any()
            .downcast_ref::<BooleanArray>()
            .ok_or_else(|| vortex_err!("expected BooleanArray, got {}", value.data_type()))?;
        Ok(BoolVector::from(array))
    }
}
