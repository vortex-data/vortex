// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use crate::VectorOps;
use crate::null::NullVector;
use arrow_array::Array;
use arrow_array::ArrayRef;
use arrow_array::NullArray;
use vortex_error::VortexError;
use vortex_error::vortex_err;

impl TryFrom<NullVector> for ArrayRef {
    type Error = VortexError;

    fn try_from(value: NullVector) -> Result<Self, Self::Error> {
        Ok(Arc::new(NullArray::new(value.len())))
    }
}

impl TryFrom<ArrayRef> for NullVector {
    type Error = VortexError;

    fn try_from(value: ArrayRef) -> Result<Self, Self::Error> {
        let array = value
            .as_any()
            .downcast_ref::<NullArray>()
            .ok_or_else(|| vortex_err!("expected NullArray, got {}", value.data_type()))?;
        Ok(NullVector::new(array.len()))
    }
}
