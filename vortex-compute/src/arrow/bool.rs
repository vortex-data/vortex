// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::{ArrayRef, BooleanArray};
use vortex_error::VortexResult;
use vortex_vector::bool::BoolVector;

use crate::arrow::IntoArrow;

impl IntoArrow<ArrayRef> for BoolVector {
    fn into_arrow(self) -> VortexResult<ArrayRef> {
        let (bits, validity) = self.into_parts();
        Ok(Arc::new(BooleanArray::new(
            bits.into(),
            validity.into_arrow()?,
        )))
    }
}
