// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::{ArrayRef, NullArray};
use vortex_error::VortexResult;
use vortex_vector::VectorOps;
use vortex_vector::null::NullVector;

use crate::arrow::IntoArrow;

impl IntoArrow<ArrayRef> for NullVector {
    fn into_arrow(self) -> VortexResult<ArrayRef> {
        Ok(Arc::new(NullArray::new(self.len())))
    }
}
