// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_array::cast::AsArray;
use vortex_error::{VortexError, VortexResult, vortex_bail};
use vortex_mask::Mask;

use crate::arrow::compute::to_arrow_preferred;
use crate::compute::filter;
use crate::validity::Validity;
use crate::vtable::ValidityHelper;
use crate::{Array, Canonical, ToCanonical};

impl TryFrom<&dyn Array> for RecordBatch {
    type Error = VortexError;

    fn try_from(value: &dyn Array) -> VortexResult<Self> {
        let Canonical::Struct(struct_array) = value.to_canonical() else {
            vortex_bail!("RecordBatch can only be constructed from ")
        };
        // If there are any top-level nulls, mask them from the result.
        match struct_array.validity() {
            // Easy
            Validity::NonNullable | Validity::AllValid => Ok(RecordBatch::from(
                to_arrow_preferred(struct_array.as_ref())?.as_struct(),
            )),
            Validity::AllInvalid => {
                // New empty struct array
                let schema = Arc::new(value.dtype().to_arrow_schema()?);
                Ok(RecordBatch::new_empty(schema))
            }
            Validity::Array(validity) => {
                // Create new ones here instead
                let valid = Mask::from_buffer(validity.to_bool().boolean_buffer().clone());
                let masked = filter(value, &valid)?;

                Ok(RecordBatch::from(
                    to_arrow_preferred(masked.as_ref())?.as_struct(),
                ))
            }
        }
    }
}
