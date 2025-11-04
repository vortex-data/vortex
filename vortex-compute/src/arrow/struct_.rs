// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::{ArrayRef, StructArray};
use arrow_schema::{Field, Fields};
use vortex_error::VortexResult;
use vortex_vector::VectorOps;
use vortex_vector::struct_::StructVector;

use crate::arrow::IntoArrow;

impl IntoArrow<ArrayRef> for StructVector {
    fn into_arrow(self) -> VortexResult<ArrayRef> {
        let len = self.len();
        let (fields, validity) = self.into_parts();
        let arrow_fields = fields
            .iter()
            .map(|field| field.clone().into_arrow())
            .collect::<VortexResult<Vec<ArrayRef>>>()?;

        // We need to make up the field names since vectors are unnamed, so we just use the field
        // indices.
        let fields = Fields::from(
            (0..arrow_fields.len())
                .map(|i| {
                    Field::new(
                        i.to_string(),
                        arrow_fields[i].data_type().clone(),
                        true, // Vectors are always nullable.
                    )
                })
                .collect::<Vec<Field>>(),
        );

        // SAFETY: Since all of these components came from a valid `StructVector`, we know that all
        // of the lengths of the vectors are correct. Additionally, all extra metadata is directly
        // derived from the existing components so all invariants are upheld.
        Ok(Arc::new(unsafe {
            StructArray::new_unchecked_with_length(
                fields,
                arrow_fields,
                validity.into_arrow()?,
                len,
            )
        }))
    }
}
