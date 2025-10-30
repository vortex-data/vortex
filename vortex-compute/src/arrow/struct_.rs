// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::{ArrayRef, StructArray};
use arrow_schema::{Field, Fields};
use vortex_error::VortexResult;
use vortex_vector::StructVector;

use crate::arrow::IntoArrow;

impl IntoArrow<ArrayRef> for StructVector {
    fn into_arrow(self) -> VortexResult<ArrayRef> {
        let (fields, validity) = self.into_parts();
        let arrow_fields = fields
            .iter()
            .map(|field| field.clone().into_arrow())
            .collect::<VortexResult<Vec<ArrayRef>>>()?;

        // We need to make up the field names since vectors are unnamed.
        let fields = Fields::from(
            (0..arrow_fields.len())
                .map(|i| Field::new(i.to_string(), arrow_fields[i].data_type().clone(), true))
                .collect::<Vec<Field>>(),
        );

        Ok(Arc::new(StructArray::new(
            fields,
            arrow_fields,
            validity.into_arrow()?,
        )))
    }
}
