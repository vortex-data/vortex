// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use crate::Vector;
use crate::VectorOps;
use crate::arrow::nulls_to_mask;
use crate::struct_::StructVector;
use arrow_array::Array;
use arrow_array::ArrayRef;
use arrow_array::StructArray;
use arrow_schema::Field;
use arrow_schema::Fields;
use vortex_error::VortexError;
use vortex_error::vortex_err;

impl TryFrom<StructVector> for ArrayRef {
    type Error = VortexError;

    fn try_from(value: StructVector) -> Result<Self, Self::Error> {
        let len = value.len();
        let (fields, validity) = value.into_parts();
        let arrow_fields = fields
            .iter()
            .map(|field| ArrayRef::try_from(field.clone()))
            .collect::<Result<Vec<ArrayRef>, _>>()?;

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
            StructArray::new_unchecked_with_length(fields, arrow_fields, validity.into(), len)
        }))
    }
}

impl TryFrom<ArrayRef> for StructVector {
    type Error = VortexError;

    fn try_from(value: ArrayRef) -> Result<Self, Self::Error> {
        let array = value
            .as_any()
            .downcast_ref::<StructArray>()
            .ok_or_else(|| vortex_err!("expected StructArray, got {}", value.data_type()))?;

        let fields: Box<[Vector]> = array
            .columns()
            .iter()
            .map(|col| Vector::try_from(col.clone()))
            .collect::<Result<_, _>>()?;

        let validity = nulls_to_mask(array.nulls(), array.len());

        Ok(StructVector::new(Arc::new(fields), validity))
    }
}
