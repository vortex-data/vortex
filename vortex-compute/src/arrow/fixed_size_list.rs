// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::{ArrayRef, FixedSizeListArray};
use arrow_schema::Field;
use vortex_error::VortexResult;
use vortex_vector::FixedSizeListVector;

use crate::arrow::IntoArrow;

impl IntoArrow<ArrayRef> for FixedSizeListVector {
    fn into_arrow(self) -> VortexResult<ArrayRef> {
        let (elements, list_size, validity) = self.into_parts();

        let converted_elements = elements.as_ref().clone().into_arrow()?;
        let field = Arc::new(Field::new_list_field(
            converted_elements.data_type().clone(),
            true, // Vectors are always nullable.
        ));

        Ok(Arc::new(FixedSizeListArray::try_new(
            field,
            list_size as i32,
            converted_elements,
            validity.into_arrow()?,
        )?))
    }
}
