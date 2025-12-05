// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::Array;
use arrow_array::FixedSizeListArray;
use arrow_schema::DataType;
use arrow_schema::Field;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_vector::Vector;
use vortex_vector::fixed_size_list::FixedSizeListVector;

use crate::arrow::IntoArrow;
use crate::arrow::IntoVector;
use crate::arrow::nulls_to_mask;

impl IntoArrow for FixedSizeListVector {
    type Output = FixedSizeListArray;

    fn into_arrow(self) -> VortexResult<Self::Output> {
        let (elements, list_size, validity) = self.into_parts();

        let converted_elements = elements.as_ref().clone().into_arrow()?;
        let field = Arc::new(Field::new_list_field(
            converted_elements.data_type().clone(),
            true, // Vectors are always nullable.
        ));

        Ok(FixedSizeListArray::try_new(
            field,
            list_size as i32,
            converted_elements,
            validity.into(),
        )?)
    }
}

impl IntoVector for &FixedSizeListArray {
    type Output = FixedSizeListVector;

    fn into_vector(self) -> VortexResult<Self::Output> {
        let list_size = match self.data_type() {
            DataType::FixedSizeList(_, size) => *size as u32,
            dt => return Err(vortex_err!("expected FixedSizeList data type, got {}", dt)),
        };

        let elements: Vector = self.values().as_ref().into_vector()?;
        let validity = nulls_to_mask(self.nulls(), self.len());

        Ok(FixedSizeListVector::new(
            Arc::new(elements),
            list_size,
            validity,
        ))
    }
}
