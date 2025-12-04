// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use crate::Vector;
use crate::arrow::nulls_to_mask;
use crate::fixed_size_list::FixedSizeListVector;
use arrow_array::Array;
use arrow_array::ArrayRef;
use arrow_array::FixedSizeListArray;
use arrow_schema::DataType;
use arrow_schema::Field;
use vortex_error::VortexError;
use vortex_error::vortex_err;

impl TryFrom<FixedSizeListVector> for ArrayRef {
    type Error = VortexError;

    fn try_from(value: FixedSizeListVector) -> Result<Self, Self::Error> {
        let (elements, list_size, validity) = value.into_parts();

        let converted_elements = ArrayRef::try_from(elements.as_ref().clone())?;
        let field = Arc::new(Field::new_list_field(
            converted_elements.data_type().clone(),
            true, // Vectors are always nullable.
        ));

        Ok(Arc::new(FixedSizeListArray::try_new(
            field,
            list_size as i32,
            converted_elements,
            validity.into(),
        )?))
    }
}

impl TryFrom<ArrayRef> for FixedSizeListVector {
    type Error = VortexError;

    fn try_from(value: ArrayRef) -> Result<Self, Self::Error> {
        let array = value
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .ok_or_else(|| vortex_err!("expected FixedSizeListArray, got {}", value.data_type()))?;

        let list_size = match array.data_type() {
            DataType::FixedSizeList(_, size) => *size as u32,
            dt => return Err(vortex_err!("expected FixedSizeList data type, got {}", dt)),
        };

        let elements = Vector::try_from(array.values().clone())?;
        let validity = nulls_to_mask(array.nulls(), array.len());

        Ok(FixedSizeListVector::new(
            Arc::new(elements),
            list_size,
            validity,
        ))
    }
}
