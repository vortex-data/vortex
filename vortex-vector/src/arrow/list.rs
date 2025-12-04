// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::listview::ListViewVector;
use crate::match_each_integer_pvector;
use arrow_array::Array;
use arrow_array::ArrayRef;
use arrow_array::ListViewArray;
use arrow_buffer::ScalarBuffer;
use arrow_schema::Field;
use arrow_schema::FieldRef;
use std::sync::Arc;
use vortex_error::VortexError;
use vortex_error::VortexExpect;

impl TryFrom<ListViewVector> for ArrayRef {
    type Error = VortexError;

    #[allow(clippy::unnecessary_fallible_conversions)]
    #[allow(clippy::useless_conversion)]
    fn try_from(value: ListViewVector) -> Result<Self, Self::Error> {
        let (elements, offsets, sizes, validity) = value.into_parts();

        let elements = ArrayRef::try_from(elements.as_ref().clone())?;

        let offsets = match_each_integer_pvector!(offsets, |p| {
            ScalarBuffer::<i32>::from_iter(
                p.elements()
                    .iter()
                    .map(|e| i32::try_from(*e).vortex_expect("Failed to convert to i32")),
            )
        });
        let sizes = match_each_integer_pvector!(sizes, |p| {
            ScalarBuffer::<i32>::from_iter(
                p.elements()
                    .iter()
                    .map(|e| i32::try_from(*e).vortex_expect("Failed to convert to i32")),
            )
        });

        Ok(Arc::new(ListViewArray::new(
            FieldRef::new(Field::new("elements", elements.data_type().clone(), true)),
            offsets,
            sizes,
            elements,
            validity.into(),
        )))
    }
}
