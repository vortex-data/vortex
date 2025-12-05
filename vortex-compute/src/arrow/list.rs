// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arrow_array::ListViewArray;
use arrow_buffer::ScalarBuffer;
use arrow_schema::Field;
use arrow_schema::FieldRef;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_vector::listview::ListViewVector;
use vortex_vector::match_each_integer_pvector;

use crate::arrow::IntoArrow;

impl IntoArrow for ListViewVector {
    type Output = ListViewArray;

    #[allow(clippy::unnecessary_fallible_conversions)]
    #[allow(clippy::useless_conversion)]
    fn into_arrow(self) -> VortexResult<Self::Output> {
        let (elements, offsets, sizes, validity) = self.into_parts();

        let elements = elements.as_ref().clone().into_arrow()?;

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

        Ok(ListViewArray::new(
            FieldRef::new(Field::new("elements", elements.data_type().clone(), true)),
            offsets,
            sizes,
            elements,
            validity.into(),
        ))
    }
}
