// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::arrays::slice::SliceReduce;
use vortex_array::vtable::child_to_validity;
use vortex_error::VortexResult;

use crate::Pco;

impl SliceReduce for Pco {
    fn slice(array: ArrayView<'_, Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        let unsliced_validity = child_to_validity(&array.slots()[0], array.dtype().nullability());
        Ok(Some(
            Pco::try_new(
                array.dtype().clone(),
                array._slice(range.start, range.end),
                unsliced_validity,
            )?
            .into_array(),
        ))
    }
}
