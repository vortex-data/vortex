// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use itertools::Itertools;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Struct;
use crate::arrays::StructArray;
use crate::arrays::slice::SliceReduce;

impl SliceReduce for Struct {
    fn slice(array: ArrayView<'_, Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        let fields: Vec<_> = array
            .iter_unmasked_fields()
            .map(|field| field.slice(range.clone()))
            .try_collect()?;

        // SAFETY: Slicing preserves all StructArray invariants
        Ok(Some(
            unsafe {
                StructArray::new_unchecked(
                    fields,
                    array.struct_fields().clone(),
                    range.len(),
                    array.validity().slice(range)?,
                )
            }
            .into_array(),
        ))
    }
}
