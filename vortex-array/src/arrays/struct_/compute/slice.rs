// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use itertools::Itertools;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::SliceReduce;
use crate::arrays::StructArray;
use crate::arrays::StructVTable;
use crate::vtable::ValidityHelper;

impl SliceReduce for StructVTable {
    fn slice(array: &Self::Array, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        let fields: Vec<_> = array
            .unmasked_fields()
            .iter()
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
