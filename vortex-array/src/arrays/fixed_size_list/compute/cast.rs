// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::arrays::{FixedSizeListArray, FixedSizeListVTable};
use crate::compute::{CastKernel, CastKernelAdapter, cast};
use crate::vtable::ValidityHelper;
use crate::{ArrayRef, register_kernel};

impl CastKernel for FixedSizeListVTable {
    fn cast(&self, array: &FixedSizeListArray, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        let Some(target_element_type) = dtype.as_list_element_opt() else {
            return Ok(None);
        };

        // The list elements could technically be from either `FixedSizeList` or `List`.
        if dtype.is_list() {
            return Ok(None);
        }

        let elements = cast(array.elements(), target_element_type)?;
        let validity = array
            .validity()
            .clone()
            .cast_nullability(dtype.nullability())?;

        FixedSizeListArray::try_new(elements, array.list_size(), validity, array.len())
            .map(|a| Some(a.to_array()))
    }
}

register_kernel!(CastKernelAdapter(FixedSizeListVTable).lift());
