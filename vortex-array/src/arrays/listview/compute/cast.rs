// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::arrays::{ListViewArray, ListViewVTable};
use crate::compute::{self, CastKernel, CastKernelAdapter};
use crate::vtable::ValidityHelper;
use crate::{ArrayRef, register_kernel};

impl CastKernel for ListViewVTable {
    fn cast(&self, array: &ListViewArray, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        // Check if we're casting to a `List` type.
        let Some(target_element_type) = dtype.as_list_element_opt() else {
            return Ok(None);
        };

        // Cast the elements to the target element type.
        let new_elements = compute::cast(array.elements(), target_element_type)?;
        let validity = array
            .validity()
            .clone()
            .cast_nullability(dtype.nullability(), array.len())?;

        // SAFETY: Since `cast` is length-preserving, all of the invariants remain the same.
        Ok(Some(
            unsafe {
                ListViewArray::new_unchecked(
                    new_elements,
                    array.offsets().clone(),
                    array.sizes().clone(),
                    validity,
                )
            }
            .to_array(),
        ))
    }
}

register_kernel!(CastKernelAdapter(ListViewVTable).lift());
