// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::arrays::{FixedSizeListArray, FixedSizeListVTable};
use crate::compute::{CastKernel, CastKernelAdapter, cast};
use crate::vtable::ValidityHelper;
use crate::{ArrayRef, register_kernel};

/// Cast implementation for [`FixedSizeListArray`].
///
/// Recursively casts the inner elements array to the target element type while preserving the list
/// structure.
impl CastKernel for FixedSizeListVTable {
    fn cast(&self, array: &FixedSizeListArray, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        let Some(target_element_type) = dtype.as_fixed_size_list_element_opt() else {
            return Ok(None);
        };

        let elements = cast(array.elements(), target_element_type)?;
        let validity = array
            .validity()
            .clone()
            .cast_nullability(dtype.nullability(), array.len())?;

        Ok(Some(
            // SAFETY: The only requirements for safety here are related to lengths, and no lengths
            // have changed here. So as long as the original array is valid, this is also valid.
            unsafe {
                FixedSizeListArray::new_unchecked(
                    elements,
                    array.list_size(),
                    validity,
                    array.len(),
                )
            }
            .to_array(),
        ))
    }
}

register_kernel!(CastKernelAdapter(FixedSizeListVTable).lift());
