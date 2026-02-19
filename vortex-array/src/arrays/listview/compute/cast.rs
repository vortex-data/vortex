// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::ListViewArray;
use crate::arrays::ListViewVTable;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::expr::CastReduce;
use crate::vtable::ValidityHelper;

impl CastReduce for ListViewVTable {
    fn cast(array: &ListViewArray, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        // Check if we're casting to a `List` type.
        let Some(target_element_type) = dtype.as_list_element_opt() else {
            return Ok(None);
        };

        // Cast the elements to the target element type.
        let new_elements = array
            .elements()
            .cast((**target_element_type).clone())?
            .to_canonical()?
            .into_array();
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
                .with_zero_copy_to_list(array.is_zero_copy_to_list())
            }
            .into_array(),
        ))
    }
}
