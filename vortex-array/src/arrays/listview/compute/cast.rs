// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::ListView;
use crate::arrays::ListViewArray;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::scalar_fn::fns::cast::CastReduce;

impl CastReduce for ListView {
    fn cast(array: ArrayView<'_, ListView>, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        // Check if we're casting to a `List` type.
        let Some(target_element_type) = dtype.as_list_element_opt() else {
            return Ok(None);
        };

        // Cast the elements to the target element type.
        let new_elements = array.elements().cast((**target_element_type).clone())?;
        let validity = array
            .validity()
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
