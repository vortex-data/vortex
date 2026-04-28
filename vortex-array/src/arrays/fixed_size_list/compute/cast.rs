// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::FixedSizeList;
use crate::arrays::FixedSizeListArray;
use crate::arrays::fixed_size_list::FixedSizeListArrayExt;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::scalar_fn::fns::cast::CastKernel;
use crate::scalar_fn::fns::cast::CastReduce;
use crate::validity::Validity;

fn build_with_validity(
    array: ArrayView<'_, FixedSizeList>,
    elements: ArrayRef,
    validity: Validity,
) -> ArrayRef {
    // SAFETY: The only requirements for safety here are related to lengths, and no lengths have
    // changed here. So as long as the original array is valid, this is also valid.
    unsafe { FixedSizeListArray::new_unchecked(elements, array.list_size(), validity, array.len()) }
        .into_array()
}

/// Cast implementation for [`FixedSizeListArray`].
///
/// Recursively casts the inner elements array to the target element type while preserving the list
/// structure.
impl CastReduce for FixedSizeList {
    fn cast(array: ArrayView<'_, FixedSizeList>, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        let Some(target_element_type) = dtype.as_fixed_size_list_element_opt() else {
            return Ok(None);
        };

        let elements = array.elements().cast((**target_element_type).clone())?;
        let Some(validity) = array
            .validity()?
            .try_cast_nullability(dtype.nullability(), array.len())?
        else {
            return Ok(None);
        };

        Ok(Some(build_with_validity(array, elements, validity)))
    }
}

impl CastKernel for FixedSizeList {
    fn cast(
        array: ArrayView<'_, FixedSizeList>,
        dtype: &DType,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(target_element_type) = dtype.as_fixed_size_list_element_opt() else {
            return Ok(None);
        };

        let elements = array.elements().cast((**target_element_type).clone())?;
        let validity = array
            .validity()?
            .cast_nullability(dtype.nullability(), array.len(), ctx)?;

        Ok(Some(build_with_validity(array, elements, validity)))
    }
}
