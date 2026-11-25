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
use crate::arrays::fixed_size_list::FixedSizeListArraySlotsExt;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::scalar_fn::builtin::cast::CastKernel;
use crate::scalar_fn::builtin::cast::CastReduce;
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

        let Some(validity) = array
            .validity()?
            .trivially_cast_nullability(dtype.nullability(), array.len())?
        else {
            return Ok(None);
        };
        let elements = array.elements().cast((**target_element_type).clone())?;

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

        let validity = array
            .validity()?
            .cast_nullability(dtype.nullability(), array.len(), ctx)?;
        let elements = array.elements().cast((**target_element_type).clone())?;

        Ok(Some(build_with_validity(array, elements, validity)))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_buffer::buffer;
    use vortex_session::VortexSession;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::FixedSizeListArray;
    use crate::arrays::PrimitiveArray;
    use crate::compute::conformance::cast::test_cast_conformance;
    use crate::dtype::Nullability;
    use crate::validity::Validity;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    #[rstest]
    #[case::non_nullable(FixedSizeListArray::new(
        buffer![1i32, 2, 3, 4, 5, 6].into_array(),
        3,
        Validity::NonNullable,
        2,
    ))]
    #[case::nullable_with_nulls(FixedSizeListArray::new(
        buffer![1i32, 2, 3, 4, 5, 6, 7, 8, 9].into_array(),
        3,
        Validity::from_iter([true, false, true]),
        3,
    ))]
    #[case::nullable_all_valid(FixedSizeListArray::new(
        buffer![1i32, 2, 3, 4, 5, 6].into_array(),
        2,
        Validity::AllValid,
        3,
    ))]
    #[case::nullable_elements(FixedSizeListArray::new(
        PrimitiveArray::from_option_iter([Some(1i32), None, Some(3), Some(4)]).into_array(),
        2,
        Validity::NonNullable,
        2,
    ))]
    #[case::degenerate_size_0(FixedSizeListArray::new(
        PrimitiveArray::empty::<i32>(Nullability::NonNullable).into_array(),
        0,
        Validity::from_iter([true, false, true]),
        3,
    ))]
    #[case::empty(FixedSizeListArray::new(
        PrimitiveArray::empty::<i32>(Nullability::NonNullable).into_array(),
        3,
        Validity::NonNullable,
        0,
    ))]
    #[case::large(FixedSizeListArray::new(
        PrimitiveArray::from_iter(0i32..300).into_array(),
        10,
        Validity::NonNullable,
        30,
    ))]
    fn test_cast_fsl_conformance(#[case] array: FixedSizeListArray) {
        test_cast_conformance(&array.into_array(), &mut SESSION.create_execution_ctx());
    }
}
