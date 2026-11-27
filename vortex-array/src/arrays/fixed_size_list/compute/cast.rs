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
    use std::sync::Arc;
    use std::sync::LazyLock;

    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::BoolArray;
    use crate::arrays::FixedSizeListArray;
    use crate::arrays::PrimitiveArray;
    use crate::assert_arrays_eq;
    use crate::builtins::ArrayBuiltins;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::validity::Validity;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    #[test]
    fn test_cast_fsl_element_dtype() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();

        // Non-nullable outer + non-nullable elements: hits CastReduce.
        let fsl = FixedSizeListArray::new(
            buffer![1i32, 2, 3, 4, 5, 6].into_array(),
            3,
            Validity::NonNullable,
            2,
        )
        .into_array();
        let target = DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::I64, Nullability::NonNullable)),
            3,
            Nullability::NonNullable,
        );
        let expected = FixedSizeListArray::new(
            buffer![1i64, 2, 3, 4, 5, 6].into_array(),
            3,
            Validity::NonNullable,
            2,
        );
        assert_arrays_eq!(fsl.cast(target)?, expected, &mut ctx);

        // Nullable outer with Validity::Array cast to NonNullable + widened nullable elements:
        // exercises CastKernel when trivial nullability cast returns None.
        let nullable_fsl = FixedSizeListArray::new(
            PrimitiveArray::from_option_iter([Some(1i32), None, Some(3), Some(4)]).into_array(),
            2,
            Validity::Array(BoolArray::from_iter([true, true]).into_array()),
            2,
        )
        .into_array();
        let target_non_null_outer = DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::I64, Nullability::Nullable)),
            2,
            Nullability::NonNullable,
        );
        let expected_non_null_outer = FixedSizeListArray::new(
            PrimitiveArray::from_option_iter([Some(1i64), None, Some(3), Some(4)]).into_array(),
            2,
            Validity::NonNullable,
            2,
        );
        assert_arrays_eq!(
            nullable_fsl.cast(target_non_null_outer)?,
            expected_non_null_outer,
            &mut ctx
        );

        Ok(())
    }
}
