// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::FixedSizeList;
use crate::arrays::FixedSizeListArray;
use crate::arrays::fixed_size_list::FixedSizeListArrayExt;
use crate::arrays::fixed_size_list::FixedSizeListArraySlotsExt;
use crate::scalar_fn::builtin::mask::MaskReduce;
use crate::validity::Validity;

impl MaskReduce for FixedSizeList {
    const VALIDITY_IS_METADATA_ONLY: bool = true;

    fn mask(
        array: ArrayView<'_, FixedSizeList>,
        mask: &ArrayRef,
    ) -> VortexResult<Option<ArrayRef>> {
        // SAFETY: masking the validity does not affect the invariants
        Ok(Some(
            unsafe {
                FixedSizeListArray::new_unchecked(
                    array.elements().clone(),
                    array.list_size(),
                    array.validity()?.and(Validity::Array(mask.clone()))?,
                    array.len(),
                )
            }
            .into_array(),
        ))
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
    use crate::compute::conformance::mask::test_mask_conformance;
    use crate::dtype::Nullability;
    use crate::validity::Validity;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    #[rstest]
    #[case::non_nullable(FixedSizeListArray::new(
        buffer![1i32, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12].into_array(),
        3,
        Validity::NonNullable,
        4,
    ))]
    #[case::nullable(FixedSizeListArray::new(
        PrimitiveArray::from_iter(0i32..18).into_array(),
        3,
        Validity::from_iter([true, false, true, true, false, true]),
        6,
    ))]
    #[case::nullable_elements(FixedSizeListArray::new(
        PrimitiveArray::from_option_iter([Some(1i32), None, Some(3), Some(4), None, Some(6)])
            .into_array(),
        2,
        Validity::NonNullable,
        3,
    ))]
    #[case::degenerate_size_0(FixedSizeListArray::new(
        PrimitiveArray::empty::<i32>(Nullability::NonNullable).into_array(),
        0,
        Validity::from_iter([true, false, true, true, false]),
        5,
    ))]
    #[case::single(FixedSizeListArray::new(
        buffer![10i32, 20].into_array(),
        2,
        Validity::NonNullable,
        1,
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
    fn test_mask_fsl_conformance(#[case] array: FixedSizeListArray) {
        test_mask_conformance(&array.into_array(), &mut SESSION.create_execution_ctx());
    }
}
