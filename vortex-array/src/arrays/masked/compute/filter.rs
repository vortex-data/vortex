// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Masked;
use crate::arrays::MaskedArray;
use crate::arrays::filter::FilterReduce;
use crate::arrays::masked::MaskedArraySlotsExt;

impl FilterReduce for Masked {
    fn filter(array: ArrayView<'_, Masked>, mask: &Mask) -> VortexResult<Option<ArrayRef>> {
        // Filter the validity to get the new validity
        let filtered_validity = array.validity()?.filter(mask)?;

        // Filter the child array
        // The child is guaranteed to have no nulls, so filtering it is straightforward
        let filtered_child = array.child().filter(mask.clone())?;

        // SAFETY: filtering a null-free child keeps it null-free, and the child and validity are
        // filtered by the same mask, so their lengths match.
        Ok(Some(
            unsafe { MaskedArray::new_unchecked(filtered_child, filtered_validity) }.into_array(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_session::VortexSession;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::MaskedArray;
    use crate::arrays::PrimitiveArray;
    use crate::compute::conformance::filter::test_filter_conformance;
    use crate::validity::Validity;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    #[rstest]
    #[case(
        MaskedArray::try_new(
            PrimitiveArray::from_iter([1i32, 2, 3, 4, 5]).into_array(),
            Validity::from_iter([true, true, false, true, false]),
            &mut SESSION.create_execution_ctx(),
        ).unwrap()
    )]
    #[case(
        MaskedArray::try_new(
            PrimitiveArray::from_iter([10i32, 20, 30]).into_array(),
            Validity::AllValid,
            &mut SESSION.create_execution_ctx(),
        ).unwrap()
    )]
    #[case(
        MaskedArray::try_new(
            PrimitiveArray::from_iter(0..100).into_array(),
            Validity::from_iter((0..100).map(|i| i % 3 != 0)),
            &mut SESSION.create_execution_ctx(),
        ).unwrap()
    )]
    fn test_filter_masked_conformance(#[case] array: MaskedArray) {
        test_filter_conformance(&array.into_array(), &mut SESSION.create_execution_ctx());
    }
}
