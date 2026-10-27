// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Masked;
use crate::arrays::masked::MaskedArraySlotsExt;
use crate::scalar_fn::fns::mask::Mask as MaskExpr;
use crate::scalar_fn::fns::mask::MaskReduce;
use crate::validity::Validity;

impl MaskReduce for Masked {
    fn mask(array: ArrayView<'_, Masked>, mask: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        // AND the existing validity mask with the new mask and push into child.
        let combined_mask = array
            .validity()?
            .and(Validity::Array(mask.clone()))?
            .to_array(array.len());
        let masked_child = MaskExpr::try_new(array.child().clone(), combined_mask)?.into_array();
        Ok(Some(masked_child))
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
    use crate::compute::conformance::mask::test_mask_conformance;
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
    fn test_mask_masked_conformance(#[case] array: MaskedArray) {
        test_mask_conformance(&array.into_array(), &mut SESSION.create_execution_ctx());
    }
}
