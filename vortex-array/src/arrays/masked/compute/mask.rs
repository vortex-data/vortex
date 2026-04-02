// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::array::ArrayView;
use crate::arrays::Masked;
use crate::arrays::scalar_fn::ScalarFnArrayExt;
use crate::scalar_fn::EmptyOptions;
use crate::scalar_fn::fns::mask::Mask as MaskExpr;
use crate::scalar_fn::fns::mask::MaskReduce;
use crate::validity::Validity;

impl MaskReduce for Masked {
    fn mask(array: ArrayView<'_, Masked>, mask: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        // AND the existing validity mask with the new mask and push into child.
        let combined_mask = array
            .validity()
            .and(Validity::Array(mask.clone()))?
            .to_array(array.len());
        let masked_child = MaskExpr.try_new_array(
            array.child().len(),
            EmptyOptions,
            [array.child().clone(), combined_mask],
        )?;
        Ok(Some(masked_child))
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use crate::IntoArray;
    use crate::arrays::MaskedArray;
    use crate::arrays::PrimitiveArray;
    use crate::compute::conformance::mask::test_mask_conformance;
    use crate::validity::Validity;

    #[rstest]
    #[case(
        MaskedArray::try_new(
            PrimitiveArray::from_iter([1i32, 2, 3, 4, 5]).into_array(),
            Validity::from_iter([true, true, false, true, false])
        ).unwrap()
    )]
    #[case(
        MaskedArray::try_new(
            PrimitiveArray::from_iter([10i32, 20, 30]).into_array(),
            Validity::AllValid
        ).unwrap()
    )]
    #[case(
        MaskedArray::try_new(
            PrimitiveArray::from_iter(0..100).into_array(),
            Validity::from_iter((0..100).map(|i| i % 3 != 0))
        ).unwrap()
    )]
    fn test_mask_masked_conformance(#[case] array: MaskedArray) {
        test_mask_conformance(&array.into_array());
    }
}
