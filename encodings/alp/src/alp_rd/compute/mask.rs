// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::LEGACY_SESSION;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::scalar_fn::ScalarFnFactoryExt;
use vortex_array::scalar_fn::EmptyOptions;
use vortex_array::scalar_fn::fns::mask::Mask as MaskExpr;
use vortex_array::scalar_fn::fns::mask::MaskReduce;
use vortex_error::VortexResult;

use crate::ALPRD;
use crate::ALPRDArrayExt;

impl MaskReduce for ALPRD {
    fn mask(array: ArrayView<'_, Self>, mask: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        let masked_left_parts = MaskExpr.try_new_array(
            array.left_parts().len(),
            EmptyOptions,
            [array.left_parts().clone(), mask.clone()],
        )?;
        // NOTE: `MaskReduce::mask` has a fixed trait signature without `ExecutionCtx`, so we
        // construct a legacy ctx locally at this trait boundary.
        Ok(Some(
            ALPRD::try_new(
                array.dtype().as_nullable(),
                masked_left_parts,
                array.left_parts_dictionary().clone(),
                array.right_parts().clone(),
                array.right_bit_width(),
                array.left_parts_patches(),
                &mut LEGACY_SESSION.create_execution_ctx(),
            )?
            .into_array(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::LEGACY_SESSION;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::compute::conformance::mask::test_mask_conformance;

    use crate::ALPRDFloat;
    use crate::RDEncoder;

    #[rstest]
    #[case(0.1f32, 0.2f32, 3e25f32)]
    #[case(0.1f64, 0.2f64, 3e100f64)]
    fn test_mask_simple<T: ALPRDFloat>(#[case] a: T, #[case] b: T, #[case] outlier: T) {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        test_mask_conformance(
            &RDEncoder::new(&[a, b])
                .encode(
                    PrimitiveArray::from_iter([a, b, outlier, b, outlier]).as_view(),
                    &mut ctx,
                )
                .into_array(),
        );
    }

    #[rstest]
    #[case(0.1f32, 3e25f32)]
    #[case(0.5f64, 1e100f64)]
    fn test_mask_with_nulls<T: ALPRDFloat>(#[case] a: T, #[case] outlier: T) {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        test_mask_conformance(
            &RDEncoder::new(&[a])
                .encode(
                    PrimitiveArray::from_option_iter([Some(a), None, Some(outlier), Some(a), None])
                        .as_view(),
                    &mut ctx,
                )
                .into_array(),
        );
    }
}
