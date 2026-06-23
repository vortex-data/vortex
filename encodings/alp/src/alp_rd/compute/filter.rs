// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::filter::FilterKernel;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::ALPRD;
use crate::ALPRDArrayExt;

impl FilterKernel for ALPRD {
    fn filter(
        array: ArrayView<'_, Self>,
        mask: &Mask,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let left_parts_exceptions = array
            .left_parts_patches()
            .map(|patches| patches.filter(mask, ctx))
            .transpose()?
            .flatten();

        Ok(Some(
            ALPRD::try_new(
                array.dtype().clone(),
                array.left_parts().filter(mask.clone())?,
                array.left_parts_dictionary().clone(),
                array.right_parts().filter(mask.clone())?,
                array.right_bit_width(),
                left_parts_exceptions,
                ctx,
            )?
            .into_array(),
        ))
    }
}

#[cfg(test)]
mod test {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::compute::conformance::filter::test_filter_conformance;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_mask::Mask;
    use vortex_session::VortexSession;

    use crate::ALPRDArrayExt;
    use crate::ALPRDFloat;
    use crate::RDEncoder;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = array_session();
        crate::initialize(&session);
        session
    });

    #[rstest]
    #[case(0.1f32, 0.2f32, 3e25f32)]
    #[case(0.1f64, 0.2f64, 3e100f64)]
    fn test_filter<T: ALPRDFloat>(#[case] a: T, #[case] b: T, #[case] outlier: T) {
        let mut ctx = SESSION.create_execution_ctx();
        let array = PrimitiveArray::new(buffer![a, b, outlier], Validity::NonNullable);
        let encoded = RDEncoder::new(&[a, b]).encode(array.as_view(), &mut ctx);

        // Make sure that we're testing the exception pathway.
        assert!(encoded.left_parts_patches().is_some());

        // The first two values need no patching
        let filtered = encoded
            .filter(Mask::from_iter([true, false, true]))
            .unwrap();
        assert_arrays_eq!(filtered, PrimitiveArray::from_iter([a, outlier]), &mut ctx);
    }

    #[rstest]
    #[case(0.1f32, 0.2f32, 3e25f32)]
    #[case(0.1f64, 0.2f64, 3e100f64)]
    fn test_filter_simple<T: ALPRDFloat>(#[case] a: T, #[case] b: T, #[case] outlier: T) {
        let mut ctx = SESSION.create_execution_ctx();
        test_filter_conformance(
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
    fn test_filter_with_nulls<T: ALPRDFloat>(#[case] a: T, #[case] outlier: T) {
        let mut ctx = SESSION.create_execution_ctx();
        test_filter_conformance(
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
