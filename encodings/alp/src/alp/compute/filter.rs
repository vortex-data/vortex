// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::filter::FilterKernel;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::ALP;
use crate::ALPArrayExt;
use crate::ALPArraySlotsExt;

impl FilterKernel for ALP {
    fn filter(
        array: ArrayView<'_, Self>,
        mask: &Mask,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let patches = array
            .patches()
            .map(|p| p.filter(mask, ctx))
            .transpose()?
            .flatten();

        // SAFETY: filtering the values does not change correctness
        unsafe {
            Ok(Some(
                ALP::new_unchecked(
                    array.encoded().filter(mask.clone())?,
                    array.exponents(),
                    patches,
                )
                .into_array(),
            ))
        }
    }
}

#[cfg(test)]
mod test {
    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::LEGACY_SESSION;
    use vortex_array::ToCanonical;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::compute::conformance::filter::test_filter_conformance;
    use vortex_buffer::buffer;

    use crate::alp_encode;

    #[rstest]
    #[case(buffer![1.23f32, 4.56, 7.89, 10.11, 12.13].into_array())]
    #[case(buffer![100.1f64, 200.2, 300.3, 400.4, 500.5].into_array())]
    #[case(PrimitiveArray::from_option_iter([Some(1.1f32), None, Some(2.2), Some(3.3), None]).into_array())]
    #[case(buffer![42.42f64].into_array())]
    #[case(buffer![
        1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
        11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0
    ].into_array())]
    fn test_filter_alp_conformance(#[case] array: ArrayRef) {
        let alp = alp_encode(
            array.to_primitive().as_view(),
            None,
            &mut LEGACY_SESSION.create_execution_ctx(),
        )
        .unwrap();
        test_filter_conformance(&alp.into_array());
    }
}
