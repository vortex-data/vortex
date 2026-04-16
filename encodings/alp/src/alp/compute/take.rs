// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::dict::TakeExecute;
use vortex_error::VortexResult;

use crate::ALP;
use crate::ALPArrayExt;
use crate::ALPArraySlotsExt;

impl TakeExecute for ALP {
    fn take(
        array: ArrayView<'_, Self>,
        indices: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let taken_encoded = array.encoded().take(indices.clone())?;
        let taken_patches = array
            .patches()
            .map(|p| p.take(indices, ctx))
            .transpose()?
            .flatten()
            .map(|patches| {
                patches.cast_values(
                    &array
                        .dtype()
                        .with_nullability(taken_encoded.dtype().nullability()),
                )
            })
            .transpose()?;
        Ok(Some(
            ALP::new(taken_encoded, array.exponents(), taken_patches).into_array(),
        ))
    }
}

#[cfg(test)]
mod test {
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::LEGACY_SESSION;
    use vortex_array::ToCanonical;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::compute::conformance::take::test_take_conformance;
    use vortex_buffer::buffer;

    use crate::alp_encode;

    #[rstest]
    #[case(buffer![1.23f32, 4.56, 7.89, 10.11, 12.13].into_array())]
    #[case(buffer![100.1f64, 200.2, 300.3, 400.4, 500.5].into_array())]
    #[case(PrimitiveArray::from_option_iter([Some(1.1f32), None, Some(2.2), Some(3.3), None]).into_array())]
    #[case(buffer![42.42f64].into_array())]
    fn test_take_alp_conformance(#[case] array: vortex_array::ArrayRef) {
        let alp = alp_encode(
            array.to_primitive().as_view(),
            None,
            &mut LEGACY_SESSION.create_execution_ctx(),
        )
        .unwrap();
        test_take_conformance(&alp.into_array());
    }
}
