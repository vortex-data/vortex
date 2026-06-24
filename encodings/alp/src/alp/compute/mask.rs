// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::scalar_fn::fns::mask::MaskKernel;
use vortex_array::scalar_fn::fns::mask::MaskReduce;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;

use crate::ALP;
use crate::ALPArrayExt;
use crate::ALPArraySlotsExt;

impl MaskReduce for ALP {
    fn mask(array: ArrayView<'_, Self>, mask: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        // Masking sparse patches requires reading indices, fall back to kernel.
        if array.patches().is_some() {
            return Ok(None);
        }
        let masked_encoded = array.encoded().clone().mask(mask.clone())?;
        Ok(Some(
            ALP::new(masked_encoded, array.exponents(), None).into_array(),
        ))
    }
}

impl MaskKernel for ALP {
    fn mask(
        array: ArrayView<'_, Self>,
        mask: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let vortex_mask = Validity::Array(mask.not()?).execute_mask(array.len(), ctx)?;
        let masked_encoded = array.encoded().clone().mask(mask.clone())?;
        let masked_dtype = array
            .dtype()
            .with_nullability(masked_encoded.dtype().nullability());
        let masked_patches = array
            .patches()
            .map(|p| p.mask(&vortex_mask, ctx))
            .transpose()?
            .flatten()
            .map(|patches| patches.cast_values(&masked_dtype))
            .transpose()?;
        Ok(Some(
            ALP::new(masked_encoded, array.exponents(), masked_patches).into_array(),
        ))
    }
}

#[cfg(test)]
mod test {
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::compute::conformance::mask::test_mask_conformance;
    use vortex_array::dtype::Nullability;
    use vortex_array::scalar_fn::fns::mask::MaskKernel;
    use vortex_buffer::buffer;

    use crate::alp::array::ALPArrayExt;
    use crate::alp_encode;

    #[rstest]
    #[case(buffer![10.5f32, 20.5, 30.5, 40.5, 50.5].into_array())]
    #[case(buffer![1000.123f64, 2000.456, 3000.789, 4000.012, 5000.345].into_array())]
    #[case(PrimitiveArray::from_option_iter([Some(1.1f32), None, Some(2.2), Some(3.3), None]).into_array())]
    #[case(buffer![99.99f64].into_array())]
    #[case(buffer![
        0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
        1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0
    ].into_array())]
    fn test_mask_alp_conformance(#[case] array: vortex_array::ArrayRef) {
        let mut ctx = array_session().create_execution_ctx();
        let array_primitive = array.execute::<PrimitiveArray>(&mut ctx).unwrap();
        let alp = alp_encode(array_primitive.as_view(), None, &mut ctx).unwrap();
        test_mask_conformance(&alp.into_array());
    }

    #[test]
    fn test_mask_alp_with_patches() {
        use std::f64::consts::PI;
        let mut ctx = array_session().create_execution_ctx();
        // PI doesn't encode cleanly with ALP, so it creates patches.
        let values: Vec<f64> = (0..100)
            .map(|i| if i % 4 == 3 { PI } else { 1.0 })
            .collect();
        let array = PrimitiveArray::from_iter(values);
        let alp = alp_encode(array.as_view(), None, &mut ctx).unwrap();
        assert!(alp.patches().is_some(), "expected patches");
        test_mask_conformance(&alp.into_array());
    }

    #[test]
    fn test_mask_alp_with_patches_casts_surviving_patch_values_to_nullable() {
        let mut ctx = array_session().create_execution_ctx();
        let values = PrimitiveArray::from_iter([1.234f32, f32::NAN, 2.345, f32::INFINITY, 3.456]);
        let alp = alp_encode(values.as_view(), None, &mut ctx).unwrap();
        assert!(alp.patches().is_some(), "expected patches");

        let keep_mask = BoolArray::from_iter([false, true, true, true, true]).into_array();
        let masked = <crate::ALP as MaskKernel>::mask(alp.as_view(), &keep_mask, &mut ctx)
            .unwrap()
            .unwrap();

        let masked_alp = masked.as_opt::<crate::ALP>().unwrap();
        let masked_patches = masked_alp.patches().unwrap();

        assert_eq!(masked.dtype().nullability(), Nullability::Nullable);
        assert_eq!(masked_patches.dtype().nullability(), Nullability::Nullable);
    }
}
