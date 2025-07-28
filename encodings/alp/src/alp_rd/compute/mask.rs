// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::compute::{MaskKernel, MaskKernelAdapter, mask};
use vortex_array::{ArrayRef, IntoArray, register_kernel};
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::{ALPRDArray, ALPRDVTable};

impl MaskKernel for ALPRDVTable {
    fn mask(&self, array: &ALPRDArray, filter_mask: &Mask) -> VortexResult<ArrayRef> {
        Ok(ALPRDArray::try_new(
            array.dtype().as_nullable(),
            mask(array.left_parts(), filter_mask)?,
            array.left_parts_dictionary().clone(),
            array.right_parts().clone(),
            array.right_bit_width(),
            array.left_parts_patches().cloned(),
        )?
        .into_array())
    }
}

register_kernel!(MaskKernelAdapter(ALPRDVTable).lift());

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::compute::conformance::mask::test_mask;

    use crate::{ALPRDFloat, RDEncoder};

    #[rstest]
    #[case(0.1f32, 0.2f32, 3e25f32)]
    #[case(0.1f64, 0.2f64, 3e100f64)]
    fn test_mask_simple<T: ALPRDFloat>(#[case] a: T, #[case] b: T, #[case] outlier: T) {
        test_mask(
            &RDEncoder::new(&[a, b])
                .encode(&PrimitiveArray::from_iter([a, b, outlier, b, outlier]))
                .into_array(),
        );
    }

    #[rstest]
    #[case(0.1f32, 3e25f32)]
    #[case(0.5f64, 1e100f64)]
    fn test_mask_with_nulls<T: ALPRDFloat>(#[case] a: T, #[case] outlier: T) {
        test_mask(
            &RDEncoder::new(&[a])
                .encode(&PrimitiveArray::from_option_iter([
                    Some(a),
                    None,
                    Some(outlier),
                    Some(a),
                    None,
                ]))
                .into_array(),
        );
    }
    
    #[test]
    fn test_numeric_alprd_array() {
        use vortex_array::compute::conformance::binary_numeric::test_numeric;
        
        // Test binary numeric operations with f32
        let encoder = RDEncoder::new(&[10.0f32, 20.0]);
        let alprd1 = encoder.encode(&PrimitiveArray::from_iter([10.0f32, 20.0, 30.0, 40.0, 50.0]));
        let alprd2 = encoder.encode(&PrimitiveArray::from_iter([5.0f32, 10.0, 15.0, 20.0, 25.0]));
        
        test_numeric(alprd1.into_array());
        test_numeric(alprd2.into_array());
        
        // Test with f64
        let encoder64 = RDEncoder::new(&[100.0f64, 200.0]);
        let alprd3 = encoder64.encode(&PrimitiveArray::from_iter([100.0f64, 200.0, 300.0, 400.0, 500.0]));
        let alprd4 = encoder64.encode(&PrimitiveArray::from_iter([50.0f64, 100.0, 150.0, 200.0, 250.0]));
        
        test_numeric(alprd3.into_array());
        test_numeric(alprd4.into_array());
    }
}
