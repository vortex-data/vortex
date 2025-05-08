use vortex_array::compute::{MaskKernel, MaskKernelAdapter, mask};
use vortex_array::{Array, ArrayRef, register_kernel};
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::{ALPRDArray, ALPRDEncoding};

impl MaskKernel for ALPRDEncoding {
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

register_kernel!(MaskKernelAdapter(ALPRDEncoding).lift());

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::Array;
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
}
