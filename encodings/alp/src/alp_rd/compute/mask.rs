use vortex_array::compute::{mask, MaskFn};
use vortex_array::{Array, IntoArray};
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::{ALPRDArray, ALPRDEncoding};

impl MaskFn<ALPRDArray> for ALPRDEncoding {
    fn mask(&self, array: &ALPRDArray, filter_mask: Mask) -> VortexResult<Array> {
        Ok(ALPRDArray::try_new(
            array.dtype().as_nullable(),
            mask(&array.left_parts(), filter_mask)?,
            array.left_parts_dict(),
            array.right_parts(),
            array.right_bit_width(),
            array.left_parts_patches(),
        )?
        .into_array())
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::array::PrimitiveArray;
    use vortex_array::compute::test_harness::test_mask;
    use vortex_array::IntoArray as _;

    use crate::{ALPRDFloat, RDEncoder};

    #[rstest]
    #[case(0.1f32, 0.2f32, 3e25f32)]
    #[case(0.1f64, 0.2f64, 3e100f64)]
    fn test_mask_simple<T: ALPRDFloat>(#[case] a: T, #[case] b: T, #[case] outlier: T) {
        test_mask(
            RDEncoder::new(&[a, b])
                .encode(&PrimitiveArray::from_iter([a, b, outlier, b, outlier]))
                .into_array(),
        );
    }

    #[rstest]
    #[case(0.1f32, 3e25f32)]
    #[case(0.5f64, 1e100f64)]
    fn test_mask_with_nulls<T: ALPRDFloat>(#[case] a: T, #[case] outlier: T) {
        test_mask(
            RDEncoder::new(&[a])
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
