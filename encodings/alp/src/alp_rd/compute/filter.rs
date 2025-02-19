use vortex_array::compute::{filter, FilterFn};
use vortex_array::{Array, IntoArray};
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::{ALPRDArray, ALPRDEncoding};

impl FilterFn<ALPRDArray> for ALPRDEncoding {
    fn filter(&self, array: &ALPRDArray, mask: &Mask) -> VortexResult<Array> {
        let left_parts_exceptions = array
            .left_parts_patches()
            .map(|patches| patches.filter(mask))
            .transpose()?
            .flatten();

        Ok(ALPRDArray::try_new(
            array.dtype().clone(),
            filter(&array.left_parts(), mask)?,
            array.left_parts_dict(),
            filter(&array.right_parts(), mask)?,
            array.right_bit_width(),
            left_parts_exceptions,
        )?
        .into_array())
    }
}

#[cfg(test)]
mod test {
    use rstest::rstest;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::compute::filter;
    use vortex_array::validity::Validity;
    use vortex_array::IntoArrayVariant;
    use vortex_buffer::buffer;
    use vortex_mask::Mask;

    use crate::{ALPRDFloat, RDEncoder};

    #[rstest]
    #[case(0.1f32, 0.2f32, 3e25f32)]
    #[case(0.1f64, 0.2f64, 3e100f64)]
    fn test_filter<T: ALPRDFloat>(#[case] a: T, #[case] b: T, #[case] outlier: T) {
        let array = PrimitiveArray::new(buffer![a, b, outlier], Validity::NonNullable);
        let encoded = RDEncoder::new(&[a, b]).encode(&array);

        // Make sure that we're testing the exception pathway.
        assert!(encoded.left_parts_patches().is_some());

        // The first two values need no patching
        let filtered = filter(encoded.as_ref(), &Mask::from_iter([true, false, true]))
            .unwrap()
            .into_primitive()
            .unwrap();
        assert_eq!(filtered.as_slice::<T>(), &[a, outlier]);
    }
}
