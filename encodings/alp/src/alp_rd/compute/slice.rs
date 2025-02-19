use vortex_array::compute::{slice, SliceFn};
use vortex_array::{Array, IntoArray};
use vortex_error::VortexResult;

use crate::{ALPRDArray, ALPRDEncoding};

impl SliceFn<ALPRDArray> for ALPRDEncoding {
    fn slice(&self, array: &ALPRDArray, start: usize, stop: usize) -> VortexResult<Array> {
        let left_parts_exceptions = array
            .left_parts_patches()
            .map(|patches| patches.slice(start, stop))
            .transpose()?
            .flatten();

        Ok(ALPRDArray::try_new(
            array.dtype().clone(),
            slice(array.left_parts(), start, stop)?,
            array.left_parts_dict(),
            slice(array.right_parts(), start, stop)?,
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
    use vortex_array::compute::slice;
    use vortex_array::IntoArrayVariant;

    use crate::{ALPRDFloat, RDEncoder};

    #[rstest]
    #[case(0.1f32, 0.2f32, 3e25f32)]
    #[case(0.1f64, 0.2f64, 3e100f64)]
    fn test_slice<T: ALPRDFloat>(#[case] a: T, #[case] b: T, #[case] outlier: T) {
        let array = PrimitiveArray::from_iter([a, b, outlier]);
        let encoded = RDEncoder::new(&[a, b]).encode(&array);

        assert!(encoded.left_parts_patches().is_some());

        let decoded = slice(encoded.as_ref(), 1, 3)
            .unwrap()
            .into_primitive()
            .unwrap();

        assert_eq!(decoded.as_slice::<T>(), &[b, outlier]);
    }
}
