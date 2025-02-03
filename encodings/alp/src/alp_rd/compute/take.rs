use vortex_array::compute::{take, TakeFn};
use vortex_array::{Array, IntoArray};
use vortex_error::VortexResult;

use crate::{ALPRDArray, ALPRDEncoding};

impl TakeFn<ALPRDArray> for ALPRDEncoding {
    fn take(&self, array: &ALPRDArray, indices: &Array) -> VortexResult<Array> {
        let left_parts_exceptions = array
            .left_parts_patches()
            .map(|patches| patches.take(indices))
            .transpose()?
            .flatten();

        Ok(ALPRDArray::try_new(
            array.dtype().clone(),
            take(array.left_parts(), indices)?,
            array.left_parts_dict(),
            take(array.right_parts(), indices)?,
            array.right_bit_width(),
            left_parts_exceptions,
        )?
        .into_array())
    }
}

#[cfg(test)]
mod test {
    use rstest::rstest;
    use vortex_array::array::PrimitiveArray;
    use vortex_array::compute::take;
    use vortex_array::IntoArrayVariant;

    use crate::{ALPRDFloat, RDEncoder};

    #[rstest]
    #[case(0.1f32, 0.2f32, 3e25f32)]
    #[case(0.1f64, 0.2f64, 3e100f64)]
    fn test_take<T: ALPRDFloat>(#[case] a: T, #[case] b: T, #[case] outlier: T) {
        let array = PrimitiveArray::from_iter([a, b, outlier]);
        let encoded = RDEncoder::new(&[a, b]).encode(&array);

        assert!(encoded.left_parts_patches().is_some());
        assert!(encoded
            .left_parts_patches()
            .unwrap()
            .dtype()
            .is_unsigned_int());

        let taken = take(encoded.as_ref(), PrimitiveArray::from_iter([0, 2]).as_ref())
            .unwrap()
            .into_primitive()
            .unwrap();

        assert_eq!(taken.as_slice::<T>(), &[a, outlier]);
    }
}
