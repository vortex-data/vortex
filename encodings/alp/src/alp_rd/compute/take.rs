use vortex_array::compute::{fill_null, take, TakeFn};
use vortex_array::{Array, IntoArray};
use vortex_error::VortexResult;
use vortex_scalar::{Scalar, ScalarValue};

use crate::{ALPRDArray, ALPRDEncoding};

impl TakeFn<ALPRDArray> for ALPRDEncoding {
    fn take(&self, array: &ALPRDArray, indices: &Array) -> VortexResult<Array> {
        let taken_left_parts = take(array.left_parts(), indices)?;
        let left_parts_exceptions = array
            .left_parts_patches()
            .map(|patches| patches.take(indices))
            .transpose()?
            .flatten()
            .map(|p| {
                let values_dtype = p
                    .values()
                    .dtype()
                    .with_nullability(taken_left_parts.dtype().nullability());
                p.cast_values(&values_dtype)
            })
            .transpose()?;

        let right_parts = fill_null(
            take(array.right_parts(), indices)?,
            Scalar::new(array.right_parts().dtype().clone(), ScalarValue::from(0)),
        )?;
        Ok(ALPRDArray::try_new(
            if taken_left_parts.dtype().is_nullable() {
                array.dtype().as_nullable()
            } else {
                array.dtype().clone()
            },
            taken_left_parts,
            array.left_parts_dict(),
            right_parts,
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

    #[rstest]
    #[case(0.1f32, 0.2f32, 3e25f32)]
    #[case(0.1f64, 0.2f64, 3e100f64)]
    fn take_with_nulls<T: ALPRDFloat>(#[case] a: T, #[case] b: T, #[case] outlier: T) {
        let array = PrimitiveArray::from_iter([a, b, outlier]);
        let encoded = RDEncoder::new(&[a, b]).encode(&array);

        assert!(encoded.left_parts_patches().is_some());
        assert!(encoded
            .left_parts_patches()
            .unwrap()
            .dtype()
            .is_unsigned_int());

        let taken = take(
            encoded.as_ref(),
            PrimitiveArray::from_option_iter([Some(0), Some(2), None]).as_ref(),
        )
        .unwrap()
        .into_primitive()
        .unwrap();

        assert_eq!(taken.as_slice::<T>()[0], a);
        assert_eq!(taken.as_slice::<T>()[1], outlier);
        assert!(!taken.validity_mask().unwrap().value(2));
    }
}
