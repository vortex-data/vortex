use vortex_array::compute::{take, TakeFn, TakeOptions};
use vortex_array::{ArrayDType, ArrayData, IntoArrayData};
use vortex_error::VortexResult;

use crate::{ALPRDArray, ALPRDEncoding};

impl TakeFn<ALPRDArray> for ALPRDEncoding {
    fn take(
        &self,
        array: &ALPRDArray,
        indices: &ArrayData,
        options: TakeOptions,
    ) -> VortexResult<ArrayData> {
        let left_parts_exceptions = array
            .left_parts_exceptions()
            .map(|array| take(&array, indices, options))
            .transpose()?;

        Ok(ALPRDArray::try_new(
            array.dtype().clone(),
            take(array.left_parts(), indices, options)?,
            array.left_parts_dict(),
            take(array.right_parts(), indices, options)?,
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
    use vortex_array::compute::{take, TakeOptions};
    use vortex_array::IntoArrayVariant;

    use crate::{ALPRDFloat, RDEncoder};

    #[rstest]
    #[case(0.1f32, 0.2f32, 3e25f32)]
    #[case(0.1f64, 0.2f64, 3e100f64)]
    fn test_take<T: ALPRDFloat>(#[case] a: T, #[case] b: T, #[case] outlier: T) {
        let array = PrimitiveArray::from(vec![a, b, outlier]);
        let encoded = RDEncoder::new(&[a, b]).encode(&array);

        assert!(encoded.left_parts_exceptions().is_some());

        let taken = take(
            encoded.as_ref(),
            PrimitiveArray::from(vec![0, 2]).as_ref(),
            TakeOptions::default(),
        )
        .unwrap()
        .into_primitive()
        .unwrap();

        assert_eq!(taken.maybe_null_slice::<T>(), &[a, outlier]);
    }
}
