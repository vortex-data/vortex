use vortex_array::compute::{filter, FilterFn};
use vortex_array::{ArrayDType, ArrayData, IntoArrayData};
use vortex_error::VortexResult;

use crate::ALPRDArray;

impl FilterFn for ALPRDArray {
    fn filter(&self, predicate: &ArrayData) -> VortexResult<ArrayData> {
        let left_parts_exceptions = self
            .left_parts_exceptions()
            .map(|array| filter(&array, predicate))
            .transpose()?;

        Ok(ALPRDArray::try_new(
            self.dtype().clone(),
            filter(self.left_parts(), predicate)?,
            self.left_parts_dict(),
            filter(self.right_parts(), predicate)?,
            self.right_bit_width(),
            left_parts_exceptions,
        )?
        .into_array())
    }
}

#[cfg(test)]
mod test {
    use rstest::rstest;
    use vortex_array::array::{BoolArray, PrimitiveArray};
    use vortex_array::compute::filter;
    use vortex_array::IntoArrayVariant;

    use crate::{ALPRDFloat, RDEncoder};

    #[rstest]
    #[case(0.1f32, 0.2f32, 3e25f32)]
    #[case(0.1f64, 0.2f64, 3e100f64)]
    fn test_filter<T: ALPRDFloat>(#[case] a: T, #[case] b: T, #[case] outlier: T) {
        let array = PrimitiveArray::from(vec![a, b, outlier]);
        let encoded = RDEncoder::new(&[a, b]).encode(&array);

        // Make sure that we're testing the exception pathway.
        assert!(encoded.left_parts_exceptions().is_some());

        // The first two values need no patching
        let filtered = filter(encoded.as_ref(), BoolArray::from(vec![true, false, true]))
            .unwrap()
            .into_primitive()
            .unwrap();
        assert_eq!(filtered.maybe_null_slice::<T>(), &[a, outlier]);
    }
}
