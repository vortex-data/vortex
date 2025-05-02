use vortex_array::{Array, ArrayOperationsImpl, ArrayRef};
use vortex_error::VortexResult;

use crate::ALPRDArray;

impl ArrayOperationsImpl for ALPRDArray {
    fn _slice(&self, start: usize, stop: usize) -> VortexResult<ArrayRef> {
        let left_parts_exceptions = self
            .left_parts_patches()
            .map(|patches| patches.slice(start, stop))
            .transpose()?
            .flatten();

        Ok(ALPRDArray::try_new(
            self.dtype().clone(),
            self.left_parts().slice(start, stop)?,
            self.left_parts_dictionary().clone(),
            self.right_parts().slice(start, stop)?,
            self.right_bit_width(),
            left_parts_exceptions,
        )?
        .into_array())
    }
}

#[cfg(test)]
mod test {
    use rstest::rstest;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::{Array, ToCanonical};

    use crate::{ALPRDFloat, RDEncoder};

    #[rstest]
    #[case(0.1f32, 0.2f32, 3e25f32)]
    #[case(0.1f64, 0.2f64, 3e100f64)]
    fn test_slice<T: ALPRDFloat>(#[case] a: T, #[case] b: T, #[case] outlier: T) {
        let array = PrimitiveArray::from_iter([a, b, outlier]);
        let encoded = RDEncoder::new(&[a, b]).encode(&array);

        assert!(encoded.left_parts_patches().is_some());

        let decoded = encoded.slice(1, 3).unwrap().to_primitive().unwrap();

        assert_eq!(decoded.as_slice::<T>(), &[b, outlier]);
    }
}
