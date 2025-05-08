use vortex_array::{Array, ArrayOperationsImpl, ArrayRef};
use vortex_error::VortexResult;
use vortex_scalar::Scalar;

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

    fn _scalar_at(&self, index: usize) -> VortexResult<Scalar> {
        // The left value can either be a direct value, or an exception.
        // The exceptions array represents exception positions with non-null values.
        let maybe_patched_value = self
            .left_parts_patches()
            .map(|patches| patches.get_patched(index))
            .transpose()?
            .flatten();
        let left = match maybe_patched_value {
            Some(patched_value) => u16::try_from(patched_value)?,
            _ => {
                let left_code: u16 = self.left_parts().scalar_at(index)?.try_into()?;
                self.left_parts_dictionary()[left_code as usize]
            }
        };

        // combine left and right values
        if self.is_f32() {
            let right: u32 = self.right_parts().scalar_at(index)?.try_into()?;
            let packed = f32::from_bits((left as u32) << self.right_bit_width() | right);
            Ok(Scalar::primitive(packed, self.dtype().nullability()))
        } else {
            let right: u64 = self.right_parts().scalar_at(index)?.try_into()?;
            let packed = f64::from_bits(((left as u64) << self.right_bit_width()) | right);
            Ok(Scalar::primitive(packed, self.dtype().nullability()))
        }
    }
}

#[cfg(test)]
mod test {
    use rstest::rstest;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::{Array, ToCanonical};
    use vortex_dtype::Nullability;
    use vortex_scalar::Scalar;

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

    #[rstest]
    #[case(0.1f32, 0.2f32, 3e25f32)]
    #[case(0.1f64, 0.2f64, 3e100f64)]
    fn test_scalar_at<T: ALPRDFloat + Into<Scalar>>(
        #[case] a: T,
        #[case] b: T,
        #[case] outlier: T,
    ) {
        let array = PrimitiveArray::from_iter([a, b, outlier]);
        let encoded = RDEncoder::new(&[a, b]).encode(&array);

        // Make sure that we're testing the exception pathway.
        assert!(encoded.left_parts_patches().is_some());

        // The first two values need no patching
        assert_eq!(encoded.scalar_at(0).unwrap(), a.into());
        assert_eq!(encoded.scalar_at(1).unwrap(), b.into());

        // The right value hits the left_part_exceptions
        assert_eq!(encoded.scalar_at(2).unwrap(), outlier.into());
    }

    #[test]
    fn nullable_scalar_at() {
        let a = 0.1f64;
        let b = 0.2f64;
        let outlier = 3e100f64;
        let array = PrimitiveArray::from_option_iter([Some(a), Some(b), Some(outlier)]);
        let encoded = RDEncoder::new(&[a, b]).encode(&array);

        // Make sure that we're testing the exception pathway.
        assert!(encoded.left_parts_patches().is_some());

        // The first two values need no patching
        assert_eq!(
            encoded.scalar_at(0).unwrap(),
            Scalar::primitive(a, Nullability::Nullable)
        );
        assert_eq!(
            encoded.scalar_at(1).unwrap(),
            Scalar::primitive(b, Nullability::Nullable)
        );

        // The right value hits the left_part_exceptions
        assert_eq!(
            encoded.scalar_at(2).unwrap(),
            Scalar::primitive(outlier, Nullability::Nullable)
        );
    }
}
