use vortex_array::compute::{scalar_at, ScalarAtFn};
use vortex_array::ArrayDType;
use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::alp_rd::array::ALPRDArray;
use crate::ALPRDEncoding;

impl ScalarAtFn<ALPRDArray> for ALPRDEncoding {
    fn scalar_at(&self, array: &ALPRDArray, index: usize) -> VortexResult<Scalar> {
        // The left value can either be a direct value, or an exception.
        // The exceptions array represents exception positions with non-null values.
        let maybe_patched_value = array
            .left_parts_patches()
            .map(|patches| patches.get_patched(index))
            .transpose()?
            .flatten();
        let left = match maybe_patched_value {
            Some(patched_value) => u16::try_from(patched_value)?,
            _ => {
                let left_code: u16 = scalar_at(array.left_parts(), index)?.try_into()?;
                array.left_parts_dict()[left_code as usize]
            }
        };

        // combine left and right values
        if array.is_f32() {
            let right: u32 = scalar_at(array.right_parts(), index)?.try_into()?;
            let packed = f32::from_bits((left as u32) << array.right_bit_width() | right);
            Ok(Scalar::primitive(packed, array.dtype().nullability()))
        } else {
            let right: u64 = scalar_at(array.right_parts(), index)?.try_into()?;
            let packed = f64::from_bits(((left as u64) << array.right_bit_width()) | right);
            Ok(Scalar::primitive(packed, array.dtype().nullability()))
        }
    }
}

#[cfg(test)]
mod test {
    use rstest::rstest;
    use vortex_array::array::PrimitiveArray;
    use vortex_array::compute::scalar_at;
    use vortex_dtype::Nullability;
    use vortex_scalar::Scalar;

    use crate::{ALPRDFloat, RDEncoder};

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
        assert_eq!(scalar_at(encoded.as_ref(), 0).unwrap(), a.into());
        assert_eq!(scalar_at(encoded.as_ref(), 1).unwrap(), b.into());

        // The right value hits the left_part_exceptions
        assert_eq!(scalar_at(encoded.as_ref(), 2).unwrap(), outlier.into());
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
            scalar_at(encoded.as_ref(), 0).unwrap(),
            Scalar::primitive(a, Nullability::Nullable)
        );
        assert_eq!(
            scalar_at(encoded.as_ref(), 1).unwrap(),
            Scalar::primitive(b, Nullability::Nullable)
        );

        // The right value hits the left_part_exceptions
        assert_eq!(
            scalar_at(encoded.as_ref(), 2).unwrap(),
            Scalar::primitive(outlier, Nullability::Nullable)
        );
    }
}
