// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_array::vtable::OperationsVTable;
use vortex_array::{Array, ArrayRef, IntoArray};
use vortex_error::VortexExpect;
use vortex_scalar::Scalar;

use crate::{ALPRDArray, ALPRDVTable};

impl OperationsVTable<ALPRDVTable> for ALPRDVTable {
    fn slice(array: &ALPRDArray, range: Range<usize>) -> ArrayRef {
        let left_parts_exceptions = array
            .left_parts_patches()
            .and_then(|patches| patches.slice(range.clone()));

        // SAFETY: slicing components does not change the encoded values
        unsafe {
            ALPRDArray::new_unchecked(
                array.dtype().clone(),
                array.left_parts().slice(range.clone()),
                array.left_parts_dictionary().clone(),
                array.right_parts().slice(range),
                array.right_bit_width(),
                left_parts_exceptions,
            )
            .into_array()
        }
    }

    fn scalar_at(array: &ALPRDArray, index: usize) -> Scalar {
        // The left value can either be a direct value, or an exception.
        // The exceptions array represents exception positions with non-null values.
        let maybe_patched_value = array
            .left_parts_patches()
            .and_then(|patches| patches.get_patched(index));
        let left = match maybe_patched_value {
            Some(patched_value) => patched_value
                .as_primitive()
                .as_::<u16>()
                .vortex_expect("patched values must be non-null"),
            _ => {
                let left_code: u16 = array
                    .left_parts()
                    .scalar_at(index)
                    .as_primitive()
                    .as_::<u16>()
                    .vortex_expect("left_code must be non-null");
                array.left_parts_dictionary()[left_code as usize]
            }
        };

        // combine left and right values
        if array.is_f32() {
            let right: u32 = array
                .right_parts()
                .scalar_at(index)
                .as_primitive()
                .as_::<u32>()
                .vortex_expect("non-null");
            let packed = f32::from_bits((left as u32) << array.right_bit_width() | right);
            Scalar::primitive(packed, array.dtype().nullability())
        } else {
            let right: u64 = array
                .right_parts()
                .scalar_at(index)
                .as_primitive()
                .as_::<u64>()
                .vortex_expect("non-null");
            let packed = f64::from_bits(((left as u64) << array.right_bit_width()) | right);
            Scalar::primitive(packed, array.dtype().nullability())
        }
    }
}

#[cfg(test)]
mod test {
    use rstest::rstest;
    use vortex_array::ToCanonical;
    use vortex_array::arrays::PrimitiveArray;
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

        let decoded = encoded.slice(1..3).to_primitive();

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
        assert_eq!(encoded.scalar_at(0), a.into());
        assert_eq!(encoded.scalar_at(1), b.into());

        // The right value hits the left_part_exceptions
        assert_eq!(encoded.scalar_at(2), outlier.into());
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
            encoded.scalar_at(0),
            Scalar::primitive(a, Nullability::Nullable)
        );
        assert_eq!(
            encoded.scalar_at(1),
            Scalar::primitive(b, Nullability::Nullable)
        );

        // The right value hits the left_part_exceptions
        assert_eq!(
            encoded.scalar_at(2),
            Scalar::primitive(outlier, Nullability::Nullable)
        );
    }
}
