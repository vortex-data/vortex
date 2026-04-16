// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::dtype::PType;
use vortex_array::scalar::Scalar;
use vortex_array::vtable::OperationsVTable;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::ALPRD;
use crate::ALPRDArrayExt;

impl OperationsVTable<ALPRD> for ALPRD {
    fn scalar_at(
        array: ArrayView<'_, ALPRD>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        // The left value can either be a direct value, or an exception.
        // The exceptions array represents exception positions with non-null values.
        let maybe_patched_value = match array.left_parts_patches() {
            Some(patches) => patches.get_patched(index)?,
            None => None,
        };
        let left = match maybe_patched_value {
            Some(patched_value) => patched_value
                .as_primitive()
                .as_::<u16>()
                .vortex_expect("patched values must be non-null"),
            _ => {
                let left_code: u16 = array
                    .left_parts()
                    .execute_scalar(index, ctx)?
                    .as_primitive()
                    .as_::<u16>()
                    .vortex_expect("left_code must be non-null");
                array.left_parts_dictionary()[left_code as usize]
            }
        };

        // combine left and right values
        Ok(if array.dtype().as_ptype() == PType::F32 {
            let right: u32 = array
                .right_parts()
                .execute_scalar(index, ctx)?
                .as_primitive()
                .as_::<u32>()
                .vortex_expect("non-null");
            let packed = f32::from_bits((left as u32) << array.right_bit_width() | right);
            Scalar::primitive(packed, array.dtype().nullability())
        } else {
            let right: u64 = array
                .right_parts()
                .execute_scalar(index, ctx)?
                .as_primitive()
                .as_::<u64>()
                .vortex_expect("non-null");
            let packed = f64::from_bits(((left as u64) << array.right_bit_width()) | right);
            Scalar::primitive(packed, array.dtype().nullability())
        })
    }
}

#[cfg(test)]
mod test {
    use rstest::rstest;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::scalar::Scalar;

    use crate::ALPRDArrayExt;
    use crate::ALPRDFloat;
    use crate::RDEncoder;

    #[rstest]
    #[case(0.1f32, 0.2f32, 3e25f32)]
    #[case(0.1f64, 0.2f64, 3e100f64)]
    fn test_slice<T: ALPRDFloat>(#[case] a: T, #[case] b: T, #[case] outlier: T) {
        let array = PrimitiveArray::from_iter([a, b, outlier]);
        let encoded = RDEncoder::new(&[a, b]).encode(array.as_view());

        assert!(encoded.left_parts_patches().is_some());
        assert_arrays_eq!(encoded, array);
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
        let encoded = RDEncoder::new(&[a, b]).encode(array.as_view());
        assert!(encoded.left_parts_patches().is_some());
        assert_arrays_eq!(encoded, array);
    }

    #[test]
    fn nullable_scalar_at() {
        let a = 0.1f64;
        let b = 0.2f64;
        let outlier = 3e100f64;
        let array = PrimitiveArray::from_option_iter([Some(a), Some(b), Some(outlier)]);
        let encoded = RDEncoder::new(&[a, b]).encode(array.as_view());
        assert!(encoded.left_parts_patches().is_some());
        assert_arrays_eq!(
            encoded,
            PrimitiveArray::from_option_iter([Some(a), Some(b), Some(outlier)])
        );
    }
}
