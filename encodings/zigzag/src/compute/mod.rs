// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod cast;

use vortex_array::compute::{
    FilterKernel, FilterKernelAdapter, MaskKernel, MaskKernelAdapter, TakeKernel,
    TakeKernelAdapter, filter, mask, take,
};
use vortex_array::{Array, ArrayRef, IntoArray, register_kernel};
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::{ZigZagArray, ZigZagVTable};

impl FilterKernel for ZigZagVTable {
    fn filter(&self, array: &ZigZagArray, mask: &Mask) -> VortexResult<ArrayRef> {
        let encoded = filter(array.encoded(), mask)?;
        Ok(ZigZagArray::try_new(encoded)?.into_array())
    }
}

register_kernel!(FilterKernelAdapter(ZigZagVTable).lift());

impl TakeKernel for ZigZagVTable {
    fn take(&self, array: &ZigZagArray, indices: &dyn Array) -> VortexResult<ArrayRef> {
        let encoded = take(array.encoded(), indices)?;
        Ok(ZigZagArray::try_new(encoded)?.into_array())
    }
}

register_kernel!(TakeKernelAdapter(ZigZagVTable).lift());

impl MaskKernel for ZigZagVTable {
    fn mask(&self, array: &ZigZagArray, filter_mask: &Mask) -> VortexResult<ArrayRef> {
        let encoded = mask(array.encoded(), filter_mask)?;
        Ok(ZigZagArray::try_new(encoded)?.into_array())
    }
}

register_kernel!(MaskKernelAdapter(ZigZagVTable).lift());

pub(crate) trait ZigZagEncoded {
    type Int: zigzag::ZigZag;
}

impl ZigZagEncoded for u8 {
    type Int = i8;
}

impl ZigZagEncoded for u16 {
    type Int = i16;
}

impl ZigZagEncoded for u32 {
    type Int = i32;
}

impl ZigZagEncoded for u64 {
    type Int = i64;
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::arrays::{BooleanBuffer, PrimitiveArray};
    use vortex_array::compute::conformance::binary_numeric::test_binary_numeric_array;
    use vortex_array::compute::conformance::consistency::test_array_consistency;
    use vortex_array::compute::{filter, take};
    use vortex_array::validity::Validity;
    use vortex_array::{Array, ArrayRef, IntoArray, ToCanonical};
    use vortex_buffer::buffer;
    use vortex_dtype::Nullability;
    use vortex_scalar::Scalar;

    use crate::{ZigZagArray, ZigZagEncoding, zigzag_encode};

    #[test]
    pub fn nullable_scalar_at() {
        let zigzag = ZigZagEncoding
            .encode(
                &PrimitiveArray::new(buffer![-189, -160, 1], Validity::AllValid).to_canonical(),
                None,
            )
            .unwrap()
            .unwrap();
        assert_eq!(
            zigzag.scalar_at(1),
            Scalar::primitive(-160, Nullability::Nullable)
        );
    }

    #[test]
    fn take_zigzag() {
        let zigzag = ZigZagEncoding
            .encode(&buffer![-189, -160, 1].into_array().to_canonical(), None)
            .unwrap()
            .unwrap();

        let indices = buffer![0, 2].into_array();
        let actual = take(&zigzag, &indices).unwrap().to_primitive();
        let expected = ZigZagEncoding
            .encode(&buffer![-189, 1].into_array().to_canonical(), None)
            .unwrap()
            .unwrap()
            .to_primitive();
        assert_eq!(actual.as_slice::<i32>(), expected.as_slice::<i32>());
    }

    #[test]
    fn filter_zigzag() {
        let zigzag = ZigZagEncoding
            .encode(&buffer![-189, -160, 1].into_array().to_canonical(), None)
            .unwrap()
            .unwrap();
        let filter_mask = BooleanBuffer::from(vec![true, false, true]).into();
        let actual = filter(&zigzag, &filter_mask).unwrap().to_primitive();
        let expected = ZigZagEncoding
            .encode(&buffer![-189, 1].into_array().to_canonical(), None)
            .unwrap()
            .unwrap()
            .to_primitive();
        assert_eq!(actual.as_slice::<i32>(), expected.as_slice::<i32>());
    }

    #[test]
    fn test_filter_conformance() {
        use vortex_array::compute::conformance::filter::test_filter_conformance;

        // Test with i32 values
        let zigzag = ZigZagEncoding
            .encode(
                &buffer![-189i32, -160, 1, 42, -73]
                    .into_array()
                    .to_canonical(),
                None,
            )
            .unwrap()
            .unwrap();
        test_filter_conformance(zigzag.as_ref());

        // Test with i64 values
        let zigzag = ZigZagEncoding
            .encode(
                &buffer![1000i64, -2000, 3000, -4000, 5000]
                    .into_array()
                    .to_canonical(),
                None,
            )
            .unwrap()
            .unwrap();
        test_filter_conformance(zigzag.as_ref());

        // Test with nullable values
        let array =
            PrimitiveArray::from_option_iter([Some(-10i16), None, Some(20), Some(-30), None]);
        let zigzag = ZigZagEncoding
            .encode(&array.to_canonical(), None)
            .unwrap()
            .unwrap();
        test_filter_conformance(zigzag.as_ref());
    }

    #[test]
    fn test_mask_conformance() {
        use vortex_array::compute::conformance::mask::test_mask_conformance;

        // Test with i32 values
        let zigzag = ZigZagEncoding
            .encode(
                &buffer![-100i32, 200, -300, 400, -500]
                    .into_array()
                    .to_canonical(),
                None,
            )
            .unwrap()
            .unwrap();
        test_mask_conformance(zigzag.as_ref());

        // Test with i8 values
        let zigzag = ZigZagEncoding
            .encode(
                &buffer![-127i8, 0, 127, -1, 1].into_array().to_canonical(),
                None,
            )
            .unwrap()
            .unwrap();
        test_mask_conformance(zigzag.as_ref());
    }

    #[rstest]
    #[case(buffer![-189i32, -160, 1, 42, -73].into_array())]
    #[case(buffer![1000i64, -2000, 3000, -4000, 5000].into_array())]
    #[case(PrimitiveArray::from_option_iter([Some(-10i16), None, Some(20), Some(-30), None]).into_array())]
    #[case(buffer![42i32].into_array())]
    fn test_take_zigzag_conformance(#[case] array: ArrayRef) {
        use vortex_array::compute::conformance::take::test_take_conformance;

        let zigzag = ZigZagEncoding
            .encode(&array.to_canonical(), None)
            .unwrap()
            .unwrap();
        test_take_conformance(zigzag.as_ref());
    }

    #[rstest]
    // Basic ZigZag arrays
    #[case::zigzag_i8(zigzag_encode(PrimitiveArray::from_iter([-128i8, -1, 0, 1, 127])).unwrap())]
    #[case::zigzag_i16(zigzag_encode(PrimitiveArray::from_iter([-1000i16, -100, 0, 100, 1000])).unwrap())]
    #[case::zigzag_i32(zigzag_encode(PrimitiveArray::from_iter([-100000i32, -1000, 0, 1000, 100000])).unwrap())]
    #[case::zigzag_i64(zigzag_encode(PrimitiveArray::from_iter([-1000000i64, -10000, 0, 10000, 1000000])).unwrap())]
    // Nullable arrays
    #[case::zigzag_nullable_i32(zigzag_encode(PrimitiveArray::from_option_iter([Some(-100i32), None, Some(0), Some(100), None])).unwrap())]
    #[case::zigzag_nullable_i64(zigzag_encode(PrimitiveArray::from_option_iter([Some(-1000i64), None, Some(0), Some(1000), None])).unwrap())]
    // Edge cases
    #[case::zigzag_single(zigzag_encode(PrimitiveArray::from_iter([-42i32])).unwrap())]
    #[case::zigzag_alternating(zigzag_encode(PrimitiveArray::from_iter([-1i32, 1, -2, 2, -3, 3])).unwrap())]
    // Large arrays
    #[case::zigzag_large_i32(zigzag_encode(PrimitiveArray::from_iter(-500..500)).unwrap())]
    #[case::zigzag_large_i64(zigzag_encode(PrimitiveArray::from_iter((-1000..1000).map(|i| i as i64 * 100))).unwrap())]
    fn test_zigzag_consistency(#[case] array: ZigZagArray) {
        test_array_consistency(array.as_ref());
    }

    #[rstest]
    #[case::zigzag_i8_basic(zigzag_encode(PrimitiveArray::from_iter([-10i8, -5, 0, 5, 10])).unwrap())]
    #[case::zigzag_i16_basic(zigzag_encode(PrimitiveArray::from_iter([-100i16, -50, 0, 50, 100])).unwrap())]
    #[case::zigzag_i32_basic(zigzag_encode(PrimitiveArray::from_iter([-1000i32, -500, 0, 500, 1000])).unwrap())]
    #[case::zigzag_i64_basic(zigzag_encode(PrimitiveArray::from_iter([-10000i64, -5000, 0, 5000, 10000])).unwrap())]
    #[case::zigzag_i32_large(zigzag_encode(PrimitiveArray::from_iter((-50..50).map(|i| i * 10))).unwrap())]
    fn test_zigzag_binary_numeric(#[case] array: ZigZagArray) {
        test_binary_numeric_array(array.into_array());
    }
}
