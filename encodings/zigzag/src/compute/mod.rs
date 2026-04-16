// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod cast;

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::dict::TakeExecute;
use vortex_array::arrays::filter::FilterReduce;
use vortex_array::arrays::scalar_fn::ScalarFnFactoryExt;
use vortex_array::scalar_fn::EmptyOptions;
use vortex_array::scalar_fn::fns::mask::Mask as MaskExpr;
use vortex_array::scalar_fn::fns::mask::MaskReduce;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::ZigZag;
use crate::array::ZigZagArrayExt;

impl FilterReduce for ZigZag {
    fn filter(array: ArrayView<'_, Self>, mask: &Mask) -> VortexResult<Option<ArrayRef>> {
        let encoded = array.encoded().filter(mask.clone())?;
        Ok(Some(ZigZag::try_new(encoded)?.into_array()))
    }
}

impl TakeExecute for ZigZag {
    fn take(
        array: ArrayView<'_, Self>,
        indices: &ArrayRef,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let encoded = array.encoded().take(indices.clone())?;
        Ok(Some(ZigZag::try_new(encoded)?.into_array()))
    }
}

impl MaskReduce for ZigZag {
    fn mask(array: ArrayView<'_, Self>, mask: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        let masked_encoded = MaskExpr.try_new_array(
            array.encoded().len(),
            EmptyOptions,
            [array.encoded().clone(), mask.clone()],
        )?;
        Ok(Some(ZigZag::try_new(masked_encoded)?.into_array()))
    }
}

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
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::LEGACY_SESSION;
    #[expect(deprecated)]
    use vortex_array::ToCanonical;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::compute::conformance::binary_numeric::test_binary_numeric_array;
    use vortex_array::compute::conformance::consistency::test_array_consistency;
    use vortex_array::dtype::Nullability;
    use vortex_array::scalar::Scalar;
    use vortex_array::validity::Validity;
    use vortex_buffer::BitBuffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::ZigZagArray;
    use crate::zigzag_encode;

    #[test]
    pub fn nullable_scalar_at() -> VortexResult<()> {
        let zigzag = zigzag_encode(
            PrimitiveArray::new(buffer![-189, -160, 1], Validity::AllValid).as_view(),
        )?;
        assert_eq!(
            zigzag.execute_scalar(1, &mut LEGACY_SESSION.create_execution_ctx())?,
            Scalar::primitive(-160, Nullability::Nullable)
        );
        Ok(())
    }

    #[test]
    fn take_zigzag() -> VortexResult<()> {
        let zigzag = zigzag_encode(
            PrimitiveArray::new(buffer![-189, -160, 1], Validity::AllValid).as_view(),
        )?;

        let indices = buffer![0, 2].into_array();
        let actual = zigzag.take(indices).unwrap();
        let expected =
            zigzag_encode(PrimitiveArray::new(buffer![-189, 1], Validity::AllValid).as_view())?
                .into_array();
        assert_arrays_eq!(actual, expected);
        Ok(())
    }

    #[test]
    fn filter_zigzag() -> VortexResult<()> {
        let zigzag = zigzag_encode(
            PrimitiveArray::new(buffer![-189, -160, 1], Validity::AllValid).as_view(),
        )?;

        let filter_mask = BitBuffer::from(vec![true, false, true]).into();
        let actual = zigzag.filter(filter_mask).unwrap();
        let expected =
            zigzag_encode(PrimitiveArray::new(buffer![-189, 1], Validity::AllValid).as_view())?
                .into_array();
        assert_arrays_eq!(actual, expected);
        Ok(())
    }

    #[test]
    fn test_filter_conformance() -> VortexResult<()> {
        use vortex_array::compute::conformance::filter::test_filter_conformance;

        // Test with i32 values
        let zigzag = zigzag_encode(
            PrimitiveArray::new(buffer![-189i32, -160, 1, 42, -73], Validity::AllValid).as_view(),
        )?;
        test_filter_conformance(&zigzag.into_array());

        // Test with i64 values
        let zigzag = zigzag_encode(
            PrimitiveArray::new(
                buffer![1000i64, -2000, 3000, -4000, 5000],
                Validity::AllValid,
            )
            .as_view(),
        )?;
        test_filter_conformance(&zigzag.into_array());

        // Test with nullable values
        let array =
            PrimitiveArray::from_option_iter([Some(-10i16), None, Some(20), Some(-30), None]);
        let zigzag = zigzag_encode(array.as_view())?;
        test_filter_conformance(&zigzag.into_array());
        Ok(())
    }

    #[test]
    fn test_mask_conformance() -> VortexResult<()> {
        use vortex_array::compute::conformance::mask::test_mask_conformance;

        // Test with i32 values
        let zigzag = zigzag_encode(
            PrimitiveArray::new(buffer![-100i32, 200, -300, 400, -500], Validity::AllValid)
                .as_view(),
        )?;
        test_mask_conformance(&zigzag.into_array());

        // Test with i8 values
        let zigzag = zigzag_encode(
            PrimitiveArray::new(buffer![-127i8, 0, 127, -1, 1], Validity::AllValid).as_view(),
        )?;
        test_mask_conformance(&zigzag.into_array());
        Ok(())
    }

    #[rstest]
    #[case(buffer![-189i32, -160, 1, 42, -73].into_array())]
    #[case(buffer![1000i64, -2000, 3000, -4000, 5000].into_array())]
    #[case(PrimitiveArray::from_option_iter([Some(-10i16), None, Some(20), Some(-30), None]).into_array()
    )]
    #[case(buffer![42i32].into_array())]
    fn test_take_zigzag_conformance(#[case] array: ArrayRef) -> VortexResult<()> {
        use vortex_array::compute::conformance::take::test_take_conformance;

        #[expect(deprecated)]
        let array_primitive = array.to_primitive();
        let zigzag = zigzag_encode(array_primitive.as_view())?;
        test_take_conformance(&zigzag.into_array());
        Ok(())
    }

    #[rstest]
    // Basic ZigZag arrays
    #[case::zigzag_i8(zigzag_encode(PrimitiveArray::from_iter([-128i8, -1, 0, 1, 127]).as_view()).unwrap())]
    #[case::zigzag_i16(zigzag_encode(PrimitiveArray::from_iter([-1000i16, -100, 0, 100, 1000]).as_view()).unwrap())]
    #[case::zigzag_i32(zigzag_encode(PrimitiveArray::from_iter([-100000i32, -1000, 0, 1000, 100000]).as_view()).unwrap())]
    #[case::zigzag_i64(zigzag_encode(PrimitiveArray::from_iter([-1000000i64, -10000, 0, 10000, 1000000]).as_view()).unwrap())]
    // Nullable arrays
    #[case::zigzag_nullable_i32(zigzag_encode(PrimitiveArray::from_option_iter([Some(-100i32), None, Some(0), Some(100), None]).as_view()).unwrap())]
    #[case::zigzag_nullable_i64(zigzag_encode(PrimitiveArray::from_option_iter([Some(-1000i64), None, Some(0), Some(1000), None]).as_view()).unwrap())]
    // Edge cases
    #[case::zigzag_single(zigzag_encode(PrimitiveArray::from_iter([-42i32]).as_view()).unwrap())]
    #[case::zigzag_alternating(zigzag_encode(PrimitiveArray::from_iter([-1i32, 1, -2, 2, -3, 3]).as_view()).unwrap())]
    // Large arrays
    #[case::zigzag_large_i32(zigzag_encode(PrimitiveArray::from_iter(-500..500).as_view()).unwrap())]
    #[case::zigzag_large_i64(zigzag_encode(PrimitiveArray::from_iter((-1000..1000).map(|i| i as i64 * 100)).as_view()).unwrap())]
    fn test_zigzag_consistency(#[case] array: ZigZagArray) {
        test_array_consistency(&array.into_array());
    }

    #[rstest]
    #[case::zigzag_i8_basic(zigzag_encode(PrimitiveArray::from_iter([-10i8, -5, 0, 5, 10]).as_view()).unwrap())]
    #[case::zigzag_i16_basic(zigzag_encode(PrimitiveArray::from_iter([-100i16, -50, 0, 50, 100]).as_view()).unwrap())]
    #[case::zigzag_i32_basic(zigzag_encode(PrimitiveArray::from_iter([-1000i32, -500, 0, 500, 1000]).as_view()).unwrap())]
    #[case::zigzag_i64_basic(zigzag_encode(PrimitiveArray::from_iter([-10000i64, -5000, 0, 5000, 10000]).as_view()).unwrap())]
    #[case::zigzag_i32_large(zigzag_encode(PrimitiveArray::from_iter((-50..50).map(|i| i * 10)).as_view()).unwrap())]
    fn test_zigzag_binary_numeric(#[case] array: ZigZagArray) {
        test_binary_numeric_array(array.into_array());
    }
}
