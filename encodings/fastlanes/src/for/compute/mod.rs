// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod cast;
mod compare;
pub(crate) mod is_constant;
pub(crate) mod is_sorted;

use vortex_array::ArrayRef;
use vortex_array::DynArray;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::dict::TakeExecute;
use vortex_array::arrays::filter::FilterReduce;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::FoR;
use crate::FoRArray;

impl TakeExecute for FoR {
    fn take(
        array: &FoRArray,
        indices: &ArrayRef,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        Ok(Some(
            FoRArray::try_new(
                array.encoded().take(indices.to_array())?,
                array.reference_scalar().clone(),
            )?
            .into_array(),
        ))
    }
}

impl FilterReduce for FoR {
    fn filter(array: &FoRArray, mask: &Mask) -> VortexResult<Option<ArrayRef>> {
        FoRArray::try_new(
            array.encoded().filter(mask.clone())?,
            array.reference_scalar().clone(),
        )
        .map(|a| Some(a.into_array()))
    }
}

#[cfg(test)]
mod test {
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::compute::conformance::filter::test_filter_conformance;
    use vortex_array::scalar::Scalar;
    use vortex_buffer::buffer;

    use crate::FoRArray;

    #[test]
    fn test_filter_for_array() {
        // Test with i32 values
        let values = buffer![100i32, 101, 102, 103, 104].into_array();
        let reference = Scalar::from(100i32);
        let for_array = FoRArray::try_new(values, reference).unwrap();
        test_filter_conformance(&for_array.into_array());

        // Test with u64 values
        let values = buffer![1000u64, 1001, 1002, 1003, 1004].into_array();
        let reference = Scalar::from(1000u64);
        let for_array = FoRArray::try_new(values, reference).unwrap();
        test_filter_conformance(&for_array.into_array());

        // Test with nullable values
        let values =
            PrimitiveArray::from_option_iter([Some(50i16), None, Some(52), Some(53), None]);
        let reference = Scalar::from(50i16);
        let for_array = FoRArray::try_new(values.into_array(), reference).unwrap();
        test_filter_conformance(&for_array.into_array());
    }

    #[rstest]
    #[case(FoRArray::try_new(buffer![100i32, 101, 102, 103, 104].into_array(), Scalar::from(100i32)).unwrap())]
    #[case(FoRArray::try_new(buffer![1000u64, 1001, 1002, 1003, 1004].into_array(), Scalar::from(1000u64)).unwrap())]
    #[case(FoRArray::try_new(
        PrimitiveArray::from_option_iter([Some(50i16), None, Some(52), Some(53), None]).into_array(),
        Scalar::from(50i16)
    ).unwrap())]
    #[case(FoRArray::try_new(buffer![-100i32, -99, -98, -97, -96].into_array(), Scalar::from(-100i32)).unwrap())]
    #[case(FoRArray::try_new(buffer![42i64].into_array(), Scalar::from(40i64)).unwrap())]
    fn test_take_for_conformance(#[case] for_array: FoRArray) {
        use vortex_array::compute::conformance::take::test_take_conformance;
        test_take_conformance(&for_array.into_array());
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::compute::conformance::binary_numeric::test_binary_numeric_array;
    use vortex_array::compute::conformance::consistency::test_array_consistency;
    use vortex_array::scalar::Scalar;
    use vortex_buffer::buffer;

    use crate::FoRArray;

    #[rstest]
    // Basic FoR arrays
    #[case::for_i32(FoRArray::try_new(
        buffer![100i32, 101, 102, 103, 104].into_array(),
        Scalar::from(100i32)
    ).unwrap())]
    #[case::for_u64(FoRArray::try_new(
        buffer![1000u64, 1001, 1002, 1003, 1004].into_array(),
        Scalar::from(1000u64)
    ).unwrap())]
    // Nullable arrays
    #[case::for_nullable_i16(FoRArray::try_new(
        PrimitiveArray::from_option_iter([Some(50i16), None, Some(52), Some(53), None]).into_array(),
        Scalar::from(50i16)
    ).unwrap())]
    #[case::for_nullable_i32(FoRArray::try_new(
        PrimitiveArray::from_option_iter([Some(200i32), None, Some(202), Some(203), None]).into_array(),
        Scalar::from(200i32)
    ).unwrap())]
    // Negative values
    #[case::for_negative(FoRArray::try_new(
        buffer![-100i32, -99, -98, -97, -96].into_array(),
        Scalar::from(-100i32)
    ).unwrap())]
    // Edge cases
    #[case::for_single(FoRArray::try_new(
        buffer![42i64].into_array(),
        Scalar::from(40i64)
    ).unwrap())]
    #[case::for_zero_ref(FoRArray::try_new(
        buffer![0u32, 1, 2, 3, 4].into_array(),
        Scalar::from(0u32)
    ).unwrap())]
    // Large arrays (> 1024 elements for fastlanes chunking)
    #[case::for_large(FoRArray::try_new(
        PrimitiveArray::from_iter((0..1500).map(|i| 5000 + i)).into_array(),
        Scalar::from(5000i32)
    ).unwrap())]
    #[case::for_very_large(FoRArray::try_new(
        PrimitiveArray::from_iter((0..3072).map(|i| 10000 + i as i64)).into_array(),
        Scalar::from(10000i64)
    ).unwrap())]
    #[case::for_large_nullable(FoRArray::try_new(
        PrimitiveArray::from_option_iter((0..2048).map(|i| (i % 15 == 0).then_some(1000 + i))).into_array(),
        Scalar::from(1000i32)
    ).unwrap())]
    // Arrays with large deltas from reference
    #[case::for_large_deltas(FoRArray::try_new(
        buffer![100i64, 200, 300, 400, 500].into_array(),
        Scalar::from(100i64)
    ).unwrap())]

    fn test_for_consistency(#[case] array: FoRArray) {
        test_array_consistency(&array.into_array());
    }

    #[rstest]
    #[case::for_i32_basic(FoRArray::try_new(
        buffer![100i32, 101, 102, 103, 104].into_array(),
        Scalar::from(100i32)
    ).unwrap())]
    #[case::for_u32_basic(FoRArray::try_new(
        buffer![1000u32, 1001, 1002, 1003, 1004].into_array(),
        Scalar::from(1000u32)
    ).unwrap())]
    #[case::for_i64_basic(FoRArray::try_new(
        buffer![5000i64, 5001, 5002, 5003, 5004].into_array(),
        Scalar::from(5000i64)
    ).unwrap())]
    #[case::for_u64_basic(FoRArray::try_new(
        buffer![10000u64, 10001, 10002, 10003, 10004].into_array(),
        Scalar::from(10000u64)
    ).unwrap())]
    #[case::for_i32_large(FoRArray::try_new(
        PrimitiveArray::from_iter((0..100).map(|i| 2000 + i)).into_array(),
        Scalar::from(2000i32)
    ).unwrap())]
    fn test_for_binary_numeric(#[case] array: FoRArray) {
        test_binary_numeric_array(array.into_array());
    }
}
