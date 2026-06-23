// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod cast;
mod compare;
pub(crate) mod is_constant;
pub(crate) mod is_sorted;

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::dict::TakeExecute;
use vortex_array::arrays::filter::FilterReduce;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::FoR;
use crate::r#for::array::FoRArrayExt;

impl TakeExecute for FoR {
    fn take(
        array: ArrayView<'_, Self>,
        indices: &ArrayRef,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        Ok(Some(
            FoR::try_new(
                array.encoded().take(indices.clone())?,
                array.reference_scalar().clone(),
            )?
            .into_array(),
        ))
    }
}

impl FilterReduce for FoR {
    fn filter(array: ArrayView<'_, Self>, mask: &Mask) -> VortexResult<Option<ArrayRef>> {
        FoR::try_new(
            array.encoded().filter(mask.clone())?,
            array.reference_scalar().clone(),
        )
        .map(|a| Some(a.into_array()))
    }
}

#[cfg(test)]
mod test {
    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::compute::conformance::filter::test_filter_conformance;
    use vortex_array::scalar::Scalar;
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;

    use crate::FoR;
    use crate::FoRArray;

    fn fa(encoded: ArrayRef, reference: Scalar) -> FoRArray {
        FoR::try_new(encoded, reference).vortex_expect("FoR array construction should succeed")
    }

    #[test]
    fn test_filter_for_array() {
        let for_array = fa(
            buffer![100i32, 101, 102, 103, 104].into_array(),
            Scalar::from(100i32),
        );
        test_filter_conformance(&for_array.into_array());

        let for_array = fa(
            buffer![1000u64, 1001, 1002, 1003, 1004].into_array(),
            Scalar::from(1000u64),
        );
        test_filter_conformance(&for_array.into_array());

        let values =
            PrimitiveArray::from_option_iter([Some(50i16), None, Some(52), Some(53), None]);
        let for_array = fa(values.into_array(), Scalar::from(50i16));
        test_filter_conformance(&for_array.into_array());
    }

    #[rstest]
    #[case(fa(buffer![100i32, 101, 102, 103, 104].into_array(), Scalar::from(100i32)))]
    #[case(fa(buffer![1000u64, 1001, 1002, 1003, 1004].into_array(), Scalar::from(1000u64)))]
    #[case(fa(
        PrimitiveArray::from_option_iter([Some(50i16), None, Some(52), Some(53), None]).into_array(),
        Scalar::from(50i16)
    ))]
    #[case(fa(buffer![-100i32, -99, -98, -97, -96].into_array(), Scalar::from(-100i32)))]
    #[case(fa(buffer![42i64].into_array(), Scalar::from(40i64)))]
    fn test_take_for_conformance(#[case] for_array: FoRArray) {
        use vortex_array::compute::conformance::take::test_take_conformance;
        test_take_conformance(&for_array.into_array());
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::compute::conformance::binary_numeric::test_binary_numeric_array;
    use vortex_array::compute::conformance::consistency::test_array_consistency;
    use vortex_array::scalar::Scalar;
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;

    use crate::FoR;
    use crate::FoRArray;

    fn fa(encoded: ArrayRef, reference: Scalar) -> FoRArray {
        FoR::try_new(encoded, reference).vortex_expect("FoR array construction should succeed")
    }

    #[rstest]
    #[case::for_i32(fa(buffer![100i32, 101, 102, 103, 104].into_array(), Scalar::from(100i32)))]
    #[case::for_u64(fa(buffer![1000u64, 1001, 1002, 1003, 1004].into_array(), Scalar::from(1000u64)))]
    #[case::for_nullable_i16(fa(
        PrimitiveArray::from_option_iter([Some(50i16), None, Some(52), Some(53), None]).into_array(),
        Scalar::from(50i16)
    ))]
    #[case::for_nullable_i32(fa(
        PrimitiveArray::from_option_iter([Some(200i32), None, Some(202), Some(203), None]).into_array(),
        Scalar::from(200i32)
    ))]
    #[case::for_negative(fa(buffer![-100i32, -99, -98, -97, -96].into_array(), Scalar::from(-100i32)))]
    #[case::for_single(fa(buffer![42i64].into_array(), Scalar::from(40i64)))]
    #[case::for_zero_ref(fa(buffer![0u32, 1, 2, 3, 4].into_array(), Scalar::from(0u32)))]
    #[case::for_large(fa(
        PrimitiveArray::from_iter((0..1500).map(|i| 5000 + i)).into_array(),
        Scalar::from(5000i32)
    ))]
    #[case::for_very_large(fa(
        PrimitiveArray::from_iter((0..3072).map(|i| 10000 + i as i64)).into_array(),
        Scalar::from(10000i64)
    ))]
    #[case::for_large_nullable(fa(
        PrimitiveArray::from_option_iter((0..2048).map(|i| (i % 15 == 0).then_some(1000 + i))).into_array(),
        Scalar::from(1000i32)
    ))]
    #[case::for_large_deltas(fa(buffer![100i64, 200, 300, 400, 500].into_array(), Scalar::from(100i64)))]

    fn test_for_consistency(#[case] array: FoRArray) {
        test_array_consistency(
            &array.into_array(),
            &mut array_session().create_execution_ctx(),
        );
    }

    #[rstest]
    #[case::for_i32_basic(fa(buffer![100i32, 101, 102, 103, 104].into_array(), Scalar::from(100i32)))]
    #[case::for_u32_basic(fa(buffer![1000u32, 1001, 1002, 1003, 1004].into_array(), Scalar::from(1000u32)))]
    #[case::for_i64_basic(fa(buffer![5000i64, 5001, 5002, 5003, 5004].into_array(), Scalar::from(5000i64)))]
    #[case::for_u64_basic(fa(buffer![10000u64, 10001, 10002, 10003, 10004].into_array(), Scalar::from(10000u64)))]
    #[case::for_i32_large(fa(
        PrimitiveArray::from_iter((0..100).map(|i| 2000 + i)).into_array(),
        Scalar::from(2000i32)
    ))]
    fn test_for_binary_numeric(#[case] array: FoRArray) {
        test_binary_numeric_array(
            &array.into_array(),
            &mut array_session().create_execution_ctx(),
        );
    }
}
