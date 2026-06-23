// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod cast;
mod compare;
mod filter;
pub(crate) mod is_constant;
pub(crate) mod kernel;
mod mask;
mod take;

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::compute::conformance::consistency::test_array_consistency;
    use vortex_array::dtype::DecimalDType;
    use vortex_buffer::buffer;

    use crate::DecimalByteParts;
    use crate::DecimalBytePartsArray;

    #[rstest]
    // Basic decimal byte parts arrays
    #[case::decimal_i32(DecimalByteParts::try_new(
        buffer![100i32, 200, 300, 400, 500].into_array(),
        DecimalDType::new(10, 2)
    ).unwrap())]
    #[case::decimal_i64(DecimalByteParts::try_new(
        buffer![1000i64, 2000, 3000, 4000, 5000].into_array(),
        DecimalDType::new(19, 4)
    ).unwrap())]
    // Nullable arrays
    #[case::decimal_nullable_i32(DecimalByteParts::try_new(
        PrimitiveArray::from_option_iter([Some(100i32), None, Some(300), Some(400), None]).into_array(),
        DecimalDType::new(10, 2)
    ).unwrap())]
    #[case::decimal_nullable_i64(DecimalByteParts::try_new(
        PrimitiveArray::from_option_iter([Some(1000i64), None, Some(3000), Some(4000), None]).into_array(),
        DecimalDType::new(19, 4)
    ).unwrap())]
    // Different precision/scale combinations
    #[case::decimal_high_precision(DecimalByteParts::try_new(
        buffer![123456789i32, 987654321, -123456789].into_array(),
        DecimalDType::new(38, 10)
    ).unwrap())]
    #[case::decimal_zero_scale(DecimalByteParts::try_new(
        buffer![100i32, 200, 300].into_array(),
        DecimalDType::new(10, 0)
    ).unwrap())]
    // Edge cases
    #[case::decimal_single(DecimalByteParts::try_new(
        buffer![42i32].into_array(),
        DecimalDType::new(5, 1)
    ).unwrap())]
    #[case::decimal_negative(DecimalByteParts::try_new(
        buffer![-100i32, -200, 300, -400, 500].into_array(),
        DecimalDType::new(10, 2)
    ).unwrap())]
    // Large arrays
    #[case::decimal_large(DecimalByteParts::try_new(
        PrimitiveArray::from_iter((0..1500).map(|i| i * 100)).into_array(),
        DecimalDType::new(10, 2)
    ).unwrap())]
    #[case::decimal_large_i64(DecimalByteParts::try_new(
        PrimitiveArray::from_iter((0..2000i64).map(|i| i * 1000000)).into_array(),
        DecimalDType::new(19, 6)
    ).unwrap())]

    fn test_decimal_byte_parts_consistency(#[case] array: DecimalBytePartsArray) {
        let ctx = &mut array_session().create_execution_ctx();
        test_array_consistency(&array.into_array(), ctx);
    }
}
