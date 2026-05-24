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
// Splitting decimal values into 64-bit parts intentionally truncates wider integers.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap
)]
mod tests {
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::compute::conformance::consistency::test_array_consistency;
    use vortex_array::dtype::DecimalDType;
    use vortex_buffer::buffer;

    use crate::DecimalByteParts;
    use crate::DecimalBytePartsArray;

    /// Builds a two-part (`i128`) byte-parts array from raw values, splitting them the same way the
    /// compressor does.
    fn i128_parts(values: &[i128], decimal_dtype: DecimalDType) -> DecimalBytePartsArray {
        let msp = PrimitiveArray::from_iter(values.iter().map(|v| (v >> 64) as i64)).into_array();
        let low = PrimitiveArray::from_iter(values.iter().map(|v| *v as u64)).into_array();
        DecimalByteParts::try_new_parts(msp, vec![low], decimal_dtype).unwrap()
    }

    /// Builds a four-part (`i256`) byte-parts array from raw `i128` values widened to `i256`.
    fn i256_parts(values: &[i128], decimal_dtype: DecimalDType) -> DecimalBytePartsArray {
        use vortex_array::dtype::i256;
        let widened: Vec<i256> = values.iter().map(|v| i256::from_i128(*v)).collect();
        let msp = PrimitiveArray::from_iter(widened.iter().map(|v| {
            let (_, upper) = v.to_parts();
            (upper >> 64) as i64
        }))
        .into_array();
        let p0 =
            PrimitiveArray::from_iter(widened.iter().map(|v| v.to_parts().1 as u64)).into_array();
        let p1 = PrimitiveArray::from_iter(widened.iter().map(|v| (v.to_parts().0 >> 64) as u64))
            .into_array();
        let p2 =
            PrimitiveArray::from_iter(widened.iter().map(|v| v.to_parts().0 as u64)).into_array();
        DecimalByteParts::try_new_parts(msp, vec![p0, p1, p2], decimal_dtype).unwrap()
    }

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
    // Multi-part i128 decimals (msp + 1 lower part)
    #[case::decimal_i128(i128_parts(
        &[0, 1, -1, 10i128.pow(30), -(10i128.pow(30)), i64::MAX as i128 + 1, 10i128.pow(37)],
        DecimalDType::new(38, 4)
    ))]
    #[case::decimal_i128_nullable(DecimalByteParts::try_new_parts(
        PrimitiveArray::from_option_iter([Some(1i64), None, Some(-3), Some(0), None]).into_array(),
        vec![PrimitiveArray::from_iter([5u64, 0, 7, 9, 0]).into_array()],
        DecimalDType::new(38, 2)
    ).unwrap())]
    // Multi-part i256 decimals (msp + 3 lower parts)
    #[case::decimal_i256(i256_parts(
        &[0, 1, -1, 10i128.pow(30), -(10i128.pow(30)), i128::MAX, i128::MIN],
        DecimalDType::new(76, 8)
    ))]

    fn test_decimal_byte_parts_consistency(#[case] array: DecimalBytePartsArray) {
        test_array_consistency(&array.into_array());
    }
}
