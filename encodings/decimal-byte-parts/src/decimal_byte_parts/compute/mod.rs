// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod between;
mod cast;
pub(crate) mod compare;
mod filter;
pub(crate) mod is_constant;
pub(crate) mod kernel;
mod mask;
mod take;
mod two_limb;

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::compute::conformance::consistency::test_array_consistency;
    use vortex_array::dtype::DecimalDType;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_buffer::buffer;

    use crate::DecimalByteParts;
    use crate::DecimalBytePartsArray;

    // 99 i128 values whose high (i64) limb varies in sign and whose low (u64) limb spans the full
    // unsigned range, so limb tie-breaks and sign handling are covered across SIMD chunks.
    #[allow(clippy::cast_possible_truncation)]
    fn large_two_limb_values() -> Vec<i128> {
        (0..99i128)
            .map(|i| ((i - 50) << 64) | i128::from((i as u64).wrapping_mul(0x0123_4567_89ab_cdef)))
            .collect()
    }

    #[allow(clippy::cast_possible_truncation)]
    fn two_limb(values: &[i128], validity: Validity, dt: DecimalDType) -> DecimalBytePartsArray {
        let highs: Buffer<i64> = values.iter().map(|v| (v >> 64) as i64).collect();
        let lows: Buffer<u64> = values.iter().map(|v| *v as u64).collect();
        DecimalByteParts::try_new_with_lower(
            PrimitiveArray::new(highs, validity).into_array(),
            PrimitiveArray::new(lows, Validity::NonNullable).into_array(),
            dt,
        )
        .unwrap()
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
    // Two-limb i128 representation
    #[case::two_limb(two_limb(
        &[0, -1, (3i128 << 64) | 42, -(9i128 << 64) | 17, i128::from(i64::MIN), (7i128 << 64)],
        Validity::NonNullable,
        DecimalDType::new(38, 2),
    ))]
    #[case::two_limb_nullable(two_limb(
        &[0, -1, (3i128 << 64) | 42, -(9i128 << 64) | 17, 5],
        Validity::Array(BoolArray::from_iter([true, false, true, true, false]).into_array()),
        DecimalDType::new(38, 2),
    ))]
    // 99 values (not a multiple of 8) so the AVX-512 limb kernel's vectorized main loop *and* its
    // scalar tail are both exercised when validating compare/between against canonical.
    #[case::two_limb_large(two_limb(
        &large_two_limb_values(),
        Validity::NonNullable,
        DecimalDType::new(38, 2),
    ))]
    #[case::two_limb_large_nullable(two_limb(
        &large_two_limb_values(),
        Validity::Array(
            BoolArray::from_iter((0..99).map(|i| i % 3 != 0)).into_array(),
        ),
        DecimalDType::new(38, 2),
    ))]

    fn test_decimal_byte_parts_consistency(#[case] array: DecimalBytePartsArray) {
        test_array_consistency(&array.into_array());
    }
}
