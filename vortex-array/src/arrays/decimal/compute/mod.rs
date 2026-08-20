// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod between;
mod cast;
mod fill_null;
mod fixed_width;
mod mask;
pub mod rules;

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_buffer::buffer;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::DecimalArray;
    use crate::compute::conformance::binary_numeric::test_binary_numeric_array;
    use crate::compute::conformance::consistency::test_array_consistency;
    use crate::dtype::DecimalDType;
    use crate::dtype::NativeDecimalType;
    use crate::validity::Validity;

    #[rstest]
    // From test_all_consistency
    #[case::decimal_array(DecimalArray::new(
        buffer![100i128, 200i128, 300i128, 400i128, 500i128],
        DecimalDType::new(19, 2),
        Validity::NonNullable,
    ))]
    #[case::decimal_nullable(DecimalArray::new(
        buffer![1000i128, 2000i128, 3000i128, 4000i128, 5000i128],
        DecimalDType::new(19, 3),
        Validity::from_iter([true, false, true, true, false]),
    ))]
    // Additional test cases
    #[case::decimal_small_precision(DecimalArray::new(
        buffer![10i128, 20i128, 30i128],
        DecimalDType::new(5, 1),
        Validity::NonNullable,
    ))]
    #[case::decimal_single(DecimalArray::new(
        buffer![42i128],
        DecimalDType::new(10, 0),
        Validity::NonNullable,
    ))]
    #[case::decimal_large_scale(DecimalArray::new(
        buffer![123456789012345i128, 987654321098765i128],
        DecimalDType::new(20, 10),
        Validity::NonNullable,
    ))]
    #[case::decimal_negative(DecimalArray::new(
        buffer![-100i128, -200i128, 300i128, -400i128],
        DecimalDType::new(10, 2),
        Validity::NonNullable,
    ))]
    fn test_decimal_consistency(#[case] array: DecimalArray) {
        test_array_consistency(
            &array.into_array(),
            &mut array_session().create_execution_ctx(),
        );
    }

    #[rstest]
    #[case::decimal_array(DecimalArray::new(
        buffer![100i128, 200i128, 300i128, 400i128, 500i128],
        DecimalDType::new(19, 2),
        Validity::NonNullable,
    ))]
    #[case::decimal_nullable(DecimalArray::new(
        buffer![1000i128, 2000i128, 3000i128, 4000i128, 5000i128],
        DecimalDType::new(19, 3),
        Validity::from_iter([true, false, true, true, false]),
    ))]
    #[case::decimal_small_precision(DecimalArray::new(
        buffer![10i128, 20i128, 30i128],
        DecimalDType::new(5, 1),
        Validity::NonNullable,
    ))]
    #[case::decimal_narrow_storage(DecimalArray::new(
        buffer![10i8, 20i8, 30i8],
        DecimalDType::new(5, 1),
        Validity::NonNullable,
    ))]
    #[case::decimal_widened_carry(DecimalArray::new(
        buffer![
            <i128 as NativeDecimalType>::MAX_BY_PRECISION[38],
            <i128 as NativeDecimalType>::MIN_BY_PRECISION[38],
        ],
        DecimalDType::new(38, 0),
        Validity::NonNullable,
    ))]
    #[case::decimal_single(DecimalArray::new(
        buffer![42i128],
        DecimalDType::new(10, 0),
        Validity::NonNullable,
    ))]
    #[case::decimal_large_scale(DecimalArray::new(
        buffer![123456789012345i128, 987654321098765i128],
        DecimalDType::new(20, 10),
        Validity::NonNullable,
    ))]
    #[case::decimal_negative(DecimalArray::new(
        buffer![-100i128, -200i128, 300i128, -400i128],
        DecimalDType::new(10, 2),
        Validity::NonNullable,
    ))]
    #[case::decimal_negative_scale(DecimalArray::new(
        buffer![5i64, -3i64, 600i64],
        DecimalDType::new(10, -2),
        Validity::NonNullable,
    ))]
    fn test_decimal_binary_numeric(#[case] array: DecimalArray) {
        test_binary_numeric_array(
            &array.into_array(),
            &mut array_session().create_execution_ctx(),
        );
    }
}
