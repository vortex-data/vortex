// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod cast;
mod fill_null;
mod fixed_width;
mod mask;
pub mod rules;

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::DecimalArray;
    use crate::compute::conformance::binary_numeric::test_binary_numeric_array;
    use crate::compute::conformance::consistency::test_array_consistency;
    use crate::dtype::DecimalDType;
    use crate::dtype::NativeDecimalType;
    use crate::patches::Patches;
    use crate::validity::Validity;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    #[test]
    fn patch_rejects_mismatched_decimal_dtype() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let array = DecimalArray::from_iter([1i8], DecimalDType::new(2, 0));
        let values = DecimalArray::from_iter([500i16], DecimalDType::new(4, 0));
        let patches = Patches::new(
            1,
            0,
            buffer![0u32].into_array(),
            values.into_array(),
            None,
            &mut ctx,
        )?;

        let error = array.patch(&patches, &mut ctx).unwrap_err().to_string();
        assert!(error.contains("decimal(4,0)"), "{error}");
        assert!(error.contains("decimal(2,0)"), "{error}");
        Ok(())
    }

    #[rstest]
    // From test_all_consistency
    #[case::decimal_array(DecimalArray::from_iter([100i128, 200i128, 300i128, 400i128, 500i128], DecimalDType::new(19, 2)))]
    #[case::decimal_nullable(DecimalArray::new(
        buffer![1000i128, 2000i128, 3000i128, 4000i128, 5000i128],
        DecimalDType::new(19, 3),
        Validity::from_iter([true, false, true, true, false]),
        &mut SESSION.create_execution_ctx(),
    ))]
    // Additional test cases
    #[case::decimal_small_precision(DecimalArray::from_iter([10i32, 20, 30], DecimalDType::new(5, 1)))]
    #[case::decimal_single(DecimalArray::from_iter([42i64], DecimalDType::new(10, 0)))]
    #[case::decimal_large_scale(DecimalArray::from_iter([123456789012345i128, 987654321098765i128], DecimalDType::new(20, 10)))]
    #[case::decimal_negative(DecimalArray::from_iter([-100i64, -200, 300, -400], DecimalDType::new(10, 2)))]
    fn test_decimal_consistency(#[case] array: DecimalArray) {
        test_array_consistency(&array.into_array(), &mut SESSION.create_execution_ctx());
    }

    #[rstest]
    #[case::decimal_array(DecimalArray::from_iter([100i128, 200i128, 300i128, 400i128, 500i128], DecimalDType::new(19, 2)))]
    #[case::decimal_nullable(DecimalArray::new(
        buffer![1000i128, 2000i128, 3000i128, 4000i128, 5000i128],
        DecimalDType::new(19, 3),
        Validity::from_iter([true, false, true, true, false]),
        &mut SESSION.create_execution_ctx(),
    ))]
    #[case::decimal_small_precision(DecimalArray::from_iter([10i32, 20, 30], DecimalDType::new(5, 1)))]
    #[case::decimal_i8(DecimalArray::from_iter([10i8, 20, 30], DecimalDType::new(2, 1)))]
    #[case::decimal_widened_carry(DecimalArray::from_iter([
            <i128 as NativeDecimalType>::MAX_BY_PRECISION[38],
            <i128 as NativeDecimalType>::MIN_BY_PRECISION[38],
        ], DecimalDType::new(38, 0)))]
    #[case::decimal_single(DecimalArray::from_iter([42i64], DecimalDType::new(10, 0)))]
    #[case::decimal_large_scale(DecimalArray::from_iter([123456789012345i128, 987654321098765i128], DecimalDType::new(20, 10)))]
    #[case::decimal_negative(DecimalArray::from_iter([-100i64, -200, 300, -400], DecimalDType::new(10, 2)))]
    #[case::decimal_negative_scale(DecimalArray::from_iter([5i64, -3i64, 600i64], DecimalDType::new(10, -2)))]
    fn test_decimal_binary_numeric(#[case] array: DecimalArray) {
        test_binary_numeric_array(&array.into_array(), &mut SESSION.create_execution_ctx());
    }
}
