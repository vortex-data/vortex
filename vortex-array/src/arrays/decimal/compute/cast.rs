// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_panic;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Decimal;
use crate::arrays::DecimalArray;
use crate::dtype::DType;
use crate::dtype::DecimalType;
use crate::dtype::NativeDecimalType;
use crate::match_each_decimal_value_type;
use crate::scalar_fn::fns::cast::CastKernel;

impl CastKernel for Decimal {
    fn cast(
        array: ArrayView<'_, Decimal>,
        dtype: &DType,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        // Early return if not casting to decimal
        let DType::Decimal(to_decimal_dtype, to_nullability) = dtype else {
            return Ok(None);
        };
        let DType::Decimal(from_decimal_dtype, _) = array.dtype() else {
            vortex_panic!(
                "DecimalArray must have decimal dtype, got {:?}",
                array.dtype()
            );
        };

        // Scale changes are not yet supported
        if from_decimal_dtype.scale() != to_decimal_dtype.scale() {
            vortex_bail!(
                "Casting decimal with scale {} to scale {} not yet implemented",
                from_decimal_dtype.scale(),
                to_decimal_dtype.scale()
            );
        }

        // Downcasting precision is not yet supported
        if to_decimal_dtype.precision() < from_decimal_dtype.precision() {
            vortex_bail!(
                "Downcasting decimal from precision {} to {} not yet implemented",
                from_decimal_dtype.precision(),
                to_decimal_dtype.precision()
            );
        }

        // If the dtype is exactly the same, return self
        if array.dtype() == dtype {
            return Ok(Some(array.array().clone()));
        }

        // Cast the validity to the new nullability
        let new_validity = array
            .validity()
            .cast_nullability(*to_nullability, array.len())?;

        // If the target needs a wider physical type, upcast the values
        let target_values_type = DecimalType::smallest_decimal_value_type(to_decimal_dtype);
        let array = if target_values_type > array.values_type() {
            upcast_decimal_values(array, target_values_type)?
        } else {
            array.array().as_::<Decimal>().into_owned()
        };

        // SAFETY: new_validity same length as previous validity, just cast
        unsafe {
            Ok(Some(
                DecimalArray::new_unchecked_handle(
                    array.buffer_handle().clone(),
                    array.values_type(),
                    *to_decimal_dtype,
                    new_validity,
                )
                .into_array(),
            ))
        }
    }
}

/// Upcast a DecimalArray to a wider physical representation (e.g., i32 -> i64) while keeping
/// the same precision and scale.
///
/// This is useful when you need to widen the underlying storage type to accommodate operations
/// that might overflow the current representation, or to match the physical type expected by
/// downstream consumers.
///
/// # Errors
///
/// Returns an error if `to_values_type` is narrower than the array's current values type.
/// Only upcasting (widening) is supported.
pub fn upcast_decimal_values(
    array: ArrayView<'_, Decimal>,
    to_values_type: DecimalType,
) -> VortexResult<DecimalArray> {
    let from_values_type = array.values_type();

    // If already the target type, just clone
    if from_values_type == to_values_type {
        return Ok(array.array().as_::<Decimal>().into_owned());
    }

    // Only allow upcasting (widening)
    if to_values_type < from_values_type {
        vortex_bail!(
            "Cannot downcast decimal values from {:?} to {:?}. Only upcasting is supported.",
            from_values_type,
            to_values_type
        );
    }

    let decimal_dtype = array.decimal_dtype();
    let validity = array.validity();

    // Use match_each_decimal_value_type to dispatch based on source and target types
    match_each_decimal_value_type!(from_values_type, |F| {
        let from_buffer = array.buffer::<F>();
        match_each_decimal_value_type!(to_values_type, |T| {
            let to_buffer = upcast_decimal_buffer::<F, T>(from_buffer);
            Ok(DecimalArray::new(to_buffer, decimal_dtype, validity))
        })
    })
}

/// Upcast a buffer of decimal values from type F to type T.
/// Since T is wider than F, this conversion never fails.
fn upcast_decimal_buffer<F: NativeDecimalType, T: NativeDecimalType>(from: Buffer<F>) -> Buffer<T> {
    from.iter()
        .map(|&v| T::from(v).vortex_expect("upcast should never fail"))
        .collect()
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_buffer::buffer;

    use super::upcast_decimal_values;
    use crate::IntoArray;
    use crate::arrays::DecimalArray;
    use crate::builtins::ArrayBuiltins;
    use crate::canonical::ToCanonical;
    use crate::compute::conformance::cast::test_cast_conformance;
    use crate::dtype::DType;
    use crate::dtype::DecimalDType;
    use crate::dtype::DecimalType;
    use crate::dtype::Nullability;
    use crate::validity::Validity;

    #[test]
    fn cast_decimal_to_nullable() {
        let decimal_dtype = DecimalDType::new(10, 2);
        let array = DecimalArray::new(
            buffer![100i32, 200, 300],
            decimal_dtype,
            Validity::NonNullable,
        );

        // Cast to nullable
        let nullable_dtype = DType::Decimal(decimal_dtype, Nullability::Nullable);
        let casted = array
            .into_array()
            .cast(nullable_dtype.clone())
            .unwrap()
            .to_decimal();

        assert_eq!(casted.dtype(), &nullable_dtype);
        assert!(matches!(casted.validity(), Validity::AllValid));
        assert_eq!(casted.len(), 3);
    }

    #[test]
    fn cast_nullable_to_non_nullable() {
        let decimal_dtype = DecimalDType::new(10, 2);

        // Create nullable array with no nulls
        let array = DecimalArray::new(buffer![100i32, 200, 300], decimal_dtype, Validity::AllValid);

        // Cast to non-nullable
        let non_nullable_dtype = DType::Decimal(decimal_dtype, Nullability::NonNullable);
        let casted = array
            .into_array()
            .cast(non_nullable_dtype.clone())
            .unwrap()
            .to_decimal();

        assert_eq!(casted.dtype(), &non_nullable_dtype);
        assert!(matches!(casted.validity(), Validity::NonNullable));
    }

    #[test]
    #[should_panic(expected = "Cannot cast array with invalid values to non-nullable type")]
    fn cast_nullable_with_nulls_to_non_nullable_fails() {
        let decimal_dtype = DecimalDType::new(10, 2);

        // Create nullable array with nulls
        let array = DecimalArray::from_option_iter([Some(100i32), None, Some(300)], decimal_dtype);

        // Attempt to cast to non-nullable should fail
        let non_nullable_dtype = DType::Decimal(decimal_dtype, Nullability::NonNullable);
        array
            .into_array()
            .cast(non_nullable_dtype)
            .and_then(|a| a.to_canonical().map(|c| c.into_array()))
            .unwrap();
    }

    #[test]
    fn cast_different_scale_fails() {
        let array = DecimalArray::new(
            buffer![100i32],
            DecimalDType::new(10, 2),
            Validity::NonNullable,
        );

        // Try to cast to different scale - not supported
        let different_dtype = DType::Decimal(DecimalDType::new(15, 3), Nullability::NonNullable);
        let result = array
            .into_array()
            .cast(different_dtype)
            .and_then(|a| a.to_canonical().map(|c| c.into_array()));

        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Casting decimal with scale 2 to scale 3 not yet implemented")
        );
    }

    #[test]
    fn cast_downcast_precision_fails() {
        let array = DecimalArray::new(
            buffer![100i64],
            DecimalDType::new(18, 2),
            Validity::NonNullable,
        );

        // Try to downcast precision - not supported
        let smaller_dtype = DType::Decimal(DecimalDType::new(10, 2), Nullability::NonNullable);
        let result = array
            .into_array()
            .cast(smaller_dtype)
            .and_then(|a| a.to_canonical().map(|c| c.into_array()));

        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Downcasting decimal from precision 18 to 10 not yet implemented")
        );
    }

    #[test]
    fn cast_upcast_precision_succeeds() {
        let array = DecimalArray::new(
            buffer![100i32, 200, 300],
            DecimalDType::new(10, 2),
            Validity::NonNullable,
        );

        // Cast to higher precision with same scale - should succeed
        let wider_dtype = DType::Decimal(DecimalDType::new(38, 2), Nullability::NonNullable);
        let casted = array.into_array().cast(wider_dtype).unwrap().to_decimal();

        assert_eq!(casted.precision(), 38);
        assert_eq!(casted.scale(), 2);
        assert_eq!(casted.len(), 3);
        // Should be stored in i128 now (precision 38 requires i128)
        assert_eq!(casted.values_type(), DecimalType::I128);
    }

    #[test]
    fn cast_to_non_decimal_returns_err() {
        let array = DecimalArray::new(
            buffer![100i32],
            DecimalDType::new(10, 2),
            Validity::NonNullable,
        );

        // Try to cast to non-decimal type - should fail since no kernel can handle it
        let result = array
            .into_array()
            .cast(DType::Utf8(Nullability::NonNullable))
            .and_then(|a| a.to_canonical().map(|c| c.into_array()));

        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("No CastKernel to cast canonical array")
        );
    }

    #[rstest]
    #[case(DecimalArray::new(buffer![100i32, 200, 300], DecimalDType::new(10, 2), Validity::NonNullable))]
    #[case(DecimalArray::new(buffer![10000i64, 20000, 30000], DecimalDType::new(18, 4), Validity::NonNullable))]
    #[case(DecimalArray::from_option_iter([Some(100i32), None, Some(300)], DecimalDType::new(10, 2)))]
    #[case(DecimalArray::new(buffer![42i32], DecimalDType::new(5, 1), Validity::NonNullable))]
    fn test_cast_decimal_conformance(#[case] array: DecimalArray) {
        test_cast_conformance(&array.into_array());
    }

    #[test]
    fn upcast_decimal_values_i32_to_i64() {
        let decimal_dtype = DecimalDType::new(10, 2);
        let array = DecimalArray::new(
            buffer![100i32, 200, 300],
            decimal_dtype,
            Validity::NonNullable,
        );

        assert_eq!(array.values_type(), DecimalType::I32);

        let array = array.as_view();
        let casted = upcast_decimal_values(array, DecimalType::I64).unwrap();

        assert_eq!(casted.values_type(), DecimalType::I64);
        assert_eq!(casted.decimal_dtype(), decimal_dtype);
        assert_eq!(casted.len(), 3);

        // Verify values are preserved
        let buffer = casted.buffer::<i64>();
        assert_eq!(buffer.as_ref(), &[100i64, 200, 300]);
    }

    #[test]
    fn upcast_decimal_values_i64_to_i128() {
        let decimal_dtype = DecimalDType::new(18, 4);
        let array = DecimalArray::new(
            buffer![10000i64, 20000, 30000],
            decimal_dtype,
            Validity::NonNullable,
        );

        let array = array.as_view();
        let casted = upcast_decimal_values(array, DecimalType::I128).unwrap();

        assert_eq!(casted.values_type(), DecimalType::I128);
        assert_eq!(casted.decimal_dtype(), decimal_dtype);

        let buffer = casted.buffer::<i128>();
        assert_eq!(buffer.as_ref(), &[10000i128, 20000, 30000]);
    }

    #[test]
    fn upcast_decimal_values_same_type_returns_clone() {
        let decimal_dtype = DecimalDType::new(10, 2);
        let array = DecimalArray::new(
            buffer![100i32, 200, 300],
            decimal_dtype,
            Validity::NonNullable,
        );

        let array = array.as_view();
        let casted = upcast_decimal_values(array, DecimalType::I32).unwrap();

        assert_eq!(casted.values_type(), DecimalType::I32);
        assert_eq!(casted.decimal_dtype(), decimal_dtype);
    }

    #[test]
    fn upcast_decimal_values_with_nulls() {
        let decimal_dtype = DecimalDType::new(10, 2);
        let array = DecimalArray::from_option_iter([Some(100i32), None, Some(300)], decimal_dtype);

        let array = array.as_view();
        let casted = upcast_decimal_values(array, DecimalType::I64).unwrap();

        assert_eq!(casted.values_type(), DecimalType::I64);
        assert_eq!(casted.len(), 3);

        // Check validity is preserved
        let mask = casted.validity_mask().unwrap();
        assert!(mask.value(0));
        assert!(!mask.value(1));
        assert!(mask.value(2));

        // Check non-null values
        let buffer = casted.buffer::<i64>();
        assert_eq!(buffer[0], 100);
        assert_eq!(buffer[2], 300);
    }

    #[test]
    fn upcast_decimal_values_downcast_fails() {
        let decimal_dtype = DecimalDType::new(18, 4);
        let array = DecimalArray::new(
            buffer![10000i64, 20000, 30000],
            decimal_dtype,
            Validity::NonNullable,
        );

        // Attempt to downcast from i64 to i32 should fail
        let array = array.as_view();
        let result = upcast_decimal_values(array, DecimalType::I32);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Cannot downcast decimal values")
        );
    }
}
