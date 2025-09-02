// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_bail, vortex_panic};

use crate::arrays::{DecimalArray, DecimalVTable};
use crate::compute::{CastKernel, CastKernelAdapter};
use crate::stats::ArrayStats;
use crate::vtable::ValidityHelper;
use crate::{ArrayRef, register_kernel};

impl CastKernel for DecimalVTable {
    fn cast(&self, array: &DecimalArray, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        // Early return if not casting to decimal
        let DType::Decimal(to_precision_scale, to_nullability) = dtype else {
            return Ok(None);
        };
        let DType::Decimal(from_precision_scale, _) = array.dtype() else {
            vortex_panic!(
                "DecimalArray must have decimal dtype, got {:?}",
                array.dtype()
            );
        };

        // We only support casting to the same decimal type with different nullability
        if from_precision_scale != to_precision_scale {
            vortex_bail!(
                "Cannot cast decimal({},{}) to decimal({},{})",
                from_precision_scale.precision(),
                from_precision_scale.scale(),
                to_precision_scale.precision(),
                to_precision_scale.scale()
            );
        }

        // If the dtype is exactly the same, return self
        if array.dtype() == dtype {
            return Ok(Some(array.to_array()));
        }

        // Cast the validity to the new nullability
        let new_validity = array.validity().clone().cast_nullability(*to_nullability)?;

        // Construct DecimalArray directly since we can't use new() without knowing the concrete type
        Ok(Some(
            DecimalArray {
                dtype: DType::Decimal(*from_precision_scale, *to_nullability),
                values: array.byte_buffer(),
                values_type: array.values_type(),
                validity: new_validity,
                stats_set: ArrayStats::default(),
            }
            .to_array(),
        ))
    }
}

register_kernel!(CastKernelAdapter(DecimalVTable).lift());

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_buffer::buffer;
    use vortex_dtype::{DType, DecimalDType, Nullability};

    use crate::arrays::DecimalArray;
    use crate::canonical::ToCanonical;
    use crate::compute::cast;
    use crate::compute::conformance::cast::test_cast_conformance;
    use crate::validity::Validity;
    use crate::vtable::ValidityHelper;

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
        let casted = cast(array.as_ref(), &nullable_dtype).unwrap().to_decimal();

        assert_eq!(casted.dtype(), &nullable_dtype);
        assert_eq!(casted.validity(), &Validity::AllValid);
        assert_eq!(casted.len(), 3);
    }

    #[test]
    fn cast_nullable_to_non_nullable() {
        let decimal_dtype = DecimalDType::new(10, 2);

        // Create nullable array with no nulls
        let array = DecimalArray::new(buffer![100i32, 200, 300], decimal_dtype, Validity::AllValid);

        // Cast to non-nullable
        let non_nullable_dtype = DType::Decimal(decimal_dtype, Nullability::NonNullable);
        let casted = cast(array.as_ref(), &non_nullable_dtype)
            .unwrap()
            .to_decimal();

        assert_eq!(casted.dtype(), &non_nullable_dtype);
        assert_eq!(casted.validity(), &Validity::NonNullable);
    }

    #[test]
    #[should_panic(expected = "Cannot cast array with invalid values to non-nullable type")]
    fn cast_nullable_with_nulls_to_non_nullable_fails() {
        let decimal_dtype = DecimalDType::new(10, 2);

        // Create nullable array with nulls
        let array = DecimalArray::from_option_iter([Some(100i32), None, Some(300)], decimal_dtype);

        // Attempt to cast to non-nullable should fail
        let non_nullable_dtype = DType::Decimal(decimal_dtype, Nullability::NonNullable);
        cast(array.as_ref(), &non_nullable_dtype).unwrap();
    }

    #[test]
    fn cast_different_precision_fails() {
        let array = DecimalArray::new(
            buffer![100i32],
            DecimalDType::new(10, 2),
            Validity::NonNullable,
        );

        // Try to cast to different precision
        let different_dtype = DType::Decimal(DecimalDType::new(15, 3), Nullability::NonNullable);
        let result = cast(array.as_ref(), &different_dtype);

        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Cannot cast decimal(10,2) to decimal(15,3)")
        );
    }

    #[test]
    fn cast_to_non_decimal_returns_err() {
        let array = DecimalArray::new(
            buffer![100i32],
            DecimalDType::new(10, 2),
            Validity::NonNullable,
        );

        // Try to cast to non-decimal type - should fail since no kernel can handle it
        let result = cast(array.as_ref(), &DType::Utf8(Nullability::NonNullable));

        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("No compute kernel to cast")
        );
    }

    #[rstest]
    #[case(DecimalArray::new(buffer![100i32, 200, 300], DecimalDType::new(10, 2), Validity::NonNullable))]
    #[case(DecimalArray::new(buffer![10000i64, 20000, 30000], DecimalDType::new(18, 4), Validity::NonNullable))]
    #[case(DecimalArray::from_option_iter([Some(100i32), None, Some(300)], DecimalDType::new(10, 2)))]
    #[case(DecimalArray::new(buffer![42i32], DecimalDType::new(5, 1), Validity::NonNullable))]
    fn test_cast_decimal_conformance(#[case] array: DecimalArray) {
        test_cast_conformance(array.as_ref());
    }
}
