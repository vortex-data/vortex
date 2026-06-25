// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::CheckedMul;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_compute::lane_kernels::IndexedSourceExt;
use vortex_error::VortexError;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Decimal;
use crate::arrays::DecimalArray;
use crate::dtype::BigCast;
use crate::dtype::DType;
use crate::dtype::DecimalDType;
use crate::dtype::DecimalType;
use crate::dtype::NativeDecimalType;
use crate::dtype::i256;
use crate::match_each_decimal_value_type;
use crate::scalar::DecimalValue;
use crate::scalar_fn::fns::cast::CastKernel;
use crate::scalar_fn::fns::cast::CastReduce;
use crate::validity::Validity;

impl CastReduce for Decimal {
    fn cast(array: ArrayView<'_, Decimal>, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        // Only nullability changes within the same decimal dtype are reducible without execution.
        // Precision/scale changes need the kernel.
        let DType::Decimal(to_decimal_dtype, to_nullability) = dtype else {
            return Ok(None);
        };
        let DType::Decimal(from_decimal_dtype, _) = array.dtype() else {
            vortex_panic!(
                "DecimalArray must have decimal dtype, got {:?}",
                array.dtype()
            );
        };

        if from_decimal_dtype != to_decimal_dtype {
            return Ok(None);
        }

        let Some(new_validity) = array
            .validity()?
            .trivially_cast_nullability(*to_nullability, array.len())?
        else {
            return Ok(None);
        };

        // SAFETY: validity has the same length, only its nullability tag changes.
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

impl CastKernel for Decimal {
    fn cast(
        array: ArrayView<'_, Decimal>,
        dtype: &DType,
        ctx: &mut ExecutionCtx,
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

        // If the dtype is exactly the same, return self
        if array.dtype() == dtype {
            return Ok(Some(array.array().clone()));
        }

        let validity = array.validity()?;

        // Cast the validity to the new nullability
        let new_validity = validity
            .clone()
            .cast_nullability(*to_nullability, array.len(), ctx)?;

        // Reuse the values buffer untouched when no rescale is required, the target precision
        // only widens (so every value still fits), and the current physical type is already wide
        // enough to hold the target precision. This keeps the common precision-widening cast
        // (and pure nullability changes) zero-copy instead of allocating and re-scanning.
        if from_decimal_dtype.scale() == to_decimal_dtype.scale()
            && to_decimal_dtype.precision() >= from_decimal_dtype.precision()
            && array
                .values_type()
                .is_compatible_decimal_value_type(*to_decimal_dtype)
        {
            // SAFETY: the source values are bit-identical and remain in range for the wider
            // precision, and new_validity has the same length, only its nullability tag changes.
            unsafe {
                return Ok(Some(
                    DecimalArray::new_unchecked_handle(
                        array.buffer_handle().clone(),
                        array.values_type(),
                        *to_decimal_dtype,
                        new_validity,
                    )
                    .into_array(),
                ));
            }
        }

        let valid_values = validity.execute_mask(array.len(), ctx)?;
        let target_values_type = DecimalType::smallest_decimal_value_type(to_decimal_dtype);

        match_each_decimal_value_type!(array.values_type(), |F| {
            match_each_decimal_value_type!(target_values_type, |T| {
                cast_decimal_values::<F, T>(
                    array,
                    *from_decimal_dtype,
                    *to_decimal_dtype,
                    new_validity,
                    &valid_values,
                )
                .map(Some)
            })
        })
    }
}

fn cast_decimal_values<F, T>(
    array: ArrayView<'_, Decimal>,
    from_decimal_dtype: DecimalDType,
    to_decimal_dtype: DecimalDType,
    validity: Validity,
    valid_values: &Mask,
) -> VortexResult<ArrayRef>
where
    F: NativeDecimalType,
    T: NativeDecimalType + CheckedMul,
    DecimalValue: From<F>,
{
    let values = array.buffer::<F>();
    let values = values.as_slice();
    let cast_plan = DecimalCastPlan::<T>::new(from_decimal_dtype, to_decimal_dtype);

    let buffer = match valid_values {
        Mask::AllTrue(_) => {
            let mut buffer = BufferMut::<T>::with_capacity(values.len());
            values
                .try_map_into(&mut buffer.spare_capacity_mut()[..values.len()], |value| {
                    cast_plan.cast(value)
                })
                .map_err(|idx| {
                    decimal_cast_error::<F, T>(values[idx], from_decimal_dtype, to_decimal_dtype)
                })?;
            // SAFETY: try_map_into initializes every lane before returning Ok.
            unsafe { buffer.set_len(values.len()) };
            buffer.freeze()
        }
        Mask::AllFalse(_) => BufferMut::<T>::zeroed(values.len()).freeze(),
        Mask::Values(mask) => {
            let mut buffer = BufferMut::<T>::with_capacity(values.len());
            values
                .try_map_masked_into(
                    mask.bit_buffer(),
                    &mut buffer.spare_capacity_mut()[..values.len()],
                    |value| cast_plan.cast(value),
                )
                .map_err(|idx| {
                    decimal_cast_error::<F, T>(values[idx], from_decimal_dtype, to_decimal_dtype)
                })?;
            // SAFETY: try_map_masked_into initializes every lane before returning Ok.
            unsafe { buffer.set_len(values.len()) };
            buffer.freeze()
        }
    };

    Ok(DecimalArray::new(buffer, to_decimal_dtype, validity).into_array())
}

#[cold]
fn decimal_cast_error<F, T>(
    value: F,
    from_decimal_dtype: DecimalDType,
    to_decimal_dtype: DecimalDType,
) -> VortexError
where
    F: NativeDecimalType,
    T: NativeDecimalType,
    DecimalValue: From<F>,
{
    match DecimalValue::from(value)
        .cast_decimal(from_decimal_dtype, to_decimal_dtype)
        .and_then(|value| {
            value.cast::<T>().ok_or_else(|| {
                vortex_err!(
                    "decimal value cannot be represented as {} after casting to {}",
                    T::DECIMAL_TYPE,
                    to_decimal_dtype
                )
            })
        }) {
        Ok(_) => {
            // The fast path only returns `None` for values the slow path also rejects, so this
            // arm should be unreachable. If it is hit, the fast and slow paths have drifted and
            // we are erroring on a value that is actually representable.
            debug_assert!(
                false,
                "decimal fast-path cast rejected value {value} that the slow path accepts \
                 (from {from_decimal_dtype} to {to_decimal_dtype})"
            );
            vortex_err!(
                "decimal value cannot be represented as {} after casting from {} to {}",
                T::DECIMAL_TYPE,
                from_decimal_dtype,
                to_decimal_dtype
            )
        }
        Err(error) => error,
    }
}

#[derive(Debug, Clone, Copy)]
enum DecimalCastPlan<T> {
    SameScale { min: T, max: T },
    ScaleUp { factor: T, min: T, max: T },
    ScaleUpOverflow,
    ScaleDown { factor: i256, min: i256, max: i256 },
    ScaleDownOverflow,
}

impl<T> DecimalCastPlan<T>
where
    T: NativeDecimalType + CheckedMul,
{
    fn new(from_decimal_dtype: DecimalDType, to_decimal_dtype: DecimalDType) -> Self {
        let scale_delta = to_decimal_dtype.scale() as i16 - from_decimal_dtype.scale() as i16;
        if scale_delta == 0 {
            let (min, max) = decimal_precision_range::<T>(to_decimal_dtype);
            return Self::SameScale { min, max };
        }

        if scale_delta > 0 {
            let Some(factor) = decimal_scale_factor::<T>(scale_delta as u32) else {
                return Self::ScaleUpOverflow;
            };
            let (min, max) = decimal_precision_range::<T>(to_decimal_dtype);
            return Self::ScaleUp { factor, min, max };
        }

        let Some(factor) = decimal_scale_factor::<i256>((-scale_delta) as u32) else {
            return Self::ScaleDownOverflow;
        };
        let (min, max) = decimal_precision_range::<i256>(to_decimal_dtype);
        Self::ScaleDown { factor, min, max }
    }

    #[inline]
    fn cast<F>(&self, value: F) -> Option<T>
    where
        F: NativeDecimalType,
    {
        match *self {
            DecimalCastPlan::SameScale { min, max } => {
                let value = <T as BigCast>::from(value)?;
                (value >= min && value <= max).then_some(value)
            }
            DecimalCastPlan::ScaleUp { factor, min, max } => {
                let value = <T as BigCast>::from(value)?;
                let value = value.checked_mul(&factor)?;
                (value >= min && value <= max).then_some(value)
            }
            DecimalCastPlan::ScaleUpOverflow | DecimalCastPlan::ScaleDownOverflow => {
                (value == F::default()).then_some(T::default())
            }
            DecimalCastPlan::ScaleDown { factor, min, max } => {
                let value = <i256 as BigCast>::from(value)?;
                if value == i256::ZERO {
                    return Some(T::default());
                }
                if value % factor != i256::ZERO {
                    return None;
                }

                let value = value / factor;
                if value < min || value > max {
                    return None;
                }
                <T as BigCast>::from(value)
            }
        }
    }
}

fn decimal_precision_range<T: NativeDecimalType>(decimal_dtype: DecimalDType) -> (T, T) {
    let precision = usize::from(decimal_dtype.precision());
    (
        T::MIN_BY_PRECISION[precision],
        T::MAX_BY_PRECISION[precision],
    )
}

fn decimal_scale_factor<T>(exp: u32) -> Option<T>
where
    T: NativeDecimalType + CheckedMul,
{
    let ten = <T as BigCast>::from(10_i8)?;
    let mut factor = <T as BigCast>::from(1_i8)?;
    for _ in 0..exp {
        factor = factor.checked_mul(&ten)?;
    }
    Some(factor)
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
    let validity = array.validity()?;

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
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::DecimalArray;
    use crate::builtins::ArrayBuiltins;
    #[expect(deprecated)]
    use crate::canonical::ToCanonical as _;
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
        #[expect(deprecated)]
        let casted = array
            .into_array()
            .cast(nullable_dtype.clone())
            .unwrap()
            .to_decimal();

        assert_eq!(casted.dtype(), &nullable_dtype);
        assert!(matches!(casted.validity(), Ok(Validity::AllValid)));
        assert_eq!(casted.len(), 3);
    }

    #[test]
    fn cast_nullable_to_non_nullable() {
        let decimal_dtype = DecimalDType::new(10, 2);

        // Create nullable array with no nulls
        let array = DecimalArray::new(buffer![100i32, 200, 300], decimal_dtype, Validity::AllValid);

        // Cast to non-nullable
        let non_nullable_dtype = DType::Decimal(decimal_dtype, Nullability::NonNullable);
        #[expect(deprecated)]
        let casted = array
            .into_array()
            .cast(non_nullable_dtype.clone())
            .unwrap()
            .to_decimal();

        assert_eq!(casted.dtype(), &non_nullable_dtype);
        assert!(matches!(casted.validity(), Ok(Validity::NonNullable)));
    }

    #[test]
    #[should_panic(expected = "Cannot cast array with invalid values to non-nullable type")]
    fn cast_nullable_with_nulls_to_non_nullable_fails() {
        let decimal_dtype = DecimalDType::new(10, 2);

        // Create nullable array with nulls
        let array = DecimalArray::from_option_iter([Some(100i32), None, Some(300)], decimal_dtype);

        // Attempt to cast to non-nullable should fail
        let non_nullable_dtype = DType::Decimal(decimal_dtype, Nullability::NonNullable);
        #[expect(deprecated)]
        let result = array
            .into_array()
            .cast(non_nullable_dtype)
            .and_then(|a| a.to_canonical().map(|c| c.into_array()));
        result.unwrap();
    }

    #[test]
    fn cast_different_scale_rescales() {
        let array = DecimalArray::new(
            buffer![100i32],
            DecimalDType::new(10, 2),
            Validity::NonNullable,
        );

        // Cast 1.00 to scale 3, where it is stored as 1000.
        let different_dtype = DType::Decimal(DecimalDType::new(15, 3), Nullability::NonNullable);
        #[expect(deprecated)]
        let casted = array
            .into_array()
            .cast(different_dtype)
            .unwrap()
            .to_decimal();

        assert_eq!(casted.precision(), 15);
        assert_eq!(casted.scale(), 3);
        assert_eq!(casted.values_type(), DecimalType::I64);
        assert_eq!(casted.buffer::<i64>().as_ref(), &[1000]);
    }

    #[test]
    fn cast_downcast_precision_succeeds_when_values_fit() {
        let array = DecimalArray::new(
            buffer![100i64],
            DecimalDType::new(18, 2),
            Validity::NonNullable,
        );

        // Downcasting precision is allowed when every value fits.
        let smaller_dtype = DType::Decimal(DecimalDType::new(10, 2), Nullability::NonNullable);
        #[expect(deprecated)]
        let casted = array.into_array().cast(smaller_dtype).unwrap().to_decimal();

        assert_eq!(casted.precision(), 10);
        assert_eq!(casted.scale(), 2);
        assert_eq!(casted.buffer::<i64>().as_ref(), &[100]);
    }

    #[test]
    fn cast_downcast_precision_checks_values() {
        let array = DecimalArray::new(
            buffer![1000i64],
            DecimalDType::new(18, 0),
            Validity::NonNullable,
        );

        let smaller_dtype = DType::Decimal(DecimalDType::new(3, 0), Nullability::NonNullable);
        #[expect(deprecated)]
        let result = array
            .into_array()
            .cast(smaller_dtype)
            .and_then(|a| a.to_canonical().map(|c| c.into_array()));

        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("does not fit in precision")
        );
    }

    #[test]
    fn cast_lower_scale_requires_exact_rescale() {
        let array = DecimalArray::new(
            buffer![123456i64],
            DecimalDType::new(10, 4),
            Validity::NonNullable,
        );

        let lower_scale_dtype = DType::Decimal(DecimalDType::new(10, 2), Nullability::NonNullable);
        #[expect(deprecated)]
        let result = array
            .into_array()
            .cast(lower_scale_dtype)
            .and_then(|a| a.to_canonical().map(|c| c.into_array()));

        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("would lose precision")
        );
    }

    #[test]
    fn cast_lower_scale_ignores_null_lane_failures() {
        let array = DecimalArray::new(
            buffer![100i64, 123456],
            DecimalDType::new(10, 4),
            Validity::from_iter([true, false]),
        );

        let lower_scale_dtype = DType::Decimal(DecimalDType::new(3, 2), Nullability::Nullable);
        #[expect(deprecated)]
        let casted = array
            .into_array()
            .cast(lower_scale_dtype)
            .unwrap()
            .to_decimal();

        let mask = casted
            .as_ref()
            .validity()
            .unwrap()
            .execute_mask(
                casted.as_ref().len(),
                &mut array_session().create_execution_ctx(),
            )
            .unwrap();
        assert!(mask.value(0));
        assert!(!mask.value(1));
        assert_eq!(casted.buffer::<i16>().as_ref()[0], 1);
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
        #[expect(deprecated)]
        let casted = array.into_array().cast(wider_dtype).unwrap().to_decimal();

        assert_eq!(casted.precision(), 38);
        assert_eq!(casted.scale(), 2);
        assert_eq!(casted.len(), 3);
        // Should be stored in i128 now (precision 38 requires i128)
        assert_eq!(casted.values_type(), DecimalType::I128);
    }

    #[test]
    fn cast_widening_same_physical_type_is_zero_copy() {
        // Decimal(10,2) and Decimal(18,2) are both physically i64 with the same scale, so widening
        // the precision must reuse the values buffer rather than allocate and re-scan it.
        let array = DecimalArray::new(
            buffer![100i64, 200, 300],
            DecimalDType::new(10, 2),
            Validity::NonNullable,
        );
        let src_ptr = array.buffer::<i64>().as_ptr();

        let wider_dtype = DType::Decimal(DecimalDType::new(18, 2), Nullability::NonNullable);
        #[expect(deprecated)]
        let casted = array.into_array().cast(wider_dtype).unwrap().to_decimal();

        assert_eq!(casted.precision(), 18);
        assert_eq!(casted.scale(), 2);
        assert_eq!(casted.values_type(), DecimalType::I64);
        assert_eq!(casted.buffer::<i64>().as_ref(), &[100, 200, 300]);
        // The values buffer must be shared with the source (zero-copy), not reallocated.
        assert_eq!(
            casted.buffer::<i64>().as_ptr(),
            src_ptr,
            "precision-widening cast must reuse the source values buffer"
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
        #[expect(deprecated)]
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
        let mask = casted
            .as_ref()
            .validity()
            .unwrap()
            .execute_mask(
                casted.as_ref().len(),
                &mut array_session().create_execution_ctx(),
            )
            .unwrap();
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
