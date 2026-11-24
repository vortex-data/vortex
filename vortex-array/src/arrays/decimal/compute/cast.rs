// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Decimal casts with checked precision, scale, and nullability.
//!
//! Same-scale casts that increase the canonical width defer the copy through a [`WidenArray`].
//! Scale changes and precision reductions run the conversion kernel to check or rescale valid values.

use vortex_buffer::BufferMut;
use vortex_compute::lane_kernels::IndexedSourceExt;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::ConstantArray;
use crate::arrays::Decimal;
use crate::arrays::DecimalArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::WidenArray;
use crate::arrays::decimal::DecimalArrayExt;
use crate::arrays::primitive::compute::integer_cast::IntegerCast;
use crate::arrays::primitive::compute::integer_cast::IntegerType;
use crate::arrays::primitive::compute::integer_cast::Remainder;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::dtype::DecimalDType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::dtype::i256;
use crate::match_each_decimal_value_type;
use crate::scalar::DecimalToFloatCast;
use crate::scalar::DecimalValue;
use crate::scalar::Scalar;
use crate::scalar_fn::builtin::cast::CastKernel;
use crate::scalar_fn::builtin::cast::CastReduce;
use crate::validity::Validity;

impl CastReduce for Decimal {
    fn cast(array: ArrayView<'_, Decimal>, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        let DType::Decimal(to_decimal_dtype, to_nullability) = dtype else {
            return Ok(None);
        };
        let DType::Decimal(from_decimal_dtype, _) = array.dtype() else {
            vortex_panic!(
                "DecimalArray must have decimal dtype, got {:?}",
                array.dtype()
            );
        };

        let grows_width = grows_canonical_width(*from_decimal_dtype, *to_decimal_dtype);
        if from_decimal_dtype != to_decimal_dtype && !grows_width {
            return Ok(None);
        }

        let Some(new_validity) = array
            .validity()?
            .trivially_cast_nullability(*to_nullability, array.len())?
        else {
            return Ok(None);
        };

        if grows_width {
            return widen_decimal_values(array, dtype, new_validity).map(Some);
        }

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
        let DType::Decimal(from_decimal_dtype, _) = array.dtype() else {
            vortex_panic!(
                "DecimalArray must have decimal dtype, got {:?}",
                array.dtype()
            );
        };
        if let DType::Primitive(ptype, nullability) = dtype
            && ptype.is_int()
        {
            return cast_to_integer(array, *ptype, *nullability, ctx).map(Some);
        }
        if let DType::Primitive(PType::F64, nullability) = dtype {
            let scale = from_decimal_dtype.scale();
            return cast_to_f64(array, scale, *nullability, ctx).map(Some);
        }
        let DType::Decimal(to_decimal_dtype, to_nullability) = dtype else {
            return Ok(None);
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

        if grows_canonical_width(*from_decimal_dtype, *to_decimal_dtype) {
            return widen_decimal_values(array, dtype, new_validity).map(Some);
        }

        // The cast above handles a wider canonical type. Increasing precision within the same
        // width or changing nullability alone leaves the values buffer unchanged.
        if from_decimal_dtype.scale() == to_decimal_dtype.scale()
            && to_decimal_dtype.precision() >= from_decimal_dtype.precision()
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
        if !array.is_empty() && matches!(valid_values, Mask::AllFalse(_)) {
            return Ok(Some(
                ConstantArray::new(Scalar::null(dtype.clone()), array.len()).into_array(),
            ));
        }
        let cast = IntegerCast::to_decimal(
            IntegerType::of_decimal(array.values_type()),
            from_decimal_dtype.scale(),
            *to_decimal_dtype,
        );
        let values = array.buffer_handle().as_host();
        let casted = cast.cast(values, &valid_values).map_err(|index| {
            decimal_cast_error(
                cast.source_type().value_at(values, index),
                *from_decimal_dtype,
                *to_decimal_dtype,
            )
        })?;
        // Every valid value has been checked against the target precision, so skip the
        // constructor's scan.
        Ok(Some(
            DecimalArray::try_new_handle_from_trusted_values(
                BufferHandle::new_host(casted),
                to_decimal_dtype.canonical_values_type(),
                *to_decimal_dtype,
                new_validity,
            )?
            .into_array(),
        ))
    }
}

/// Returns whether a same-scale cast increases the canonical width.
fn grows_canonical_width(from: DecimalDType, to: DecimalDType) -> bool {
    from.scale() == to.scale() && from.canonical_values_type() < to.canonical_values_type()
}

/// Wraps the source values in a [`WidenArray`] with the target dtype.
///
/// The caller must cast `validity` to the target nullability without changing its length or valid
/// rows. The child keeps the source decimal dtype and statistics because its values are unchanged.
fn widen_decimal_values(
    array: ArrayView<'_, Decimal>,
    dtype: &DType,
    validity: Validity,
) -> VortexResult<ArrayRef> {
    let values = DecimalArray::try_new_handle_from_trusted_values(
        array.buffer_handle().clone(),
        array.values_type(),
        array.decimal_dtype(),
        validity,
    )?;
    values.statistics().inherit_from(array.statistics());

    Ok(WidenArray::try_new(values.into_array(), dtype.clone())?.into_array())
}

fn cast_to_integer(
    array: ArrayView<'_, Decimal>,
    ptype: PType,
    nullability: Nullability,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let source_validity = array.validity()?;
    let mask = source_validity.execute_mask(array.len(), ctx)?;
    let validity = source_validity.cast_nullability(nullability, array.len(), ctx)?;
    if !array.is_empty() && matches!(mask, Mask::AllFalse(_)) {
        return Ok(ConstantArray::new(
            Scalar::null(DType::Primitive(ptype, nullability)),
            array.len(),
        )
        .into_array());
    }
    // Dividing by the scale factor truncates toward zero, and values must be in range before
    // truncating, like the scalar cast.
    let scale = array.decimal_dtype().scale();
    let cast = IntegerCast::new(
        IntegerType::of_decimal(array.values_type()),
        IntegerType::of_ptype(ptype),
        -<i16 as From<i8>>::from(scale),
        Remainder::Truncate,
    );
    let values = array.buffer_handle().as_host();
    let casted = cast.cast(values, &mask).map_err(|index| {
        vortex_err!(
            "Decimal value {} at scale {} out of range for {}",
            cast.source_type().value_at(values, index),
            scale,
            ptype
        )
    })?;
    Ok(PrimitiveArray::from_byte_buffer(casted, ptype, validity).into_array())
}

fn cast_to_f64(
    array: ArrayView<'_, Decimal>,
    scale: i8,
    nullability: Nullability,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let source_validity = array.validity()?;
    let n = array.len();
    let mask = source_validity.execute_mask(n, ctx)?;
    let validity = source_validity.cast_nullability(nullability, n, ctx)?;
    if n > 0 && matches!(mask, Mask::AllFalse(_)) {
        return Ok(
            ConstantArray::new(Scalar::null(DType::Primitive(PType::F64, nullability)), n)
                .into_array(),
        );
    }

    let cast = DecimalToFloatCast::new(scale);

    let buffer = match &mask {
        Mask::AllFalse(_) => BufferMut::<f64>::zeroed(n),
        Mask::AllTrue(_) => match_each_decimal_value_type!(array.values_type(), |F| {
            let values = array.buffer::<F>();
            let values = values.as_slice();
            let mut out = BufferMut::<f64>::with_capacity(n);
            values.map_into(&mut out.spare_capacity_mut()[..n], |v: F| cast.cast(v));
            // SAFETY: map_into wrote every lane before returning.
            unsafe { out.set_len(n) };
            out
        }),
        Mask::Values(mask_values) => match_each_decimal_value_type!(array.values_type(), |F| {
            let values = array.buffer::<F>();
            let values = values.as_slice();
            let mut out = BufferMut::<f64>::with_capacity(n);
            let write_result = values.try_map_masked_into(
                mask_values.bit_buffer(),
                &mut out.spare_capacity_mut()[..n],
                |v: F| Some(cast.cast(v)),
            );
            debug_assert!(write_result.is_ok());
            // SAFETY: try_map_masked_into wrote every lane before returning Ok.
            unsafe { out.set_len(n) };
            out
        }),
    };

    Ok(PrimitiveArray::new(buffer, validity).into_array())
}

#[cold]
#[inline(never)]
fn decimal_cast_error(
    value: i256,
    from_decimal_dtype: DecimalDType,
    to_decimal_dtype: DecimalDType,
) -> VortexError {
    match DecimalValue::I256(value).cast_decimal(from_decimal_dtype, to_decimal_dtype) {
        Ok(_) => {
            // The fast path only rejects values the slow path also rejects, so this arm should
            // be unreachable. If it is hit, the fast and slow paths have drifted and we are
            // erroring on a value that is actually representable.
            debug_assert!(
                false,
                "decimal fast-path cast rejected value {value} that the slow path accepts \
                 (from {from_decimal_dtype} to {to_decimal_dtype})"
            );
            vortex_err!(
                "decimal value cannot be represented after casting from {} to {}",
                from_decimal_dtype,
                to_decimal_dtype
            )
        }
        Err(error) => error,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::ArrayRef;
    use crate::Canonical;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::DecimalArray;
    use crate::arrays::PrimitiveArray;
    use crate::assert_arrays_eq;
    use crate::builtins::ArrayBuiltins;
    use crate::compute::conformance::cast::test_cast_conformance;
    use crate::dtype::DType;
    use crate::dtype::DecimalDType;
    use crate::dtype::DecimalType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::dtype::i256;
    use crate::match_each_integer_ptype;
    use crate::scalar::Scalar;
    use crate::validity::Validity;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    #[rstest]
    #[case::workflow(0, vec![0, 1], vec![0, 1])]
    #[case::fractional(1, vec![-199, -1, 0, 1, 199], vec![-19, 0, 0, 0, 19])]
    #[case::negative_scale(-2, vec![-12, 0, 12], vec![-1200, 0, 1200])]
    #[case::large_exact(
        0,
        vec![1_786_639_777_684_000_001],
        vec![1_786_639_777_684_000_001]
    )]
    #[case::large_scale(76, vec![-1, 1], vec![0, 0])]
    #[case::zero_extreme_scale(-128, vec![0, 0], vec![0, 0])]
    fn cast_decimal_to_integer_policy(
        #[case] scale: i8,
        #[case] values: Vec<i128>,
        #[case] expected: Vec<i64>,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let decimal_dtype = DecimalDType::new(76, scale);
        let array = DecimalArray::new(
            values.iter().copied().map(i256::from_i128).collect(),
            decimal_dtype,
            Validity::NonNullable,
            &mut ctx,
        );
        let target = DType::Primitive(PType::I64, Nullability::NonNullable);
        let casted = array.into_array().cast(target.clone())?;
        assert_arrays_eq!(
            casted,
            PrimitiveArray::from_iter(expected.iter().copied()),
            &mut ctx
        );
        for (value, expected) in values.into_iter().zip(expected) {
            let scalar = Scalar::decimal(value.into(), decimal_dtype, Nullability::NonNullable);
            assert_eq!(scalar.cast(&target)?, Scalar::from(expected));
        }
        Ok(())
    }

    #[rstest]
    #[case::exact_scale(2)]
    #[case::inexact_scale(30)]
    #[case::past_i128_scale(39)]
    #[case::max_scale(76)]
    #[case::negative_scale(-5)]
    #[case::extreme_negative_scale(-128)]
    fn cast_decimal_to_f64_matches_scalar_cast(#[case] scale: i8) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let decimal_dtype = DecimalDType::new(76, scale);
        let beyond_i128 = i256::from_i128(i128::MAX) + i256::ONE;
        let values = vec![
            i256::ZERO,
            i256::from_i128(12345),
            i256::from_i128(-7),
            i256::from_i128(i64::MAX.into()) + i256::ONE,
            beyond_i128,
            -beyond_i128,
        ];
        let array = DecimalArray::new(
            values.iter().copied().collect(),
            decimal_dtype,
            Validity::NonNullable,
            &mut ctx,
        );

        let target = DType::Primitive(PType::F64, Nullability::NonNullable);
        let casted = array
            .into_array()
            .cast(target.clone())?
            .execute::<PrimitiveArray>(&mut ctx)?;

        assert_eq!(casted.len(), values.len());
        assert_eq!(casted.dtype(), &target);
        for (value, array_value) in values.into_iter().zip(casted.as_slice::<f64>()) {
            let scalar = Scalar::decimal(value.into(), decimal_dtype, Nullability::NonNullable);
            let scalar_value = f64::try_from(&scalar.cast(&target)?)?;
            assert!(
                !array_value.is_nan(),
                "{value} at scale {scale} cast to NaN"
            );
            assert_eq!(
                array_value.to_bits(),
                scalar_value.to_bits(),
                "{value} at scale {scale}: array {array_value}, scalar {scalar_value}"
            );
        }

        Ok(())
    }

    #[rstest]
    #[case::i8(2, 99, 9)]
    #[case::i16(4, 123, 12)]
    #[case::i32(9, 123, 12)]
    #[case::i64(18, 123, 12)]
    #[case::i128(38, 123, 12)]
    #[case::i256(76, 123, 12)]
    fn cast_decimal_to_integer_storage_and_nulls(
        #[case] precision: u8,
        #[case] value: i8,
        #[case] truncated: i8,
        #[values(
            PType::I8, PType::I16, PType::I32, PType::I64, PType::U8, PType::U16, PType::U32,
            PType::U64
        )]
        target: PType,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let decimal_dtype = DecimalDType::new(precision, 1);
        let array = DecimalArray::from_option_iter([Some(19i8), None, Some(value)], decimal_dtype);
        let dtype = DType::Primitive(target, Nullability::Nullable);
        let casted = array.into_array().cast(dtype.clone())?;
        match_each_integer_ptype!(target, |T| {
            assert_arrays_eq!(
                casted,
                PrimitiveArray::from_option_iter([Some(1 as T), None, Some(truncated as T)]),
                &mut ctx
            );
            for (value, expected) in [(19i8, 1 as T), (value, truncated as T)] {
                let scalar = Scalar::decimal(value.into(), decimal_dtype, Nullability::Nullable);
                assert_eq!(
                    scalar.cast(&dtype)?,
                    Scalar::primitive(expected, Nullability::Nullable)
                );
            }
        });
        Ok(())
    }

    #[rstest]
    fn cast_decimal_to_integer_bounds(
        #[values(
            PType::I8, PType::I16, PType::I32, PType::I64, PType::U8, PType::U16, PType::U32,
            PType::U64
        )]
        target: PType,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let decimal_dtype = DecimalDType::new(76, 1);
        let dtype = DType::Primitive(target, Nullability::NonNullable);
        match_each_integer_ptype!(target, |T| {
            let ten = i256::from_i128(10);
            let minimum = i256::from_i128(T::MIN as i128) * ten;
            let maximum = i256::from_i128(T::MAX as i128) * ten;
            let array = DecimalArray::from_iter([minimum, maximum], decimal_dtype);
            let casted = array.into_array().cast(dtype.clone())?;
            assert_arrays_eq!(
                casted,
                PrimitiveArray::from_iter([T::MIN, T::MAX]),
                &mut ctx
            );
            for (value, expected) in [(minimum, T::MIN), (maximum, T::MAX)] {
                let scalar = Scalar::decimal(value.into(), decimal_dtype, Nullability::NonNullable);
                assert_eq!(
                    scalar.cast(&dtype)?,
                    Scalar::primitive(expected, Nullability::NonNullable)
                );
            }
            // The scalar policy rejects values outside the range even when truncation would fit.
            for value in [
                minimum - i256::ONE,
                maximum + i256::ONE,
                minimum - ten,
                maximum + ten,
            ] {
                let array = DecimalArray::from_iter([i256::ZERO, value], decimal_dtype);
                let error = array
                    .into_array()
                    .cast(dtype.clone())?
                    .execute::<PrimitiveArray>(&mut ctx)
                    .unwrap_err();
                assert!(error.to_string().contains("out of range"));
                let scalar = Scalar::decimal(value.into(), decimal_dtype, Nullability::NonNullable);
                assert!(scalar.cast(&dtype).is_err());
            }
        });
        Ok(())
    }

    #[test]
    fn cast_decimal_to_integer_masks_and_empty() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let decimal_dtype = DecimalDType::new(21, 0);
        let target = DType::Primitive(PType::I8, Nullability::Nullable);
        for validity in [Validity::from_iter([false, true]), Validity::AllInvalid] {
            let expected = PrimitiveArray::new(buffer![0i8, 12], validity.clone());
            let array = DecimalArray::new(buffer![999i128, 12], decimal_dtype, validity, &mut ctx);
            let casted = array.into_array().cast(target.clone())?;
            assert_arrays_eq!(casted, expected, &mut ctx);
        }
        let array = DecimalArray::new(
            buffer![999i128, 12],
            decimal_dtype,
            Validity::from_iter([false, true]),
            &mut ctx,
        );
        assert!(
            array
                .into_array()
                .cast(target.as_nonnullable())?
                .execute::<PrimitiveArray>(&mut ctx)
                .is_err()
        );
        let empty = DecimalArray::from_option_iter([] as [Option<i128>; 0], decimal_dtype);
        let casted = empty.into_array().cast(target.as_nonnullable())?;
        assert_arrays_eq!(casted, PrimitiveArray::from_iter([] as [i8; 0]), &mut ctx);
        Ok(())
    }

    #[test]
    fn cast_decimal_to_integer_wide_storage() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let factor = i256::from_i128(10).checked_pow(40).unwrap();
        let value = i256::from_i128(123) * factor + i256::ONE;
        let decimal_dtype = DecimalDType::new(76, 40);
        let target = DType::Primitive(PType::I64, Nullability::NonNullable);
        let array = DecimalArray::from_iter([value, -value], decimal_dtype);
        let casted = array.into_array().cast(target.clone())?;
        assert_arrays_eq!(casted, PrimitiveArray::from_iter([123i64, -123]), &mut ctx);
        for (value, expected) in [(value, 123i64), (-value, -123)] {
            let scalar = Scalar::decimal(value.into(), decimal_dtype, Nullability::NonNullable);
            assert_eq!(scalar.cast(&target)?, Scalar::from(expected));
        }
        let huge = DecimalArray::from_iter([1i8], DecimalDType::new(1, -128));
        assert!(
            huge.into_array()
                .cast(target)?
                .execute::<PrimitiveArray>(&mut ctx)
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn cast_decimal_to_nullable() {
        let mut ctx = SESSION.create_execution_ctx();
        let decimal_dtype = DecimalDType::new(10, 2);
        let array = DecimalArray::from_iter([100i64, 200, 300], decimal_dtype);

        // Cast to nullable
        let nullable_dtype = DType::Decimal(decimal_dtype, Nullability::Nullable);
        let casted = array
            .into_array()
            .cast(nullable_dtype.clone())
            .unwrap()
            .execute::<DecimalArray>(&mut ctx)
            .unwrap();

        assert_eq!(casted.dtype(), &nullable_dtype);
        assert!(matches!(casted.validity(), Ok(Validity::AllValid)));
        assert_eq!(casted.len(), 3);
        assert_arrays_eq!(
            casted,
            DecimalArray::new(
                buffer![100i64, 200, 300],
                decimal_dtype,
                Validity::AllValid,
                &mut ctx
            ),
            &mut ctx
        );
    }

    #[test]
    fn cast_nullable_to_non_nullable() {
        let mut ctx = SESSION.create_execution_ctx();
        let decimal_dtype = DecimalDType::new(10, 2);

        // Create nullable array with no nulls
        let array = DecimalArray::new(
            buffer![100i64, 200, 300],
            decimal_dtype,
            Validity::AllValid,
            &mut ctx,
        );

        // Cast to non-nullable
        let non_nullable_dtype = DType::Decimal(decimal_dtype, Nullability::NonNullable);
        let casted = array
            .into_array()
            .cast(non_nullable_dtype.clone())
            .unwrap()
            .execute::<DecimalArray>(&mut ctx)
            .unwrap();

        assert_eq!(casted.dtype(), &non_nullable_dtype);
        assert!(matches!(casted.validity(), Ok(Validity::NonNullable)));
        assert_arrays_eq!(
            casted,
            DecimalArray::from_iter([100i64, 200, 300], decimal_dtype),
            &mut ctx
        );
    }

    #[test]
    #[should_panic(expected = "Cannot cast array with invalid values to non-nullable type")]
    fn cast_nullable_with_nulls_to_non_nullable_fails() {
        let mut ctx = SESSION.create_execution_ctx();
        let decimal_dtype = DecimalDType::new(10, 2);

        // Create nullable array with nulls
        let array = DecimalArray::from_option_iter([Some(100i32), None, Some(300)], decimal_dtype);

        // Attempt to cast to non-nullable should fail
        let non_nullable_dtype = DType::Decimal(decimal_dtype, Nullability::NonNullable);
        array
            .into_array()
            .cast(non_nullable_dtype)
            .and_then(|a| a.execute::<Canonical>(&mut ctx).map(|c| c.into_array()))
            .unwrap();
    }

    #[test]
    fn cast_different_scale_rescales() {
        let mut ctx = SESSION.create_execution_ctx();
        let array = DecimalArray::from_iter([100i64], DecimalDType::new(10, 2));

        // Cast 1.00 to scale 3, where it is stored as 1000.
        let different_dtype = DType::Decimal(DecimalDType::new(15, 3), Nullability::NonNullable);
        let casted = array
            .into_array()
            .cast(different_dtype)
            .unwrap()
            .execute::<DecimalArray>(&mut ctx)
            .unwrap();

        assert_eq!(casted.precision(), 15);
        assert_eq!(casted.scale(), 3);
        assert_eq!(casted.values_type(), DecimalType::I64);
        assert_eq!(casted.buffer::<i64>().as_ref(), &[1000]);
    }

    #[test]
    fn cast_downcast_precision_succeeds_when_values_fit() {
        let mut ctx = SESSION.create_execution_ctx();
        let array = DecimalArray::from_iter([100i64], DecimalDType::new(18, 2));

        // Downcasting precision is allowed when every value fits.
        let smaller_dtype = DType::Decimal(DecimalDType::new(10, 2), Nullability::NonNullable);
        let casted = array
            .into_array()
            .cast(smaller_dtype)
            .unwrap()
            .execute::<DecimalArray>(&mut ctx)
            .unwrap();

        assert_eq!(casted.precision(), 10);
        assert_eq!(casted.scale(), 2);
        assert_eq!(casted.buffer::<i64>().as_ref(), &[100]);
    }

    #[test]
    fn cast_downcast_precision_checks_values() {
        let mut ctx = SESSION.create_execution_ctx();
        let array = DecimalArray::from_iter([1000i64], DecimalDType::new(18, 0));

        let smaller_dtype = DType::Decimal(DecimalDType::new(3, 0), Nullability::NonNullable);
        let result = array
            .into_array()
            .cast(smaller_dtype)
            .and_then(|a| a.execute::<Canonical>(&mut ctx).map(|c| c.into_array()));

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
        let mut ctx = SESSION.create_execution_ctx();
        let array = DecimalArray::from_iter([123456i64], DecimalDType::new(10, 4));

        let lower_scale_dtype = DType::Decimal(DecimalDType::new(10, 2), Nullability::NonNullable);
        let result = array
            .into_array()
            .cast(lower_scale_dtype)
            .and_then(|a| a.execute::<Canonical>(&mut ctx).map(|c| c.into_array()));

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
        let mut ctx = SESSION.create_execution_ctx();
        let array = DecimalArray::new(
            buffer![100i64, 123456],
            DecimalDType::new(10, 4),
            Validity::from_iter([true, false]),
            &mut ctx,
        );

        let lower_scale_dtype = DType::Decimal(DecimalDType::new(3, 2), Nullability::Nullable);
        let casted = array
            .into_array()
            .cast(lower_scale_dtype)
            .unwrap()
            .execute::<DecimalArray>(&mut ctx)
            .unwrap();

        let mask = casted
            .as_ref()
            .validity()
            .unwrap()
            .execute_mask(casted.as_ref().len(), &mut SESSION.create_execution_ctx())
            .unwrap();
        assert!(mask.value(0));
        assert!(!mask.value(1));
        assert_eq!(casted.buffer::<i16>().as_ref()[0], 1);
    }

    #[test]
    fn cast_to_non_decimal_returns_err() {
        let mut ctx = SESSION.create_execution_ctx();
        let array = DecimalArray::from_iter([100i64], DecimalDType::new(10, 2));

        // Try to cast to non-decimal type - should fail since no kernel can handle it
        let result = array
            .into_array()
            .cast(DType::Utf8(Nullability::NonNullable))
            .and_then(|a| a.execute::<Canonical>(&mut ctx).map(|c| c.into_array()));

        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("No CastKernel to cast canonical array")
        );
    }

    #[rstest]
    #[case(DecimalArray::from_iter([100i64, 200, 300], DecimalDType::new(10, 2)))]
    #[case(DecimalArray::from_iter([10000i64, 20000, 30000], DecimalDType::new(18, 4)))]
    #[case(DecimalArray::from_option_iter([Some(100i32), None, Some(300)], DecimalDType::new(10, 2)))]
    #[case(DecimalArray::from_iter([42i32], DecimalDType::new(5, 1)))]
    fn test_cast_decimal_conformance(#[case] array: DecimalArray) {
        test_cast_conformance(&array.into_array(), &mut SESSION.create_execution_ctx());
    }

    #[test]
    fn cast_decimal_f64() {
        let dtype = DecimalDType::new(6, 2);
        let array = DecimalArray::from_iter([125i32, 250, -375], dtype);
        let target = DType::Primitive(PType::F64, Nullability::NonNullable);
        let casted = array.into_array().cast(target.clone()).unwrap();
        assert_eq!(casted.dtype(), &target);

        let mut ctx = SESSION.create_execution_ctx();
        let primitive = casted
            .execute::<Canonical>(&mut ctx)
            .unwrap()
            .into_primitive();
        assert_eq!(primitive.to_buffer::<f64>().as_ref(), &[1.25, 2.5, -3.75]);
    }

    #[test]
    fn cast_decimal_f64_null() {
        let values = [Some(100i32), None, Some(-200)];
        let array = DecimalArray::from_option_iter(values, DecimalDType::new(6, 2));
        let target = DType::Primitive(PType::F64, Nullability::Nullable);
        let casted = array.into_array().cast(target.clone()).unwrap();
        assert_eq!(casted.dtype(), &target);

        let mut ctx = SESSION.create_execution_ctx();
        let primitive = casted
            .execute::<Canonical>(&mut ctx)
            .unwrap()
            .into_primitive();

        assert!(primitive.is_valid(0, &mut ctx).unwrap());
        assert!(!primitive.is_valid(1, &mut ctx).unwrap());
        assert!(primitive.is_valid(2, &mut ctx).unwrap());

        assert_eq!(
            primitive.execute_scalar(0, &mut ctx).unwrap(),
            Scalar::from(1f64)
        );
        assert_eq!(
            primitive.execute_scalar(2, &mut ctx).unwrap(),
            Scalar::from(-2f64)
        );
    }

    #[test]
    fn cast_decimal_f64_all_null() {
        let mut ctx = SESSION.create_execution_ctx();
        let dtype = DecimalDType::new(6, 2);
        let buf = buffer![i32::MAX, i32::MIN, 12345];
        let array = DecimalArray::new(buf, dtype, Validity::AllInvalid, &mut ctx);
        let target = DType::Primitive(PType::F64, Nullability::Nullable);
        let casted = array.into_array().cast(target.clone()).unwrap();
        assert_eq!(casted.dtype(), &target);

        assert_arrays_eq!(
            casted,
            PrimitiveArray::from_option_iter([None::<f64>; 3]),
            &mut ctx
        );
    }

    /// Casts a decimal array of `values` to `target`, and checks the result against casting each
    /// value as a scalar, which shares no code with the array cast.
    #[rstest]
    // Decimal to integer truncates toward zero, but values must be in range before truncating.
    #[case::to_i8_truncates((5, 2), &[12_700, -12_800, 12_699, -12_799, -5, 5], PType::I8.into(), true)]
    #[case::to_i8_fraction_above_range((5, 2), &[12_701], PType::I8.into(), false)]
    #[case::to_i8_fraction_below_range((5, 2), &[-12_801], PType::I8.into(), false)]
    #[case::to_i8_above_range((5, 2), &[12_800], PType::I8.into(), false)]
    #[case::negative_scale_to_i8((3, -2), &[1, -1], PType::I8.into(), true)]
    #[case::negative_scale_above_i8((3, -2), &[2], PType::I8.into(), false)]
    #[case::negative_scale_to_u8((3, -2), &[2], PType::U8.into(), true)]
    #[case::negative_to_u64((18, 0), &[-1], PType::U64.into(), false)]
    #[case::i256_to_i64((40, 2), &[123_456_789, -5], PType::I64.into(), true)]
    #[case::i8_storage_to_u64((2, 1), &[99, 0], PType::U64.into(), true)]
    // Decimal to decimal rescales exactly and checks the target precision.
    #[case::scale_down((10, 4), &[12_300, -500], decimal(10, 2), true)]
    #[case::scale_down_inexact((10, 4), &[12_345], decimal(10, 2), false)]
    #[case::scale_up_widens((9, 2), &[123_456_789, -1], decimal(18, 4), true)]
    #[case::narrow((18, 0), &[999_999_999, -999_999_999], decimal(9, 0), true)]
    #[case::narrow_above_precision((18, 0), &[1_000_000_000], decimal(9, 0), false)]
    #[case::i128_to_i8((38, 0), &[99, -99], decimal(2, 0), true)]
    #[case::i128_to_i8_above_precision((38, 0), &[100], decimal(2, 0), false)]
    #[case::i8_to_i256((2, 0), &[99, -99], decimal(76, 10), true)]
    #[case::scale_down_beyond_i256_zero((18, 0), &[0], decimal(18, -100), true)]
    #[case::scale_down_beyond_i256((18, 0), &[1], decimal(18, -100), false)]
    fn array_cast_matches_scalar_cast(
        #[case] (precision, scale): (u8, i8),
        #[case] values: &[i64],
        #[case] target: DType,
        #[case] casts: bool,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let from = DecimalDType::new(precision, scale);
        let array = DecimalArray::from_iter(values.iter().copied(), from);
        let expected = (0..values.len())
            .map(|index| array.execute_scalar(index, &mut ctx)?.cast(&target))
            .collect::<VortexResult<Vec<_>>>();
        assert_eq!(
            expected.is_ok(),
            casts,
            "the scalar cast should match the case"
        );
        let casted = array
            .into_array()
            .cast(target)?
            .execute::<ArrayRef>(&mut ctx);
        match expected {
            Ok(expected) => {
                let casted = casted?;
                for (index, expected) in expected.iter().enumerate() {
                    assert_eq!(&casted.execute_scalar(index, &mut ctx)?, expected);
                }
            }
            Err(_) => assert!(casted.is_err(), "the array cast should fail"),
        }
        Ok(())
    }

    fn decimal(precision: u8, scale: i8) -> DType {
        DType::Decimal(
            DecimalDType::new(precision, scale),
            Nullability::NonNullable,
        )
    }

    /// Null lanes may hold values that do not fit the target.
    #[test]
    fn cast_decimal_to_integer_ignores_null_lanes() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let array = DecimalArray::new(
            buffer![100_000i32, 1234, -99_999],
            DecimalDType::new(6, 2),
            Validity::from_iter([false, true, false]),
            &mut ctx,
        );
        let casted = array
            .into_array()
            .cast(DType::Primitive(PType::I8, Nullability::Nullable))?;
        assert_arrays_eq!(
            casted,
            PrimitiveArray::from_option_iter([None, Some(12i8), None]),
            &mut ctx
        );
        Ok(())
    }
}
