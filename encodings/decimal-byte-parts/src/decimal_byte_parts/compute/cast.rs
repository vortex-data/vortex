// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::scalar_fn::fns::cast::CastKernel;
use vortex_array::scalar_fn::fns::cast::CastReduce;
use vortex_error::VortexResult;

use crate::DecimalByteParts;
use crate::decimal_byte_parts::DecimalBytePartsArrayExt;
use crate::decimal_byte_parts::DecimalBytePartsArraySlotsExt;

/// Rebuilds `array` over its MSP cast to `nullability`.
///
/// The MSP carries the validity of a DBP array, so the cast of the MSP sets the validity of the
/// result. The lower parts are non-nullable and stay as they are.
fn with_msp_nullability(
    array: ArrayView<'_, DecimalByteParts>,
    nullability: Nullability,
) -> VortexResult<ArrayRef> {
    let msp = array.msp();
    let new_msp = msp.cast(msp.dtype().with_nullability(nullability))?;

    Ok(array.with_msp(new_msp)?.into_array())
}

impl CastReduce for DecimalByteParts {
    fn cast(array: ArrayView<'_, Self>, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        // TODO(mk): Support non-nullability changes as well, e.g. precision.
        if !dtype.eq_ignore_nullability(array.dtype()) {
            return Ok(None);
        }

        let nullability = dtype.nullability();
        if array
            .validity()?
            .trivially_cast_nullability(nullability, array.len())?
            .is_none()
        {
            return Ok(None);
        }

        with_msp_nullability(array, nullability).map(Some)
    }
}

impl CastKernel for DecimalByteParts {
    fn cast(
        array: ArrayView<'_, Self>,
        dtype: &DType,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        if !dtype.eq_ignore_nullability(array.dtype()) {
            return Ok(None);
        }

        // The call to `cast_nullability` fails when the target is non-nullable and a row is null.
        // The `Validity` that it returns is discarded, because the MSP stores the validity of a
        // DBP array and the cast of the MSP in `with_msp_nullability` sets the new validity.
        let nullability = dtype.nullability();
        array
            .validity()?
            .cast_nullability(nullability, array.len(), ctx)?;

        with_msp_nullability(array, nullability).map(Some)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::Canonical;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::DecimalArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::builtins::ArrayBuiltins;
    use vortex_array::compute::conformance::cast::test_cast_conformance;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::DecimalDType;
    use vortex_array::dtype::Nullability;
    use vortex_array::scalar_fn::fns::cast::CastKernel;
    use vortex_array::scalar_fn::fns::cast::CastReduce;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;
    use vortex_session::VortexSession;

    use crate::DecimalByteParts;
    use crate::DecimalBytePartsArray;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();
        crate::initialize(&session);
        session
    });

    /// A DBP array at `DecimalDType::new(10, 2)` over the `i32` MSP `[100, 200, 300]` with
    /// `validity`.
    fn fixture(validity: Validity) -> VortexResult<DecimalBytePartsArray> {
        DecimalByteParts::try_new(
            PrimitiveArray::new(buffer![100i32, 200, 300], validity).into_array(),
            DecimalDType::new(10, 2),
        )
    }

    #[rstest]
    #[case::with_nulls(Validity::Array(BoolArray::from_iter([true, false, true]).into_array()))]
    #[case::without_nulls(Validity::Array(BoolArray::from_iter([true, true, true]).into_array()))]
    #[case::only_nulls(Validity::Array(BoolArray::from_iter([false, false, false]).into_array()))]
    fn test_cast_rule_declines_validity_array(#[case] validity: Validity) -> VortexResult<()> {
        let array = fixture(validity)?;
        assert!(matches!(array.validity()?, Validity::Array(_)));

        let dtype = DType::Decimal(DecimalDType::new(10, 2), Nullability::NonNullable);
        assert!(<DecimalByteParts as CastReduce>::cast(array.as_view(), &dtype)?.is_none());

        Ok(())
    }

    #[rstest]
    #[case::with_nulls(Validity::Array(BoolArray::from_iter([true, false, true]).into_array()))]
    #[case::only_nulls(Validity::Array(BoolArray::from_iter([false, false, false]).into_array()))]
    fn test_cast_kernel_rejects_nulls(#[case] validity: Validity) -> VortexResult<()> {
        const NULL_ROW_ERROR: &str = "Cannot cast array with invalid values to non-nullable type.";

        let array = fixture(validity)?;
        assert!(matches!(array.validity()?, Validity::Array(_)));

        let mut ctx = SESSION.create_execution_ctx();
        let dtype = DType::Decimal(DecimalDType::new(10, 2), Nullability::NonNullable);
        let kernel_error =
            <DecimalByteParts as CastKernel>::cast(array.as_view(), &dtype, &mut ctx)
                .expect_err("the kernel rejects a cast to non-nullable when a row is null");
        assert!(kernel_error.to_string().contains(NULL_ROW_ERROR));

        let casted = array.into_array().cast(dtype)?;
        let execute_error = casted
            .execute::<Canonical>(&mut ctx)
            .expect_err("execution rejects a cast to non-nullable when a row is null");
        assert!(execute_error.to_string().contains(NULL_ROW_ERROR));

        Ok(())
    }

    #[test]
    fn test_cast_kernel_without_nulls() -> VortexResult<()> {
        let validity = Validity::Array(BoolArray::from_iter([true, true, true]).into_array());
        let array = fixture(validity)?;
        assert!(matches!(array.validity()?, Validity::Array(_)));

        let mut ctx = SESSION.create_execution_ctx();
        let decimal_dtype = DecimalDType::new(10, 2);
        let dtype = DType::Decimal(decimal_dtype, Nullability::NonNullable);
        let kernel_result =
            <DecimalByteParts as CastKernel>::cast(array.as_view(), &dtype, &mut ctx)?.ok_or_else(
                || vortex_err!("the kernel must handle a cast that changes only the nullability"),
            )?;
        assert!(kernel_result.is::<DecimalByteParts>());
        assert_eq!(kernel_result.dtype(), &dtype);

        let casted = array.into_array().cast(dtype)?;
        assert_arrays_eq!(
            casted,
            DecimalArray::new(
                buffer![100i32, 200, 300],
                decimal_dtype,
                Validity::NonNullable
            ),
            &mut ctx
        );

        Ok(())
    }

    #[test]
    fn test_cast_rule_rejects_all_invalid() -> VortexResult<()> {
        let array = fixture(Validity::AllInvalid)?;

        let dtype = DType::Decimal(DecimalDType::new(10, 2), Nullability::NonNullable);
        let error = array
            .into_array()
            .cast(dtype)
            .expect_err("the rule rejects a cast of all nulls to non-nullable");
        assert!(
            error
                .to_string()
                .contains("Cannot cast AllInvalid to NonNullable")
        );

        Ok(())
    }

    #[rstest]
    #[case::nullable_all_valid_to_non_nullable(Validity::AllValid, Nullability::NonNullable)]
    #[case::non_nullable_to_nullable(Validity::NonNullable, Nullability::Nullable)]
    fn test_cast_rule_reduces(
        #[case] validity: Validity,
        #[case] target: Nullability,
    ) -> VortexResult<()> {
        let decimal_dtype = DecimalDType::new(10, 2);
        let casted = fixture(validity)?
            .into_array()
            .cast(DType::Decimal(decimal_dtype, target))?;
        assert!(casted.is::<DecimalByteParts>());

        let mut ctx = SESSION.create_execution_ctx();
        assert_arrays_eq!(
            casted,
            DecimalArray::new(
                buffer![100i32, 200, 300],
                decimal_dtype,
                Validity::from(target)
            ),
            &mut ctx
        );

        Ok(())
    }

    #[test]
    fn test_cast_kernel_keeps_lower_parts() -> VortexResult<()> {
        let validity = Validity::Array(BoolArray::from_iter([true, true, true]).into_array());
        let msp = PrimitiveArray::new(buffer![1i64, 0, -1], validity).into_array();
        let lower_parts = vec![buffer![5u64, 6, 7].into_array()];
        let decimal_dtype = DecimalDType::new(38, 2);
        let array = DecimalByteParts::try_new_with_lower_parts(msp, lower_parts, decimal_dtype)?;
        assert!(matches!(array.validity()?, Validity::Array(_)));

        let mut ctx = SESSION.create_execution_ctx();
        let dtype = DType::Decimal(decimal_dtype, Nullability::NonNullable);
        let kernel_result =
            <DecimalByteParts as CastKernel>::cast(array.as_view(), &dtype, &mut ctx)?.ok_or_else(
                || vortex_err!("the kernel must handle a cast that changes only the nullability"),
            )?;
        assert!(kernel_result.is::<DecimalByteParts>());
        assert_eq!(kernel_result.dtype(), &dtype);

        let casted = array.into_array().cast(dtype)?;
        assert_arrays_eq!(
            casted,
            DecimalArray::new(
                buffer![(1i128 << 64) + 5, 6, (-1i128 << 64) + 7],
                decimal_dtype,
                Validity::NonNullable
            ),
            &mut ctx
        );

        Ok(())
    }

    #[test]
    fn test_cast_decimal_byte_parts_nullability() {
        let mut ctx = SESSION.create_execution_ctx();
        let decimal_dtype = DecimalDType::new(10, 2);
        let array =
            DecimalByteParts::try_new(buffer![100i32, 200, 300, 400].into_array(), decimal_dtype)
                .unwrap();

        // Cast to nullable decimal
        let casted = array
            .into_array()
            .cast(DType::Decimal(decimal_dtype, Nullability::Nullable))
            .unwrap();
        assert_eq!(
            casted.dtype(),
            &DType::Decimal(decimal_dtype, Nullability::Nullable)
        );

        // Verify the values are preserved
        assert_arrays_eq!(
            casted,
            DecimalArray::new(
                buffer![100i32, 200, 300, 400],
                decimal_dtype,
                Validity::AllValid
            ),
            &mut ctx
        );
    }

    #[test]
    fn test_cast_decimal_byte_parts_nullable_to_non_nullable() {
        let mut ctx = SESSION.create_execution_ctx();
        let decimal_dtype = DecimalDType::new(10, 2);
        let array = DecimalByteParts::try_new(
            PrimitiveArray::from_option_iter([Some(100i32), None, Some(300)]).into_array(),
            decimal_dtype,
        )
        .unwrap();

        // Cast to non-nullable should fail due to nulls - force evaluation via execute::<Canonical>
        let result = array
            .into_array()
            .cast(DType::Decimal(decimal_dtype, Nullability::NonNullable))
            .and_then(|a| a.execute::<Canonical>(&mut ctx).map(|c| c.into_array()));
        assert!(result.is_err());
    }

    #[rstest]
    #[case::i32(DecimalByteParts::try_new(
        buffer![100i32, 200, 300, 400, 500].into_array(),
        DecimalDType::new(10, 2),
    ).unwrap())]
    #[case::i64(DecimalByteParts::try_new(
        buffer![1000i64, 2000, 3000, 4000].into_array(),
        DecimalDType::new(19, 4),
    ).unwrap())]
    #[case::nullable(DecimalByteParts::try_new(
        PrimitiveArray::from_option_iter([Some(100i32), None, Some(300), Some(400), None])
            .into_array(),
        DecimalDType::new(10, 2),
    ).unwrap())]
    #[case::single(DecimalByteParts::try_new(
        buffer![42i32].into_array(),
        DecimalDType::new(5, 1),
    ).unwrap())]
    #[case::negative(DecimalByteParts::try_new(
        buffer![-100i32, -200, 300, -400, 500].into_array(),
        DecimalDType::new(10, 2),
    ).unwrap())]
    fn test_cast_decimal_byte_parts_conformance(#[case] array: DecimalBytePartsArray) {
        test_cast_conformance(&array.into_array(), &mut SESSION.create_execution_ctx());
    }
}
