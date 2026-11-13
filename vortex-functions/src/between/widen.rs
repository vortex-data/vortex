// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::arrays::Constant;
use vortex_array::arrays::Widen;
use vortex_array::arrays::widen::WidenArraySlotsExt;
use vortex_array::arrays::widen::compute::constant_fits_values;
use vortex_array::arrays::widen::compute::constant_in_values_dtype;
use vortex_error::VortexResult;

use crate::between::BetweenOptions;
use crate::between::BetweenReduce;
use crate::between::as_two_compares;
use crate::builtins::ArrayFunctions;

impl BetweenReduce for Widen {
    fn between(
        array: ArrayView<'_, Self>,
        lower: &ArrayRef,
        upper: &ArrayRef,
        options: &BetweenOptions,
    ) -> VortexResult<Option<ArrayRef>> {
        let values = array.values();

        if let (Some(lower), Some(upper)) = (lower.as_constant(), upper.as_constant())
            && constant_fits_values(lower, values.dtype()) == Some(true)
            && constant_fits_values(upper, values.dtype()) == Some(true)
        {
            let lower = constant_in_values_dtype(lower, values, array.len())?;
            let upper = constant_in_values_dtype(upper, values, array.len())?;
            return values
                .clone()
                .between(lower, upper, options.clone())
                .map(Some);
        }

        // Each compare of a `Widen` with a constant or another `Widen` runs the compare kernel of
        // `Widen`. Any other bound declines, so that `Between` decodes the `Widen` once.
        let is_operand = |bound: &ArrayRef| bound.is::<Constant>() || bound.is::<Widen>();
        if !is_operand(lower) || !is_operand(upper) {
            return Ok(None);
        }

        as_two_compares(array.array(), lower, upper, options).map(Some)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::Canonical;
    use vortex_array::ExecutionCtx;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::DecimalArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::WidenArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::DecimalDType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::scalar::DecimalValue;
    use vortex_array::scalar::Scalar;
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::between::BetweenOptions;
    use crate::between::StrictComparison;
    use crate::builtins::ArrayFunctions;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();
        vortex_array::initialize(&session);
        crate::initialize(&session);
        session
    });

    /// Returns `array` executed to its canonical form.
    fn canonical(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
        Ok(array.clone().execute::<Canonical>(ctx)?.into_array())
    }

    /// Returns a `Widen` of the `logical` type over `values`, with the nullability of `values`.
    fn widen(values: impl IntoArray, logical: PType) -> WidenArray {
        let values = values.into_array();
        let dtype = DType::Primitive(logical, values.dtype().nullability());
        WidenArray::try_new(values, dtype).vortex_expect("fixture values are narrower integers")
    }

    /// Returns a `Widen` of the `parent` decimal dtype over `values`, with their nullability.
    fn widen_decimal(values: impl IntoArray, parent: DecimalDType) -> VortexResult<WidenArray> {
        let values = values.into_array();
        let dtype = DType::Decimal(parent, values.dtype().nullability());
        WidenArray::try_new(values, dtype)
    }

    /// Returns partially valid decimal values at both ends of the child precision.
    fn partially_valid_decimal(parent: DecimalDType) -> VortexResult<WidenArray> {
        widen_decimal(
            DecimalArray::from_option_iter(
                [Some(-99_999i32), None, Some(0), Some(99_999)],
                DecimalDType::new(5, 2),
            ),
            parent,
        )
    }

    /// Returns `value` as a scalar of the `parent` dtype, or a null scalar.
    fn decimal_scalar(
        value: Option<i64>,
        parent: DecimalDType,
        nullability: Nullability,
    ) -> Scalar {
        match value {
            Some(value) => Scalar::decimal(DecimalValue::I64(value), parent, nullability),
            None => Scalar::null(DType::Decimal(parent, Nullability::Nullable)),
        }
    }

    /// Constant bounds inside the range of the values run on the values, and the other bounds run as
    /// two compares. The strict lower and non-strict upper bounds check an exclusive and an inclusive
    /// edge.
    #[rstest]
    fn test_between_constant_bounds(
        #[values(-129i64, -128, 0, 128)] lower: i64,
        #[values(-128i64, 0, 127, 128)] upper: i64,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = PrimitiveArray::from_option_iter([Some(-128i8), None, Some(0), Some(127)]);
        let array = widen(values, PType::I64).into_array();
        let lower = ConstantArray::new(lower, array.len()).into_array();
        let upper = ConstantArray::new(upper, array.len()).into_array();
        let options = BetweenOptions {
            lower_strict: StrictComparison::Strict,
            upper_strict: StrictComparison::NonStrict,
        };

        let between = array
            .clone()
            .between(lower.clone(), upper.clone(), options.clone())?;

        let expected = canonical(&array, &mut ctx)?.between(lower, upper, options)?;
        assert_arrays_eq!(between, expected, &mut ctx);

        Ok(())
    }

    /// Nullable bounds, in range or null, give a nullable result over non-nullable values.
    #[rstest]
    #[case::in_range(
        Scalar::primitive(-1i64, Nullability::Nullable),
        Scalar::primitive(0i64, Nullability::Nullable),
    )]
    #[case::null(
        Scalar::null(DType::Primitive(PType::I64, Nullability::Nullable)),
        Scalar::primitive(0i64, Nullability::Nullable)
    )]
    fn test_between_nullable_bounds(
        #[case] lower: Scalar,
        #[case] upper: Scalar,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let array = widen(buffer![-128i8, 0, 127], PType::I64).into_array();
        let lower = ConstantArray::new(lower, array.len()).into_array();
        let upper = ConstantArray::new(upper, array.len()).into_array();
        let options = BetweenOptions {
            lower_strict: StrictComparison::NonStrict,
            upper_strict: StrictComparison::NonStrict,
        };

        let between = array
            .clone()
            .between(lower.clone(), upper.clone(), options.clone())?;

        let expected = canonical(&array, &mut ctx)?.between(lower, upper, options)?;
        assert_arrays_eq!(between, expected, &mut ctx);

        Ok(())
    }

    /// A `Widen` bound runs as two compares, and any other array bound declines.
    #[rstest]
    #[case::widen(
        widen(PrimitiveArray::from_option_iter([Some(-129i16), None, Some(128)]), PType::I64)
            .into_array()
    )]
    #[case::primitive(PrimitiveArray::from_option_iter([Some(-129i64), None, Some(128)]).into_array())]
    fn test_between_array_bound(#[case] lower: ArrayRef) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let array = widen(buffer![-128i8, 0, 127], PType::I64).into_array();
        let upper = ConstantArray::new(0i64, array.len()).into_array();
        let options = BetweenOptions {
            lower_strict: StrictComparison::NonStrict,
            upper_strict: StrictComparison::NonStrict,
        };

        let between = array
            .clone()
            .between(lower.clone(), upper.clone(), options.clone())?;

        let expected =
            canonical(&array, &mut ctx)?.between(canonical(&lower, &mut ctx)?, upper, options)?;
        assert_arrays_eq!(between, expected, &mut ctx);

        Ok(())
    }

    #[rstest]
    #[case::inside(Some(-99_999), Some(99_999))]
    #[case::outside(Some(-100_000), Some(100_000))]
    #[case::below(Some(-100_001), Some(-100_000))]
    #[case::above(Some(100_000), Some(100_001))]
    #[case::reversed(Some(99_999), Some(-99_999))]
    #[case::null_lower(None, Some(99_999))]
    #[case::null_upper(Some(-99_999), None)]
    fn test_decimal_between(
        #[case] lower: Option<i64>,
        #[case] upper: Option<i64>,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let parent = DecimalDType::new(18, 2);
        let array = partially_valid_decimal(parent)?.into_array();
        let lower = ConstantArray::new(
            decimal_scalar(lower, parent, Nullability::Nullable),
            array.len(),
        )
        .into_array();
        let upper = ConstantArray::new(
            decimal_scalar(upper, parent, Nullability::Nullable),
            array.len(),
        )
        .into_array();
        let options = BetweenOptions {
            lower_strict: StrictComparison::Strict,
            upper_strict: StrictComparison::NonStrict,
        };

        let result = array
            .clone()
            .between(lower.clone(), upper.clone(), options.clone())?;
        let expected = canonical(&array, &mut ctx)?.between(lower, upper, options)?;
        assert_arrays_eq!(result, expected, &mut ctx);

        Ok(())
    }
}
