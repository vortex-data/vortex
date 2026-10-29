// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BitBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::BoolArray;
use crate::arrays::ConstantArray;
use crate::arrays::Widen;
use crate::arrays::widen::WidenArraySlotsExt;
use crate::arrays::widen::compute::cast_keeping_nullability;
use crate::arrays::widen::compute::constant_fits_values;
use crate::arrays::widen::compute::constant_in_values_dtype;
use crate::arrays::widen::compute::ptype_holding_both_values;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::DecimalDType;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::binary::CompareKernel;
use crate::scalar_fn::fns::operators::CompareOperator;
use crate::scalar_fn::fns::operators::Operator;

impl CompareKernel for Widen {
    fn compare(
        lhs: ArrayView<'_, Self>,
        rhs: &ArrayRef,
        operator: CompareOperator,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        if !lhs.dtype().eq_ignore_nullability(rhs.dtype()) {
            return Ok(None);
        }

        if let Some(rhs) = rhs.as_opt::<Widen>() {
            return compare_narrow_values(lhs, rhs, operator);
        }

        // `CompareExecuteAdaptor` answers a null constant before it calls this kernel.
        let Some(constant) = rhs.as_constant() else {
            return Ok(None);
        };
        let values = lhs.values();
        let Some(fits) = constant_fits_values(constant, values.dtype()) else {
            return Ok(None);
        };
        if !fits {
            return compare_out_of_range(lhs, constant, operator, ctx).map(Some);
        }

        let constant = constant_in_values_dtype(constant, values, lhs.len())?;

        values.binary(constant, Operator::from(operator)).map(Some)
    }
}

/// Compares the values of two `Widen` arrays at the wider of the two value types.
///
/// Decimal values with the same scale compare at the larger of the two child precisions, which
/// holds every value of both children.
fn compare_narrow_values(
    lhs: ArrayView<'_, Widen>,
    rhs: ArrayView<'_, Widen>,
    operator: CompareOperator,
) -> VortexResult<Option<ArrayRef>> {
    if let (DType::Decimal(lhs_dtype, _), DType::Decimal(rhs_dtype, _)) =
        (lhs.values().dtype(), rhs.values().dtype())
    {
        if lhs_dtype.scale() != rhs_dtype.scale() {
            return Ok(None);
        }

        let decimal_dtype = DecimalDType::new(
            lhs_dtype.precision().max(rhs_dtype.precision()),
            lhs_dtype.scale(),
        );
        let lhs_values = lhs.values().cast(DType::Decimal(
            decimal_dtype,
            lhs.values().dtype().nullability(),
        ))?;
        let rhs_values = rhs.values().cast(DType::Decimal(
            decimal_dtype,
            rhs.values().dtype().nullability(),
        ))?;

        return lhs_values
            .binary(rhs_values, Operator::from(operator))
            .map(Some);
    }

    let Some(ptype) = ptype_holding_both_values(lhs, rhs) else {
        return Ok(None);
    };

    let lhs_values = cast_keeping_nullability(lhs.values(), ptype)?;
    let rhs_values = cast_keeping_nullability(rhs.values(), ptype)?;

    lhs_values
        .binary(rhs_values, Operator::from(operator))
        .map(Some)
}

/// Answers a compare of `lhs` with a valid `constant` outside the range of its values.
///
/// Every value is on the same side of the constant, so every valid row has the same answer, and the
/// result takes the validity of the values. Without nulls, the result is a [`ConstantArray`], so
/// that later steps such as a filter keep their constant fast paths.
fn compare_out_of_range(
    lhs: ArrayView<'_, Widen>,
    constant: &Scalar,
    operator: CompareOperator,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    // The range of the values holds zero, so the constant is below the range exactly when it is
    // below zero. An unsigned constant outside the range is always above it. A decimal range is
    // the range of the child precision, which is symmetric about zero. It trusts the child label,
    // so a valid value outside the child precision gives a wrong answer.
    let zero = Scalar::zero_value(constant.dtype());
    let below = constant
        .partial_cmp(&zero)
        .vortex_expect("a scalar compares with the zero of its own dtype")
        .is_lt();
    let answer = match operator {
        CompareOperator::Eq => false,
        CompareOperator::NotEq => true,
        CompareOperator::Lt | CompareOperator::Lte => !below,
        CompareOperator::Gt | CompareOperator::Gte => below,
    };

    let validity = lhs
        .values()
        .validity()?
        .union_nullability(constant.dtype().nullability());
    if validity.definitely_no_nulls() {
        let answer = Scalar::bool(answer, validity.nullability());
        return Ok(ConstantArray::new(answer, lhs.len()).into_array());
    }

    let bits = BitBuffer::full_in(answer, lhs.len(), ctx.allocator().clone());
    Ok(BoolArray::try_new(bits, validity)?.into_array())
}
