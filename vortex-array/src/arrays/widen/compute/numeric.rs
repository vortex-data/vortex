// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::RangeInclusive;

use itertools::Itertools;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::Canonical;
use crate::Columnar;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::aggregate_fn::NumericalAggregateOpts;
use crate::aggregate_fn::fns::min_max::min_max;
use crate::array::ArrayView;
use crate::arrays::Constant;
use crate::arrays::ConstantArray;
use crate::arrays::ScalarFn;
use crate::arrays::Widen;
use crate::arrays::scalar_fn::ExactScalarFn;
use crate::arrays::scalar_fn::ScalarFnArrayExt;
use crate::arrays::scalar_fn::ScalarFnArrayView;
use crate::arrays::widen::WidenArraySlotsExt;
use crate::arrays::widen::compute::cast_to_ptype;
use crate::arrays::widen::compute::wrap_values;
use crate::arrays::widen::encode::integer_value;
use crate::arrays::widen::encode::smallest_integer_ptype;
use crate::dtype::DType;
use crate::dtype::PType;
use crate::kernel::ExecuteParentKernel;
use crate::scalar::NumericOperator;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::binary::Binary;
use crate::scalar_fn::fns::binary::execute_numeric;
use crate::scalar_fn::fns::operators::Operator;

/// Runs checked add, subtract, multiply and divide of a [`Widen`] with another `Widen` or a
/// constant at the smallest integer type that holds both operands and every result.
///
/// Narrow values read and write less memory. [`arithmetic_ptype`] picks the type from the extrema
/// of the valid rows, so the type can be too wide but never too narrow. When no narrower type is
/// known to hold the result, the operation runs at the logical type, where the checked kernel
/// reports an overflow or a division by zero with the same error as canonical execution. Any other
/// operand declines, so the default path widens it once.
#[derive(Debug)]
pub(crate) struct WidenNumericKernel;

impl ExecuteParentKernel<Widen> for WidenNumericKernel {
    type Parent = ExactScalarFn<Binary>;

    fn execute_parent(
        &self,
        array: ArrayView<'_, Widen>,
        parent: ScalarFnArrayView<'_, Binary>,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let operator = match parent.options {
            Operator::Add => NumericOperator::Add,
            Operator::Sub => NumericOperator::Sub,
            Operator::Mul => NumericOperator::Mul,
            Operator::Div => NumericOperator::Div,
            _ => return Ok(None),
        };

        let DType::Primitive(logical_ptype, _) = array.dtype() else {
            return Ok(None);
        };

        let binary = parent
            .as_opt::<ScalarFn>()
            .vortex_expect("ExactScalarFn matcher confirmed ScalarFnArray");
        let other = match child_idx {
            0 => binary.get_child(1),
            1 => binary.get_child(0),
            _ => return Ok(None),
        };
        let other_stored = if let Some(other) = other.as_opt::<Widen>() {
            other.values().clone()
        } else if other.is::<Constant>() {
            other.clone()
        } else {
            return Ok(None);
        };

        let dtype = parent.dtype();
        if parent.is_empty() {
            return Ok(Some(Canonical::empty(dtype).into_array()));
        }

        // Execute each stored operand once. `Columnar` keeps a constant constant.
        let stored = array
            .values()
            .clone()
            .execute::<Columnar>(ctx)?
            .into_array();
        let other_stored = other_stored.execute::<Columnar>(ctx)?.into_array();
        let (lhs, rhs) = if child_idx == 0 {
            (stored, other_stored)
        } else {
            (other_stored, stored)
        };

        // An operand without a valid row makes every row null.
        let (Some(lhs_bounds), Some(rhs_bounds)) =
            (valid_bounds(&lhs, ctx)?, valid_bounds(&rhs, ctx)?)
        else {
            let null = Scalar::null(dtype.clone());
            return Ok(Some(ConstantArray::new(null, parent.len()).into_array()));
        };

        let ptype = arithmetic_ptype(&lhs_bounds, &rhs_bounds, operator, *logical_ptype);
        let lhs = cast_to_ptype(&lhs, ptype)?;
        let rhs = cast_to_ptype(&rhs, ptype)?;
        let values = execute_numeric(&lhs, &rhs, operator, ctx)?;

        wrap_values(values, dtype.clone()).map(Some)
    }
}

/// Returns the smallest type of the signedness of `logical` that holds the valid values of both
/// operands and every result of `operator`, or `logical` when no narrower type is known to hold
/// them.
///
/// Null rows do not count. A cast to the narrower type can change their payloads, and the checked
/// kernel skips them.
fn arithmetic_ptype(
    lhs: &RangeInclusive<i128>,
    rhs: &RangeInclusive<i128>,
    operator: NumericOperator,
    logical: PType,
) -> PType {
    result_interval(lhs, rhs, operator)
        .and_then(|result| {
            let min = (*lhs.start()).min(*rhs.start()).min(*result.start());
            let max = (*lhs.end()).max(*rhs.end()).max(*result.end());
            smallest_integer_ptype(min..=max, logical.is_signed_int())
        })
        .filter(|ptype| ptype.byte_width() < logical.byte_width())
        .unwrap_or(logical)
}

/// Returns the extrema of the valid values of an integer array, or `None` when no row is valid.
fn valid_bounds(
    array: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<RangeInclusive<i128>>> {
    let Some(bounds) = min_max(array, ctx, NumericalAggregateOpts::default())? else {
        return Ok(None);
    };

    Ok(Some(
        integer_value(&bounds.min)?..=integer_value(&bounds.max)?,
    ))
}

/// Returns the smallest interval that holds `lhs operator rhs` for every value of `lhs` and every
/// value of `rhs`, skipping a divisor of zero.
///
/// Returns `None` when the interval leaves the `i128` range, or when a division has no divisor
/// other than zero. The bounds of 64-bit integers never leave the `i128` range in an addition, a
/// subtraction or a division.
pub(crate) fn result_interval(
    lhs: &RangeInclusive<i128>,
    rhs: &RangeInclusive<i128>,
    operator: NumericOperator,
) -> Option<RangeInclusive<i128>> {
    let (lhs_min, lhs_max) = (*lhs.start(), *lhs.end());
    let (rhs_min, rhs_max) = (*rhs.start(), *rhs.end());

    let (min, max) = match operator {
        NumericOperator::Add => (lhs_min.checked_add(rhs_min)?, lhs_max.checked_add(rhs_max)?),
        NumericOperator::Sub => (lhs_min.checked_sub(rhs_max)?, lhs_max.checked_sub(rhs_min)?),
        NumericOperator::Mul => [
            lhs_min.checked_mul(rhs_min)?,
            lhs_min.checked_mul(rhs_max)?,
            lhs_max.checked_mul(rhs_min)?,
            lhs_max.checked_mul(rhs_max)?,
        ]
        .into_iter()
        .minmax()
        .into_option()?,
        // A truncating division is monotone in the dividend, and in the divisor on each side of
        // zero. So the extrema are quotients of the extreme dividends by the extreme divisors of
        // each side, and the divisors next to zero are -1 and 1.
        NumericOperator::Div => [rhs_min, rhs_max, -1, 1]
            .into_iter()
            .filter(|divisor| *divisor != 0 && rhs.contains(divisor))
            .map(|divisor| Some([lhs_min.checked_div(divisor)?, lhs_max.checked_div(divisor)?]))
            .collect::<Option<Vec<_>>>()?
            .into_iter()
            .flatten()
            .minmax()
            .into_option()?,
    };

    Some(min..=max)
}
