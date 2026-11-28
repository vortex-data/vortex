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
use crate::arrays::widen::compute::constant_fits_values;
use crate::arrays::widen::compute::constant_in_values_dtype;
use crate::arrays::widen::compute::ptype_holding_both_values;
use crate::arrays::widen::encode::f32_fits_in_f16;
use crate::arrays::widen::encode::f64_fits_in_f16;
use crate::arrays::widen::encode::f64_fits_in_f32;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::DecimalDType;
use crate::dtype::PType;
use crate::dtype::half::f16;
use crate::scalar::PValue;
use crate::scalar::Scalar;
use crate::scalar_fn::builtin::compare::CompareKernel;
use crate::scalar_fn::builtin::operators::CompareOperator;

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
        if values.dtype().is_float() {
            return compare_float_constant(lhs, constant, operator);
        }
        let Some(fits) = constant_fits_values(constant, values.dtype()) else {
            return Ok(None);
        };
        if !fits {
            return compare_out_of_range(lhs, constant, operator, ctx).map(Some);
        }

        let constant = constant_in_values_dtype(constant, values, lhs.len())?;

        values.compare(constant, operator).map(Some)
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
    let lhs_values = lhs.values();
    let rhs_values = rhs.values();

    let dtype = match (lhs_values.dtype(), rhs_values.dtype()) {
        (DType::Primitive(..), DType::Primitive(..)) => {
            let Some(ptype) = ptype_holding_both_values(lhs, rhs) else {
                return Ok(None);
            };

            DType::from(ptype)
        }
        (DType::Decimal(lhs_dtype, _), DType::Decimal(rhs_dtype, _))
            if lhs_dtype.scale() == rhs_dtype.scale() =>
        {
            DType::Decimal(
                DecimalDType::new(
                    lhs_dtype.precision().max(rhs_dtype.precision()),
                    lhs_dtype.scale(),
                ),
                lhs_values.dtype().nullability(),
            )
        }
        _ => return Ok(None),
    };
    let lhs_values = lhs_values.cast(dtype.with_nullability(lhs_values.dtype().nullability()))?;
    let rhs_values = rhs_values.cast(dtype.with_nullability(rhs_values.dtype().nullability()))?;

    lhs_values.compare(rhs_values, operator).map(Some)
}

/// Compares a floating-point `Widen` against a non-null floating-point constant.
///
/// Pushes the comparison down to the narrow child when the non-NaN constant is represented
/// bit-for-bit in the child type. In-between constants and NaNs decline to canonical execution.
fn compare_float_constant(
    lhs: ArrayView<'_, Widen>,
    constant: &Scalar,
    operator: CompareOperator,
) -> VortexResult<Option<ArrayRef>> {
    let values = lhs.values();
    let DType::Primitive(values_ptype, _) = values.dtype() else {
        return Ok(None);
    };
    let Some(pvalue) = constant.as_primitive().pvalue() else {
        return Ok(None);
    };
    if pvalue.is_nan() {
        return Ok(None);
    }

    let nullability = constant.dtype().nullability();
    let narrow_scalar = match (pvalue, *values_ptype) {
        (PValue::F64(c), PType::F32) if f64_fits_in_f32(c) =>
        {
            #[expect(
                clippy::cast_possible_truncation,
                reason = "f64_fits_in_f32 verified bit-exact representation"
            )]
            Scalar::primitive(c as f32, nullability)
        }
        (PValue::F64(c), PType::F16) if f64_fits_in_f16(c) => {
            Scalar::primitive(f16::from_f64(c), nullability)
        }
        (PValue::F32(c), PType::F16) if f32_fits_in_f16(c) => {
            Scalar::primitive(f16::from_f32(c), nullability)
        }
        _ => return Ok(None),
    };

    let constant = ConstantArray::new(narrow_scalar, lhs.len()).into_array();
    values.compare(constant, operator).map(Some)
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
    // below zero. An unsigned constant outside the range is always above it.
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
