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
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::DecimalDType;
use crate::dtype::Nullability;
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
            return compare_float_constant(lhs, constant, operator, ctx);
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

/// Position of a non-NaN wider floating-point constant relative to the representable values of a
/// narrower floating-point type under IEEE-754 total order.
enum FloatConstantBound {
    /// The constant is represented bit-for-bit by `scalar` in the child type.
    Exact(Scalar),
    /// The constant is finite, non-zero, and strictly between `lo` and its successor in total
    /// order.
    Between { lo: Scalar },
}

/// Compares a floating-point `Widen` against a non-null floating-point constant.
///
/// Vortex compares floats under IEEE-754 total order (`total_cmp`) and bitwise equality. Narrow
/// float types already contain `±inf` and `±NaN`, so a finite wider constant that does not fit
/// bit-exactly is never outside the child's total-order span; instead it falls strictly between
/// two adjacent narrow values `lo < constant < hi`, so ordering comparisons rewrite to
/// `values <= lo` or `values > lo` while `Eq`/`NotEq` fold to a constant preserving validity.
/// NaN constants decline so canonical comparison handles arbitrary NaN signs and payloads.
fn compare_float_constant(
    lhs: ArrayView<'_, Widen>,
    constant: &Scalar,
    operator: CompareOperator,
    ctx: &mut ExecutionCtx,
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
    let bound = match (pvalue, *values_ptype) {
        (PValue::F64(c), PType::F32) => classify_f64_for_f32(c, nullability),
        (PValue::F64(c), PType::F16) => classify_f64_for_f16(c, nullability),
        (PValue::F32(c), PType::F16) => classify_f32_for_f16(c, nullability),
        _ => return Ok(None),
    };

    match bound {
        FloatConstantBound::Exact(narrow_scalar) => {
            let constant = ConstantArray::new(narrow_scalar, lhs.len()).into_array();
            values.binary(constant, Operator::from(operator)).map(Some)
        }
        FloatConstantBound::Between { lo } => match operator {
            CompareOperator::Eq => {
                constant_bool_with_validity(lhs, nullability, false, ctx).map(Some)
            }
            CompareOperator::NotEq => {
                constant_bool_with_validity(lhs, nullability, true, ctx).map(Some)
            }
            CompareOperator::Lt | CompareOperator::Lte => {
                let constant = ConstantArray::new(lo, lhs.len()).into_array();
                values.binary(constant, Operator::Lte).map(Some)
            }
            CompareOperator::Gt | CompareOperator::Gte => {
                let constant = ConstantArray::new(lo, lhs.len()).into_array();
                values.binary(constant, Operator::Gt).map(Some)
            }
        },
    }
}

fn classify_f64_for_f32(c: f64, nullability: Nullability) -> FloatConstantBound {
    debug_assert!(!c.is_nan());
    #[expect(
        clippy::cast_possible_truncation,
        reason = "IEEE-754 rounding to nearest f32 is checked bit-for-bit against the original f64"
    )]
    let rounded = c as f32;
    let widened = f64::from(rounded);
    if widened.to_bits() == c.to_bits() {
        return FloatConstantBound::Exact(Scalar::primitive(rounded, nullability));
    }
    let lo = if widened.total_cmp(&c).is_lt() {
        rounded
    } else {
        prev_f32_total_order(rounded)
    };
    debug_assert!(f64::from(lo).total_cmp(&c).is_lt());
    FloatConstantBound::Between {
        lo: Scalar::primitive(lo, nullability),
    }
}

fn classify_f64_for_f16(c: f64, nullability: Nullability) -> FloatConstantBound {
    debug_assert!(!c.is_nan());
    let rounded = f16::from_f64(c);
    let widened = rounded.to_f64();
    if widened.to_bits() == c.to_bits() {
        return FloatConstantBound::Exact(Scalar::primitive(rounded, nullability));
    }
    let lo = if widened.total_cmp(&c).is_lt() {
        rounded
    } else {
        prev_f16_total_order(rounded)
    };
    debug_assert!(lo.to_f64().total_cmp(&c).is_lt());
    FloatConstantBound::Between {
        lo: Scalar::primitive(lo, nullability),
    }
}

fn classify_f32_for_f16(c: f32, nullability: Nullability) -> FloatConstantBound {
    debug_assert!(!c.is_nan());
    let rounded = f16::from_f32(c);
    let widened = rounded.to_f32();
    if widened.to_bits() == c.to_bits() {
        return FloatConstantBound::Exact(Scalar::primitive(rounded, nullability));
    }
    let lo = if widened.total_cmp(&c).is_lt() {
        rounded
    } else {
        prev_f16_total_order(rounded)
    };
    debug_assert!(lo.to_f32().total_cmp(&c).is_lt());
    FloatConstantBound::Between {
        lo: Scalar::primitive(lo, nullability),
    }
}

/// Immediate predecessor of a non-NaN `f32` strictly greater than `-inf` under IEEE-754 total
/// order (`-inf < ... < -min_sub < -0.0 < +0.0 < +min_sub < ... < +inf`).
fn prev_f32_total_order(v: f32) -> f32 {
    debug_assert!(!v.is_nan() && v.to_bits() != f32::NEG_INFINITY.to_bits());
    let bits = v.to_bits();
    if bits == 0 {
        f32::from_bits(1u32 << 31)
    } else if (bits & (1u32 << 31)) == 0 {
        f32::from_bits(bits - 1)
    } else {
        f32::from_bits(bits + 1)
    }
}

/// Immediate predecessor of a non-NaN `f16` strictly greater than `-inf` under IEEE-754 total
/// order (`-inf < ... < -min_sub < -0.0 < +0.0 < +min_sub < ... < +inf`).
fn prev_f16_total_order(v: f16) -> f16 {
    debug_assert!(!v.is_nan() && v.to_bits() != f16::NEG_INFINITY.to_bits());
    let bits = v.to_bits();
    if bits == 0 {
        f16::from_bits(1u16 << 15)
    } else if (bits & (1u16 << 15)) == 0 {
        f16::from_bits(bits - 1)
    } else {
        f16::from_bits(bits + 1)
    }
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

    constant_bool_with_validity(lhs, constant.dtype().nullability(), answer, ctx)
}

fn constant_bool_with_validity(
    lhs: ArrayView<'_, Widen>,
    constant_nullability: Nullability,
    answer: bool,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let validity = lhs
        .values()
        .validity()?
        .union_nullability(constant_nullability);
    if validity.definitely_no_nulls() {
        let answer = Scalar::bool(answer, validity.nullability());
        return Ok(ConstantArray::new(answer, lhs.len()).into_array());
    }

    let bits = BitBuffer::full_in(answer, lhs.len(), ctx.allocator().clone());
    Ok(BoolArray::try_new(bits, validity)?.into_array())
}
