// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Operations that run on the values of a [`Widen`] instead of on a widened copy.
//!
//! An operation runs on the values only where it gives the canonical result. Decoding a value is a
//! strictly increasing, one-to-one map that keeps validity, so comparisons, sortedness, constancy
//! and extrema carry over from the values, and every value in a result widens exactly to the
//! logical dtype.
//!
//! Most of these operations decide from dtypes and constants alone. The arithmetic kernel also
//! needs the min and max of each operand, which come from cached statistics or from one scan of the
//! operand. That scan is the one hidden cost in this module.
//!
//! [`AggregateFnSession::default`] registers the aggregate kernel, not the `Widen` encoding's
//! `initialize`, because the kernel session runs `initialize` on a temporary session and drops its
//! aggregate registry.
//!
//! [`AggregateFnSession::default`]: crate::aggregate_fn::session::AggregateFnSession::default

pub(crate) mod aggregate;
mod cast;
mod compare;
mod fill_null;
mod filter;
mod mask;
pub(crate) mod numeric;
pub(crate) mod rules;
mod slice;
mod take;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::ConstantArray;
use crate::arrays::Widen;
use crate::arrays::WidenArray;
use crate::arrays::widen::WidenArraySlotsExt;
use crate::arrays::widen::encode::smallest_float_ptype_for_value;
use crate::arrays::widen::encode::smallest_integer_ptype;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::PType;
use crate::scalar::PValue;
use crate::scalar::Scalar;

/// Returns `values` when it has `dtype`, and a `Widen` of `dtype` over `values` otherwise.
///
/// `dtype` **must** have the nullability of `values`, because the values hold the validity of the
/// `Widen`. A mask, a fill null or a take with nullable indices can change the nullability of the
/// values, so those callers take the nullability of the result.
pub fn wrap_values(values: ArrayRef, dtype: DType) -> VortexResult<ArrayRef> {
    if values.dtype() == &dtype {
        return Ok(values);
    }

    Ok(WidenArray::try_new(values, dtype)?.into_array())
}

/// Returns the type that the values of `array` widen to so that they also hold `scalar`: the type
/// of the values, or the smallest wider primitive type of the logical kind that holds `scalar`
/// losslessly.
///
/// A null `scalar` keeps the type of the values. Returns `None` when `scalar` does not have the
/// logical dtype, or when the values or the logical dtype is not primitive (a decimal `Widen`).
pub fn ptype_holding_values_and_scalar(
    array: ArrayView<'_, Widen>,
    scalar: &Scalar,
) -> Option<PType> {
    let (DType::Primitive(values_ptype, _), DType::Primitive(logical_ptype, _)) =
        (array.values().dtype(), array.dtype())
    else {
        return None;
    };
    if !scalar.dtype().eq_ignore_nullability(array.dtype()) {
        return None;
    }
    if scalar.is_null() {
        return Some(*values_ptype);
    }

    let pvalue = scalar
        .as_primitive()
        .pvalue()
        .vortex_expect("non-null primitive scalar has a value");
    let scalar_ptype = if logical_ptype.is_float() {
        smallest_float_ptype_for_value(pvalue)?
    } else {
        let value = pvalue
            .as_i128()
            .vortex_expect("the non-null scalar has the Widen's integer dtype");
        smallest_integer_ptype(value..=value, logical_ptype.is_signed_int())?
    };

    if scalar_ptype.byte_width() > values_ptype.byte_width() {
        Some(scalar_ptype)
    } else {
        Some(*values_ptype)
    }
}

/// Casts the primitive array `array` to `ptype`, keeping its nullability.
pub fn cast_keeping_nullability(array: &ArrayRef, ptype: PType) -> VortexResult<ArrayRef> {
    array.cast(DType::Primitive(ptype, array.dtype().nullability()))
}

/// Returns whether the dtype of `values` holds `constant` losslessly.
///
/// Integers must have the same signedness, floats must round-trip bit-for-bit without being NaN,
/// and decimals must have the same scale. Returns `None` for null or NaN constants, unsupported
/// dtypes, or incompatible constants.
pub fn constant_fits_values(constant: &Scalar, values: &DType) -> Option<bool> {
    match (values, constant.dtype()) {
        (DType::Decimal(values_dtype, _), DType::Decimal(constant_dtype, _))
            if values_dtype.scale() == constant_dtype.scale() =>
        {
            let value = constant.as_decimal().decimal_value()?;

            Some(value.fits_in_precision(*values_dtype))
        }
        (DType::Primitive(values_ptype, _), DType::Primitive(constant_ptype, _))
            if values_ptype.is_int()
                && constant_ptype.is_int()
                && values_ptype.is_signed_int() == constant_ptype.is_signed_int() =>
        {
            // `PValue` compares through `i64` or `u64`, so the signedness must match.
            let value = constant.as_primitive().pvalue()?;

            Some(
                PValue::min_value(*values_ptype) <= value
                    && value <= PValue::max_value(*values_ptype),
            )
        }
        _ => None,
    }
}

/// Returns the wider of the value types of two [`Widen`] arrays, or `None` when a value type is
/// not a compatible primitive type.
///
/// Both arrays must share the same primitive kind (both signed integers, both unsigned integers,
/// or both floats), so the wider value type holds every value of the narrower one.
pub fn ptype_holding_both_values(
    lhs: ArrayView<'_, Widen>,
    rhs: ArrayView<'_, Widen>,
) -> Option<PType> {
    let (DType::Primitive(lhs_ptype, _), DType::Primitive(rhs_ptype, _)) =
        (lhs.values().dtype(), rhs.values().dtype())
    else {
        return None;
    };

    if lhs_ptype.is_float() && rhs_ptype.is_float() {
        if lhs_ptype.byte_width() >= rhs_ptype.byte_width() {
            Some(*lhs_ptype)
        } else {
            Some(*rhs_ptype)
        }
    } else if lhs_ptype.is_signed_int() && rhs_ptype.is_signed_int() {
        Some(lhs_ptype.max_signed_ptype(*rhs_ptype))
    } else if lhs_ptype.is_unsigned_int() && rhs_ptype.is_unsigned_int() {
        Some(lhs_ptype.max_unsigned_ptype(*rhs_ptype))
    } else {
        None
    }
}

/// Returns `constant`, which **must** fit the type of `values`, as a constant array of that type
/// with the nullability of `constant`.
pub fn constant_in_values_dtype(
    constant: &Scalar,
    values: &ArrayRef,
    len: usize,
) -> VortexResult<ArrayRef> {
    let dtype = values
        .dtype()
        .with_nullability(constant.dtype().nullability());

    Ok(ConstantArray::new(constant.cast(&dtype)?, len).into_array())
}
