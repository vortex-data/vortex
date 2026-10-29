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
//! [`AggregateFnSession::default`] registers the aggregate kernel, not
//! [`initialize`](super::initialize), because the kernel session runs `initialize` on a temporary
//! session and drops its aggregate registry.
//!
//! [`AggregateFnSession::default`]: crate::aggregate_fn::session::AggregateFnSession::default

pub(crate) mod aggregate;
mod between;
mod cast;
mod compare;
mod fill_null;
mod filter;
mod mask;
pub(crate) mod numeric;
pub(crate) mod rules;
mod slice;
mod take;
mod zip;

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::ConstantArray;
use crate::arrays::Widen;
use crate::arrays::WidenArray;
use crate::arrays::widen::WidenArraySlotsExt;
use crate::arrays::widen::encode::integer_value;
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
pub(super) fn wrap_values(values: ArrayRef, dtype: DType) -> VortexResult<ArrayRef> {
    if values.dtype() == &dtype {
        return Ok(values);
    }

    Ok(WidenArray::try_new(values, dtype)?.into_array())
}

/// Returns the type that the values of `array` widen to so that they also hold `scalar`: the type
/// of the values, or the smallest wider type of the logical signedness that holds `scalar`.
///
/// A null `scalar` keeps the type of the values. Returns `None` when `scalar` does not have the
/// logical dtype, or when the values or the logical dtype is not an integer (a decimal `Widen`).
pub(super) fn ptype_holding_values_and_scalar(
    array: ArrayView<'_, Widen>,
    scalar: &Scalar,
) -> VortexResult<Option<PType>> {
    let (DType::Primitive(values_ptype, _), DType::Primitive(logical_ptype, _)) =
        (array.values().dtype(), array.dtype())
    else {
        return Ok(None);
    };
    if !scalar.dtype().eq_ignore_nullability(array.dtype()) {
        return Ok(None);
    }
    if scalar.is_null() {
        return Ok(Some(*values_ptype));
    }

    let value = integer_value(scalar)?;
    let Some(scalar_ptype) = smallest_integer_ptype(value..=value, logical_ptype.is_signed_int())
    else {
        return Ok(None);
    };

    if scalar_ptype.byte_width() > values_ptype.byte_width() {
        Ok(Some(scalar_ptype))
    } else {
        Ok(Some(*values_ptype))
    }
}

/// Casts the integer array `array` to `ptype`, keeping its nullability.
pub(super) fn cast_keeping_nullability(array: &ArrayRef, ptype: PType) -> VortexResult<ArrayRef> {
    array.cast(DType::Primitive(ptype, array.dtype().nullability()))
}

/// Returns whether the dtype range of `values` holds `constant`.
///
/// Integers must have the same signedness, and decimals must have the same scale. Returns `None`
/// for null constants, unsupported dtypes, or incompatible constants.
pub(super) fn constant_fits_values(constant: &Scalar, values: &DType) -> Option<bool> {
    if let DType::Decimal(decimal_dtype, _) = values {
        let DType::Decimal(constant_dtype, _) = constant.dtype() else {
            return None;
        };
        if constant_dtype.scale() != decimal_dtype.scale() {
            return None;
        }

        return Some(
            constant
                .as_decimal()
                .decimal_value()?
                .fits_in_precision(*decimal_dtype),
        );
    }

    let DType::Primitive(ptype, _) = values else {
        return None;
    };
    let value = constant.as_primitive_opt()?.pvalue()?;

    // `PValue` compares an integer with another value through `i64` or `u64`, and it panics when
    // the other value does not convert. So only integers of the same signedness compare here.
    let value_ptype = value.ptype();
    if !ptype.is_int() || !value_ptype.is_int() {
        return None;
    }
    if ptype.is_signed_int() != value_ptype.is_signed_int() {
        return None;
    }

    Some(PValue::min_value(*ptype) <= value && value <= PValue::max_value(*ptype))
}

/// Returns the wider of the value types of two [`Widen`] arrays, or `None` when a value type is
/// not primitive.
///
/// The logical dtypes **must** have the same signedness. The value types then have that signedness
/// too, so the wider one holds every value of the narrower one.
pub(super) fn ptype_holding_both_values(
    lhs: ArrayView<'_, Widen>,
    rhs: ArrayView<'_, Widen>,
) -> Option<PType> {
    let (DType::Primitive(lhs_ptype, _), DType::Primitive(rhs_ptype, _)) =
        (lhs.values().dtype(), rhs.values().dtype())
    else {
        return None;
    };

    if lhs_ptype.is_signed_int() {
        Some(lhs_ptype.max_signed_ptype(*rhs_ptype))
    } else {
        Some(lhs_ptype.max_unsigned_ptype(*rhs_ptype))
    }
}

/// Returns `constant`, which **must** fit the type of `values`, as a constant array of that type
/// with the nullability of `constant`.
pub(super) fn constant_in_values_dtype(
    constant: &Scalar,
    values: &ArrayRef,
    len: usize,
) -> VortexResult<ArrayRef> {
    let dtype = values
        .dtype()
        .with_nullability(constant.dtype().nullability());

    Ok(ConstantArray::new(constant.cast(&dtype)?, len).into_array())
}
