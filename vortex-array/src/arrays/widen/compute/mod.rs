// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Operations that run on the values of a [`Widen`](crate::arrays::Widen) instead of on a
//! widened copy.
//!
//! An operation runs on the values only where it gives the canonical result. Decoding a value is a
//! strictly increasing, one-to-one map that keeps validity, so comparisons, sortedness, constancy
//! and extrema carry over from the values, and every value in a result widens exactly to the
//! logical dtype.
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
pub(crate) mod zip;

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
use crate::dtype::DType;
use crate::dtype::PType;
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

/// Returns the integer type that holds both the values of `array` and `scalar`, without reading any
/// values.
///
/// This is the type of the values, or the smallest type of the logical signedness that holds
/// `scalar` when that type is wider. A null `scalar` gives the type of the values. The result is
/// never wider than the logical type. Returns `None` when the values or the logical dtype is not
/// primitive, or when `scalar` does not have the logical dtype.
pub(super) fn integer_storage_type(
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

/// Returns the wider of the value types of two [`Widen`] arrays, or `None` when a value type is
/// not primitive.
///
/// The logical dtypes **must** have the same signedness, which a debug build asserts. The value
/// types then have that signedness too, so the wider one holds every value of the narrower one.
pub(super) fn wider_values_ptype(
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
pub(super) fn constant_values(
    constant: &Scalar,
    values: &ArrayRef,
    len: usize,
) -> VortexResult<ArrayRef> {
    let dtype = values
        .dtype()
        .with_nullability(constant.dtype().nullability());

    Ok(ConstantArray::new(constant.cast(&dtype)?, len).into_array())
}
