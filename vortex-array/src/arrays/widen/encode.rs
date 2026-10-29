// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::RangeInclusive;

use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::aggregate_fn::NumericalAggregateOpts;
use crate::aggregate_fn::fns::min_max::min_max;
use crate::arrays::PrimitiveArray;
use crate::arrays::WidenArray;
use crate::arrays::widen::compute::wrap_values;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::PType;
use crate::scalar::Scalar;

impl WidenArray {
    /// Stores `array` as a `Widen` over the smallest integer type with the same signedness that
    /// holds its valid values, or returns `array` unchanged when no narrower type holds them.
    ///
    /// Null rows do not count, so their payloads can be outside the stored type, and an empty or
    /// all-null array gets a 1-byte type. Fails when `array` is not an integer array.
    pub fn encode(array: PrimitiveArray, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
        let dtype = array.dtype().clone();
        wrap_values(narrow_values(array, ctx)?.into_array(), dtype)
    }

    /// Returns `array` cast to the type that [`Self::encode`] chooses for its values, without a
    /// `Widen` around them.
    ///
    /// The result keeps the signedness and the validity of `array`, but its dtype can be narrower.
    /// A float array returns unchanged, where [`Self::encode`] fails. This is for an internal child
    /// whose parent chooses its dtype, such as list offsets or dictionary codes. An array whose
    /// dtype is fixed, such as a column, needs [`Self::encode`] instead.
    #[deprecated(
        note = "Changes the dtype. Each caller must decide whether it needs a Widen (#10733)"
    )]
    pub fn encode_values(
        array: PrimitiveArray,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<PrimitiveArray> {
        if !array.ptype().is_int() {
            return Ok(array);
        }

        narrow_values(array, ctx)
    }
}

/// Returns `array` cast to the stored type that [`WidenArray::encode`] chooses, or `array`
/// unchanged when no narrower type holds its valid values.
///
/// [`WidenArray::encode`] and the deprecated [`WidenArray::encode_values`] share this function,
/// so that `encode` needs no `#[expect(deprecated)]`. Fails when `array` is not an integer array.
fn narrow_values(array: PrimitiveArray, ctx: &mut ExecutionCtx) -> VortexResult<PrimitiveArray> {
    let ptype = array.ptype();
    vortex_ensure!(ptype.is_int(), "Widen requires integer values, got {ptype}");

    // No integer type is narrower than one byte, so the bounds cannot change the result.
    if ptype.byte_width() == 1 {
        return Ok(array);
    }

    // An array without valid values fits any type, and the bounds `0..=0` select the 1-byte type.
    let bounds = match min_max(array.as_ref(), ctx, NumericalAggregateOpts::default())? {
        Some(bounds) => integer_value(&bounds.min)?..=integer_value(&bounds.max)?,
        None => 0..=0,
    };
    let Some(stored_ptype) = smallest_integer_ptype(bounds, ptype.is_signed_int())
        .filter(|stored_ptype| stored_ptype.byte_width() < ptype.byte_width())
    else {
        return Ok(array);
    };

    let stored_dtype = DType::Primitive(stored_ptype, array.dtype().nullability());
    array
        .into_array()
        .cast(stored_dtype)?
        .execute::<PrimitiveArray>(ctx)
}

/// Returns the smallest integer type with the given signedness that holds every value in `bounds`,
/// or `None` when no integer type of at most 64 bits holds them.
pub(super) fn smallest_integer_ptype(bounds: RangeInclusive<i128>, signed: bool) -> Option<PType> {
    let (min, max) = bounds.into_inner();
    if signed {
        let min_ptype = PType::min_signed_ptype_for_value(i64::try_from(min).ok()?);
        let max_ptype = PType::min_signed_ptype_for_value(i64::try_from(max).ok()?);
        Some(min_ptype.max_signed_ptype(max_ptype))
    } else {
        let min_ptype = PType::min_unsigned_ptype_for_value(u64::try_from(min).ok()?);
        let max_ptype = PType::min_unsigned_ptype_for_value(u64::try_from(max).ok()?);
        Some(min_ptype.max_unsigned_ptype(max_ptype))
    }
}

/// Returns the value of a non-null integer scalar of at most 64 bits.
pub(super) fn integer_value(scalar: &Scalar) -> VortexResult<i128> {
    if scalar.dtype().is_signed_int() {
        i64::try_from(scalar).map(i128::from)
    } else {
        u64::try_from(scalar).map(i128::from)
    }
}
