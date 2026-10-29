// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::RangeInclusive;

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::aggregate_fn::NumericalAggregateOpts;
use crate::aggregate_fn::fns::min_max::min_max;
use crate::arrays::DecimalArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::WidenArray;
use crate::arrays::decimal::DecimalArrayExt;
use crate::arrays::decimal::copy_to_canonical_width;
use crate::arrays::decimal::valid_range;
use crate::arrays::widen::compute::wrap_values;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::DecimalDType;
use crate::dtype::NativeDecimalType;
use crate::dtype::PType;
use crate::dtype::i256;
use crate::match_each_decimal_value_type;
use crate::scalar::DecimalValue;
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

    /// Stores a decimal array at the smallest precision that its valid values and scale allow.
    ///
    /// The result keeps the input dtype and nullability. Empty and all-null arrays use the
    /// precision required by the scale, or 1 for a non-positive scale. Returns the input unchanged
    /// when the selected precision has the canonical width of the input dtype.
    ///
    /// # Errors
    ///
    /// Returns an error if a valid value does not fit the input precision.
    pub fn encode_decimal(array: DecimalArray, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
        let parent = array.decimal_dtype();
        let validity = array.validity()?;
        let mask = validity.execute_mask(array.len(), ctx)?;
        let range = match_each_decimal_value_type!(array.values_type(), |T| {
            valid_range(array.buffer::<T>().as_slice(), &mask).map(|(min, max)| {
                DecimalValue::from(min).as_i256()..=DecimalValue::from(max).as_i256()
            })
        });
        let child_dtype = smallest_decimal_dtype(range, parent)?;
        if child_dtype.canonical_values_type() == parent.canonical_values_type() {
            return Ok(array.into_array());
        }

        let buffer = match_each_decimal_value_type!(array.values_type(), |T| {
            copy_to_canonical_width(array.buffer::<T>().as_slice(), &mask, child_dtype)?
        });
        let dtype = array.dtype().clone();

        // SAFETY: the range proves that every valid value fits the child precision. The copy
        // preserves the scale and length, uses the child canonical width, and keeps its validity.
        let values = unsafe {
            DecimalArray::new_unchecked_handle(
                buffer,
                child_dtype.canonical_values_type(),
                child_dtype,
                validity,
            )
        };

        Ok(WidenArray::try_new(values.into_array(), dtype)?.into_array())
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

/// Returns the smallest decimal dtype that holds an unscaled valid-value range at the parent scale.
///
/// The precision is at least 1 and at least a positive scale. An absent range represents an empty
/// or all-null array and needs only that minimum precision. The result never exceeds the parent
/// precision. Callers **must** exclude null payloads from the range, for example with
/// [`valid_range`].
///
/// # Errors
///
/// Returns an error if the minimum exceeds the maximum or an endpoint exceeds the parent precision.
pub(crate) fn smallest_decimal_dtype(
    range: Option<RangeInclusive<i256>>,
    parent: DecimalDType,
) -> VortexResult<DecimalDType> {
    let minimum_precision = parent.scale().max(1) as u8;
    let Some(range) = range else {
        return Ok(DecimalDType::new(minimum_precision, parent.scale()));
    };

    let (min, max) = range.into_inner();
    vortex_ensure!(
        min <= max,
        "Decimal range requires ordered extrema, got {min} to {max}"
    );

    for precision in minimum_precision..=parent.precision() {
        let index = usize::from(precision);
        if i256::MIN_BY_PRECISION[index] <= min && max <= i256::MAX_BY_PRECISION[index] {
            return Ok(DecimalDType::new(precision, parent.scale()));
        }
    }

    vortex_bail!("Decimal range must fit {parent}, got {min} to {max}")
}
