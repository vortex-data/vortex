// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Decimal encoding and stored-value reads with valid child precision.
//!
//! Child precision depends on valid values and scale, not just storage width. For example,
//! unscaled 127 fits `i8` but needs precision 3, whose canonical width is `i16`. Both entry points
//! share child selection and cache its value range for later aggregates.
//!
//! [`WidenArray::from_stored_decimal`] is the conversion boundary for legacy decimal storage and
//! decoded DecimalByteParts buffers.

use std::ops::RangeInclusive;

use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::DecimalArray;
use crate::arrays::WidenArray;
use crate::arrays::decimal::DecimalArrayExt;
use crate::arrays::decimal::DecimalData;
use crate::arrays::decimal::DecimalDataParts;
use crate::arrays::decimal::copy_to_canonical_width;
use crate::arrays::decimal::valid_range;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::dtype::DecimalDType;
use crate::dtype::NativeDecimalType;
use crate::dtype::i256;
use crate::expr::stats::Precision;
use crate::expr::stats::Stat;
use crate::match_each_decimal_value_type;
use crate::scalar::DecimalValue;
use crate::scalar::ScalarValue;
use crate::validity::Validity;

impl WidenArray {
    /// Stores a decimal array at the smallest precision that its valid values and scale allow.
    ///
    /// The result keeps the input dtype and nullability. Empty and all-null arrays use the
    /// precision required by the scale, or 1 for a non-positive scale. Returns the input unchanged
    /// when the selected precision has the canonical width of the input dtype. The child records
    /// the exact extrema of the valid values as its [`Stat::Min`] and [`Stat::Max`].
    ///
    /// # Errors
    ///
    /// Returns an error if a valid value does not fit the input precision.
    pub fn encode_decimal(array: DecimalArray, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
        let logical_decimal_dtype = array.decimal_dtype();
        let validity = array.validity()?;
        let validity_mask = validity.execute_mask(array.len(), ctx)?;
        let child = match_each_decimal_value_type!(array.values_type(), |T| {
            let values_buffer = array.buffer::<T>();
            let valid_value_range = scan_valid_range(values_buffer.as_slice(), &validity_mask);
            narrowest_decimal_child(
                &values_buffer,
                &validity,
                &validity_mask,
                valid_value_range,
                logical_decimal_dtype,
            )?
        });

        let Some(child) = child else {
            return Ok(array.into_array());
        };

        Ok(WidenArray::try_new(child.into_array(), array.dtype().clone())?.into_array())
    }

    /// Reads stored decimal values as a canonical array or a [`WidenArray`].
    ///
    /// Keeps the dtype and validity in `parts`. Canonical-width storage is reused without reading
    /// values, including device buffers. Its declared precision is trusted, and `known_range` is
    /// ignored.
    ///
    /// Other widths use host memory and select a child as [`Self::encode_decimal`] does. Validity
    /// **must** be executable on the host. The child reuses matching stored widths.
    /// Otherwise the values are copied to the selected width, with null slots zeroed on narrowing.
    /// If no child is narrower than the logical dtype, the result is a canonical array.
    ///
    /// For noncanonical storage, `known_range` controls child selection:
    ///
    /// - [`Precision::Absent`] scans and caches exact extrema on the selected child.
    /// - [`Precision::Exact`] **must** give the actual minimum and maximum valid unscaled values.
    /// - [`Precision::Inexact`] **must** contain every valid unscaled value.
    ///
    /// Supplied ranges retain their exactness in the child's [`Stat::Min`] and [`Stat::Max`].
    /// An incorrect range or declared precision can cause incorrect results in later operations.
    ///
    /// # Errors
    ///
    /// Rejects malformed buffer layout or validity length. For noncanonical storage, also rejects
    /// failed host copies, reversed ranges, and range bounds outside the logical precision. A
    /// narrowing copy rejects valid values outside the selected precision.
    pub fn from_stored_decimal(
        parts: DecimalDataParts,
        known_range: Precision<RangeInclusive<i256>>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let DecimalDataParts {
            decimal_dtype,
            values,
            values_type,
            validity,
        } = parts;

        // Validate before typed buffer access or mask execution, which can panic on malformed data.
        DecimalData::validate(&values, values_type)?;
        let len = values.len() / values_type.byte_width();
        if let Some(validity_len) = validity.maybe_len() {
            vortex_ensure!(
                validity_len == len,
                "Decimal validity must have one entry per value ({len}), got {validity_len}"
            );
        }

        if values_type == decimal_dtype.canonical_values_type() {
            let array = DecimalArray::try_new_handle(values, values_type, decimal_dtype, validity)?;

            return Ok(array.into_array());
        }

        let values = values.try_into_host_sync()?;
        let mask = validity.execute_mask(len, ctx)?;
        match_each_decimal_value_type!(values_type, |T| {
            let values = Buffer::<T>::from_byte_buffer(values);
            let range = match known_range {
                Precision::Exact(extrema) => ValidRange::Exact(extrema),
                Precision::Inexact(bounds) => ValidRange::Bounds(bounds),
                Precision::Absent => scan_valid_range(values.as_slice(), &mask),
            };
            widen_or_canonical(values, validity, &mask, range, decimal_dtype)
        })
    }
}

/// Chooses child precision from a valid-value range and the parent scale.
///
/// Precision is at least 1 and at least the scale. An absent range represents empty or all-null
/// input. Null payloads **must** be excluded from the range.
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

/// The range of the valid decimal values that selects a decimal child.
enum ValidRange {
    /// No value is valid.
    Empty,
    /// The smallest and the largest valid value.
    Exact(RangeInclusive<i256>),
    /// Bounds that hold every valid value. They need not be valid values themselves.
    Bounds(RangeInclusive<i256>),
}

impl ValidRange {
    /// Returns the bounds that hold every valid value, or `None` when no value is valid.
    fn bounds(&self) -> Option<RangeInclusive<i256>> {
        match self {
            Self::Empty => None,
            Self::Exact(bounds) | Self::Bounds(bounds) => Some(bounds.clone()),
        }
    }
}

/// Scans valid values, keeping empty input distinct from supplied bounds.
fn scan_valid_range<F: NativeDecimalType>(values: &[F], mask: &Mask) -> ValidRange {
    match valid_range(values, mask) {
        Some((min, max)) => {
            let min = min.to_i256().vortex_expect("every decimal value fits i256");
            let max = max.to_i256().vortex_expect("every decimal value fits i256");
            ValidRange::Exact(min..=max)
        }
        None => ValidRange::Empty,
    }
}

/// Wraps a narrower child in a [`WidenArray`], or copies to the logical canonical width.
///
/// `mask` **must** be the mask of `validity`.
fn widen_or_canonical<F: NativeDecimalType>(
    values: Buffer<F>,
    validity: Validity,
    mask: &Mask,
    range: ValidRange,
    decimal_dtype: DecimalDType,
) -> VortexResult<ArrayRef> {
    if let Some(child) = narrowest_decimal_child(&values, &validity, mask, range, decimal_dtype)? {
        let dtype = DType::Decimal(decimal_dtype, validity.nullability());
        return Ok(WidenArray::try_new(child.into_array(), dtype)?.into_array());
    }

    let buffer = copy_to_canonical_width(values.as_slice(), mask, decimal_dtype)?;

    let array = DecimalArray::try_new_handle(
        buffer,
        decimal_dtype.canonical_values_type(),
        decimal_dtype,
        validity,
    )?;

    Ok(array.into_array())
}

/// Builds the narrowest canonical child, or returns `None` if no narrower width exists.
///
/// Every valid value **must** lie in `range`, and `mask` **must** be the mask of `validity`.
///
/// Reuses storage at the selected width. Other widths use [`copy_to_canonical_width`]. The child
/// caches `range` as its extrema, preserving exactness.
fn narrowest_decimal_child<F: NativeDecimalType>(
    values: &Buffer<F>,
    validity: &Validity,
    mask: &Mask,
    range: ValidRange,
    decimal_dtype: DecimalDType,
) -> VortexResult<Option<DecimalArray>> {
    let child_dtype = smallest_decimal_dtype(range.bounds(), decimal_dtype)?;
    let child_type = child_dtype.canonical_values_type();
    if child_type == decimal_dtype.canonical_values_type() {
        return Ok(None);
    }

    // The copy helper allocates even at the same width, so reuse matching storage here.
    let buffer = if F::DECIMAL_TYPE == child_type {
        BufferHandle::new_host(values.clone().into_byte_buffer())
    } else {
        copy_to_canonical_width(values.as_slice(), mask, child_dtype)?
    };

    let child = DecimalArray::try_new_handle(buffer, child_type, child_dtype, validity.clone())?;

    let extrema = match range {
        ValidRange::Empty => return Ok(Some(child)),
        ValidRange::Exact(bounds) => Precision::Exact(bounds),
        ValidRange::Bounds(bounds) => Precision::Inexact(bounds),
    };
    let min = extrema
        .clone()
        .map(|bounds| decimal_stat(*bounds.start(), child_dtype));
    let max = extrema.map(|bounds| decimal_stat(*bounds.end(), child_dtype));
    child.statistics().set(Stat::Min, min);
    child.statistics().set(Stat::Max, max);

    Ok(Some(child))
}

/// Builds a statistic at the child's canonical width. `value` **must** fit its precision.
fn decimal_stat(value: i256, decimal_dtype: DecimalDType) -> ScalarValue {
    let value = DecimalValue::try_from_i256(value, decimal_dtype)
        .vortex_expect("the selected child precision contains its range bounds");

    ScalarValue::Decimal(value)
}
