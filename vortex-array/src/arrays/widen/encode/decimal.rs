// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Decimal encoding and conversion of legacy storage into canonical decimal leaves.
//!
//! The legacy `vortex.decimal` format stores integer width independently of logical precision.
//! DecimalByteParts assembly can also produce such buffers, but we now require canonical
//! `vortex.decimal` leaves to have a width determined and fixed by their precision.
//!
//! In this new world, we can retain narrow storage as a [`WidenArray`] over a canonical child.
//! Current [`WidenPlugin`] writes still use the legacy format, so this conversion can also apply
//! to newly written data (which will happen until `Widen` is added to the `core` edition).
//!
//! We unify child selection in both encoding and legacy conversion because it has non-trivial
//! logic: width alone does not determine a valid child dtype. For example, unscaled 127 fits `i8`
//! but needs precision 3, whose canonical width is `i16`.
//!
//! [`WidenPlugin`]: crate::arrays::widen::WidenPlugin

use std::ops::RangeInclusive;

use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
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
    /// We keep the logical dtype and nullability by wrapping the narrower child in a
    /// [`WidenArray`]. Reducing precision within the same storage width saves no space, so in that
    /// case we return the input unchanged.
    ///
    /// Empty and all-null arrays have no values to constrain precision, but we still need to keep
    /// the scale. They use precision 1 or the scale, whichever is greater. If a child is selected,
    /// we cache any exact extrema as [`Stat::Min`] and [`Stat::Max`] for later aggregates.
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
            let valid_value_range =
                ValidRange::resolve(Precision::Absent, values_buffer.as_slice(), &validity_mask);
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

    /// Converts stored decimals into a canonical leaf or a [`WidenArray`] over one.
    ///
    /// Legacy decimal readers and DecimalByteParts assembly can give us a buffer whose width does
    /// not match the logical precision. We need a canonical leaf without changing the logical
    /// dtype or widening narrow storage unnecessarily.
    ///
    /// If the width is already canonical, we trust the declared precision and reuse the buffer
    /// without reading its values. This also lets device buffers stay on the device. Otherwise,
    /// child selection follows [`Self::encode_decimal`], keeping the logical dtype and validity.
    /// We reuse buffers at the selected width and convert other widths without rescaling.
    /// A narrowing copy writes zeros in null slots. A reused buffer keeps its null slots.
    ///
    /// Scanning and conversion use CPU kernels, so noncanonical buffers are brought to the host.
    /// Their validity **must** already be executable on the host because these kernels need a mask
    /// to exclude null payloads.
    ///
    /// Readers and decoders may already know the range. For noncanonical storage, `known_range`
    /// lets us reuse it:
    ///
    /// - [`Precision::Absent`] scans valid values to find their extrema.
    /// - [`Precision::Exact`] **must** give the actual minimum and maximum valid unscaled values.
    /// - [`Precision::Inexact`] **must** contain every valid unscaled value.
    ///
    /// We cache the range on the child as [`Stat::Min`] and [`Stat::Max`], keeping its exactness
    /// because enclosing bounds are not necessarily actual extrema. Scanning empty or all-null
    /// input yields no extrema to cache. Canonical-width storage ignores `known_range`.
    /// Supplied ranges are trusted, even when a narrowing copy checks the selected precision.
    /// Incorrect ranges or declared precision can cause incorrect results in later operations.
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
            vortex_ensure_eq!(
                validity_len,
                len,
                "Decimal validity must have one entry per value"
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
            let range = ValidRange::resolve(known_range, values.as_slice(), &mask);
            widen_or_canonical(values, validity, &mask, range, decimal_dtype)
        })
    }
}

/// Chooses child precision from a valid-value range and the parent scale.
///
/// We keep the parent scale so values do not need rescaling. The child precision must therefore
/// be at least 1 and at least the scale. With empty or all-null input, these are the only
/// constraints. Pass `None` for that case. Otherwise, null payloads **must** be excluded from the
/// range.
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

/// Value bounds for choosing child precision and caching statistics.
///
/// Supplied bounds need not be actual extrema, so we keep their exactness when caching them.
/// `Empty` means a scan found no valid values, rather than that statistics are unavailable.
enum ValidRange {
    /// No value is valid.
    Empty,
    /// The smallest and the largest valid value.
    Exact(RangeInclusive<i256>),
    /// Bounds that hold every valid value. They need not be valid values themselves.
    Bounds(RangeInclusive<i256>),
}

impl ValidRange {
    /// Uses known bounds when available, otherwise scans valid values.
    ///
    /// Readers and decoders may already have bounds from their input statistics. Reusing them
    /// avoids scanning the buffer again.
    fn resolve<F: NativeDecimalType>(
        known_range: Precision<RangeInclusive<i256>>,
        values: &[F],
        mask: &Mask,
    ) -> Self {
        match known_range {
            Precision::Exact(extrema) => return Self::Exact(extrema),
            Precision::Inexact(bounds) => return Self::Bounds(bounds),
            Precision::Absent => {}
        }

        let Some((min, max)) = valid_range(values, mask) else {
            return Self::Empty;
        };
        let min = min.to_i256().vortex_expect("every decimal value fits i256");
        let max = max.to_i256().vortex_expect("every decimal value fits i256");

        Self::Exact(min..=max)
    }

    /// Returns the bounds that hold every valid value, or `None` when no value is valid.
    fn bounds(&self) -> Option<RangeInclusive<i256>> {
        match self {
            Self::Empty => None,
            Self::Exact(bounds) | Self::Bounds(bounds) => Some(bounds.clone()),
        }
    }

    /// Caches bounds on the child so later aggregates can reuse them without another scan.
    ///
    /// The bounds **must** fit the child's selected precision.
    fn cache_statistics(self, child: &DecimalArray) {
        let extrema = match self {
            Self::Empty => return,
            Self::Exact(bounds) => Precision::Exact(bounds),
            Self::Bounds(bounds) => Precision::Inexact(bounds),
        };
        let child_dtype = child.decimal_dtype();
        let min = extrema
            .clone()
            .map(|bounds| decimal_stat(*bounds.start(), child_dtype));
        let max = extrema.map(|bounds| decimal_stat(*bounds.end(), child_dtype));
        child.statistics().set(Stat::Min, min);
        child.statistics().set(Stat::Max, max);
    }
}

/// Wraps a narrower canonical child, or builds a leaf at the logical width.
///
/// The result must keep the logical dtype. Returning a narrow child directly would change it.
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

    let array = build_canonical_decimal(&values, &validity, mask, decimal_dtype)?;

    Ok(array.into_array())
}

/// Builds the narrowest canonical child, or returns `None` if no narrower width exists.
///
/// Reducing precision without changing width saves no space, so we return `None` in that case.
///
/// Every valid value **must** lie in `range`, and `mask` **must** be the mask of `validity`.
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

    let child = build_canonical_decimal(values, validity, mask, child_dtype)?;
    range.cache_statistics(&child);

    Ok(Some(child))
}

/// Reuses storage at the dtype's canonical width, copying only when conversion is needed.
///
/// Every valid value **must** fit `decimal_dtype`, and `mask` **must** describe `validity`.
fn build_canonical_decimal<F: NativeDecimalType>(
    values: &Buffer<F>,
    validity: &Validity,
    mask: &Mask,
    decimal_dtype: DecimalDType,
) -> VortexResult<DecimalArray> {
    let values_type = decimal_dtype.canonical_values_type();
    let buffer = if F::DECIMAL_TYPE == values_type {
        // The copy helper allocates even at the same width, so retain the existing buffer here.
        BufferHandle::new_host(values.clone().into_byte_buffer())
    } else {
        copy_to_canonical_width(values.as_slice(), mask, decimal_dtype)?
    };

    DecimalArray::try_new_handle(buffer, values_type, decimal_dtype, validity.clone())
}

/// Represents a cached bound using the child dtype.
///
/// These statistics belong to the child, so their values need its canonical width rather than
/// the logical parent's width. `value` **must** fit the child's precision.
fn decimal_stat(value: i256, decimal_dtype: DecimalDType) -> ScalarValue {
    let value = DecimalValue::try_from_i256(value, decimal_dtype)
        .vortex_expect("the selected child precision contains its range bounds");

    ScalarValue::Decimal(value)
}
