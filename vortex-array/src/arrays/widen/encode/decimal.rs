// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Decimal child selection shared by encoding and stored-value reads.
//!
//! Precision labels must hold every valid value. Known ranges can avoid a scan, while exact
//! extrema cached on the selected child let later aggregates reuse that work.

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
use crate::dtype::Nullability;
use crate::dtype::i256;
use crate::expr::stats::Precision;
use crate::expr::stats::Stat;
use crate::match_each_decimal_value_type;
use crate::scalar::DecimalValue;
use crate::scalar::Scalar;
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
        let decimal_dtype = array.decimal_dtype();
        let validity = array.validity()?;
        let mask = validity.execute_mask(array.len(), ctx)?;
        let child = match_each_decimal_value_type!(array.values_type(), |T| {
            let values = array.buffer::<T>();
            let range = scan_valid_range(values.as_slice(), &mask);
            narrowest_decimal_child(&values, &validity, &mask, range, decimal_dtype)?
        });

        let Some(child) = child else {
            return Ok(array.into_array());
        };

        Ok(WidenArray::try_new(child.into_array(), array.dtype().clone())?.into_array())
    }

    /// Builds the canonical array, or a decimal `Widen`, from decimal values stored at any width.
    ///
    /// This is for readers of decimal values whose stored width can differ from the canonical
    /// width of their dtype, such as old `vortex.decimal` nodes and decoded DecimalByteParts.
    /// `parts.values_type` is the stored width, and the result has the dtype `parts.decimal_dtype`
    /// with the nullability of `parts.validity`.
    ///
    /// - A stored width that equals the canonical width returns the canonical array with the same
    ///   buffer, which can live on a device. It reads no values and trusts the precision, so a
    ///   valid value outside the precision gives an incorrect result, but never a panic.
    /// - Any other stored width copies the buffer to the host and picks the smallest decimal dtype
    ///   that holds the valid values, as [`Self::encode_decimal`] does. When that dtype has a
    ///   narrower canonical width, the result is a `Widen` over it, and the child reuses the
    ///   buffer when the stored width already is the child canonical width. Otherwise the result
    ///   is the canonical array. Each copy writes zeros in the null slots when it narrows.
    ///
    /// `known_range` saves the scan of the valid values, and a canonical stored width ignores it.
    /// [`Precision::Absent`] scans, and the child records the exact extrema of the scan. Every
    /// valid value **must** lie in a known range:
    ///
    /// - A [`Precision::Exact`] range **must** be the smallest and the largest valid value, for
    ///   example from exact cached statistics. The child records it as exact [`Stat::Min`] and
    ///   [`Stat::Max`], so a later [`min_max`](crate::aggregate_fn::fns::min_max::min_max)
    ///   reads them without a scan.
    /// - A [`Precision::Inexact`] range only bounds the valid values. A looser bound gives a wider
    ///   child, and the child records the bounds as inexact extrema.
    ///
    /// A range that misses a valid value fails where the copy narrows, and gives an incorrect
    /// result otherwise.
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer length is not a multiple of the stored width, if the buffer
    /// is not aligned to the stored width, or if a validity array does not have one entry per
    /// value. When the stored width is not canonical, it also returns an error if a copy to the
    /// host fails, if a valid value is outside the precision of `parts.decimal_dtype`, if
    /// `known_range` is reversed or reaches outside that precision, or if `known_range` misses a
    /// valid value and the copy to the child narrows.
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

        // NB: `execute_mask` and `valid_range` assert on a length mismatch, and a read must never
        // panic, so the parts are checked before any of them runs.
        DecimalData::validate(&values, values_type)?;
        let len = values.len() / values_type.byte_width();
        if let Some(validity_len) = validity.maybe_len() {
            vortex_ensure!(
                validity_len == len,
                "Decimal validity must have one entry per value ({len}), got {validity_len}"
            );
        }

        if values_type == decimal_dtype.canonical_values_type() {
            // SAFETY: the buffer has the canonical width, and its length and alignment are
            // checked above, as is the length of the validity. Reads trust the precision of the
            // stored dtype, so a valid value outside it gives an incorrect result.
            let array = unsafe {
                DecimalArray::new_unchecked_handle(values, values_type, decimal_dtype, validity)
            };
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

/// Returns the exact range of the valid values, with [`valid_range`].
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

/// Returns a `Widen` over the narrowest decimal child of `values`, or the canonical array of
/// `decimal_dtype` when no child is narrower than the canonical width.
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

    // SAFETY: the copy has the canonical width and one value per validity entry. A narrowing copy
    // checks every valid value against the precision. A widening copy needs no check, because a
    // type narrower than the canonical width holds no value outside the precision.
    let array = unsafe {
        DecimalArray::new_unchecked_handle(
            buffer,
            decimal_dtype.canonical_values_type(),
            decimal_dtype,
            validity,
        )
    };

    Ok(array.into_array())
}

/// Returns `values` at the canonical width of the smallest decimal dtype that holds `range`, or
/// `None` when that dtype has the canonical width of `decimal_dtype`.
///
/// [`WidenArray::encode_decimal`] and [`WidenArray::from_stored_decimal`] share this function, so
/// that one function picks the child label, whose precision **must** hold every valid value.
/// Every valid value **must** lie in `range`, and `mask` **must** be the mask of `validity`.
///
/// The child reuses `values` when their width already is its canonical width. Otherwise
/// [`copy_to_canonical_width`] widens them, or narrows them and checks every valid value against
/// the child precision. A narrowing copy catches a `range` that misses a valid value, but a reused
/// buffer is not checked, so there such a `range` gives an incorrect result.
///
/// The child records the bounds of `range` as its [`Stat::Min`] and [`Stat::Max`], exact only
/// when they are the exact extrema.
///
/// # Errors
///
/// Returns an error if `range` is outside the precision of `decimal_dtype`, or if a narrowing copy
/// finds a valid value outside the child precision.
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

    // NB: `copy_to_canonical_width` always copies, so a buffer that already has the child width
    // is reused here instead.
    let buffer = if F::DECIMAL_TYPE == child_type {
        BufferHandle::new_host(values.clone().into_byte_buffer())
    } else {
        copy_to_canonical_width(values.as_slice(), mask, child_dtype)?
    };

    // SAFETY: the buffer has the child canonical width and one value per validity entry. Every
    // valid value lies in `range`, which fits the child precision: the scan in `scan_valid_range`
    // establishes this, or else the `known_range` contract of `WidenArray::from_stored_decimal`
    // does. A narrowing copy also checks every valid value against the child precision. A reused
    // buffer is not checked, so a `known_range` that misses a valid value gives an incorrect
    // result, as reads trust the precision.
    let child = unsafe {
        DecimalArray::new_unchecked_handle(buffer, child_type, child_dtype, validity.clone())
    };

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

/// Returns `value` as a statistic of an array with the dtype `decimal_dtype`, at its canonical
/// width. `value` **must** fit the precision of `decimal_dtype`.
fn decimal_stat(value: i256, decimal_dtype: DecimalDType) -> ScalarValue {
    let value = match_each_decimal_value_type!(decimal_dtype.canonical_values_type(), |W| {
        DecimalValue::from(
            DecimalValue::I256(value)
                .cast::<W>()
                .vortex_expect("a value within the precision fits its canonical width"),
        )
    });
    let scalar = Scalar::decimal(value, decimal_dtype, Nullability::NonNullable);

    scalar
        .into_value()
        .vortex_expect("a decimal scalar built from a value is not null")
}
