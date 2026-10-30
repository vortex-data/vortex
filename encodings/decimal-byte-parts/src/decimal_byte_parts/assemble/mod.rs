// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Reassembling decimal arrays and values from their parts.
//!
//! A lone most significant part (MSP) can stay narrow behind a [`Widen`]. Lower parts assemble at a
//! width bounded by the logical precision. Shared stored-decimal conversion selects a valid child
//! precision and caches its extrema for later aggregates.
//!
//! [`Widen`]: vortex_array::arrays::Widen

use std::ops::BitOr;
use std::ops::RangeInclusive;
use std::ops::Shl;

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::aggregate_fn::NumericalAggregateOpts;
use vortex_array::aggregate_fn::fns::min_max::min_max;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::WidenArray;
use vortex_array::arrays::decimal::DecimalDataParts;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::DecimalType;
use vortex_array::dtype::NativeDecimalType;
use vortex_array::dtype::i256;
use vortex_array::expr::stats::Precision;
use vortex_array::match_each_signed_integer_ptype;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_buffer::trusted_len::TrustedLen;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;

use super::DecimalBytePartsData;
use super::LOWER_PART_BITS;
use super::MAX_LOWER_PARTS;

/// Reassemble decimal parts into the canonical array of `decimal_dtype`, or into a decimal
/// [`Widen`] over a narrower child.
///
/// The parts are checked with [`DecimalBytePartsData::validate`], which owns the rules for part
/// dtypes, lengths and counts.
///
/// The parts first assemble at their _natural width_: the MSP width with no lower parts, `i128`
/// with one lower part, and `i256` with two or three. Because the validation bounds the lower
/// parts, only a lone MSP can be wider than the canonical width. Then
/// [`WidenArray::from_stored_decimal`] picks the result from the range of the valid values, so a
/// narrow MSP becomes the child of a `Widen` without a copy.
///
/// For a lone MSP that is not at the canonical width, the range comes from [`min_max`] on the MSP:
/// exact cached [`Stat::Min`] and [`Stat::Max`] statistics when the MSP has them, and a vectorized
/// pass over the MSP otherwise. [`WidenArray::from_stored_decimal`] decides whether to scan the
/// values assembled from lower parts.
///
/// # Errors
///
/// Returns an error if [`DecimalBytePartsData::validate`] rejects the parts, or if executing a part
/// fails. When the natural width is not the canonical width, it also returns an error if a valid
/// value is outside the precision of `decimal_dtype`.
///
/// [`Widen`]: vortex_array::arrays::Widen
/// [`Stat::Min`]: vortex_array::expr::stats::Stat::Min
/// [`Stat::Max`]: vortex_array::expr::stats::Stat::Max
pub fn assemble_decimal(
    msp: &ArrayRef,
    lower_parts: &[ArrayRef],
    decimal_dtype: DecimalDType,
    exec_ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let dtype = DType::Decimal(decimal_dtype, msp.dtype().nullability());
    DecimalBytePartsData::validate(msp, lower_parts.iter(), decimal_dtype, &dtype, msp.len())?;

    // TODO(mk): Broadcast constant parts directly instead of materializing their buffers.
    let msp = msp.clone().execute::<PrimitiveArray>(exec_ctx)?;
    let validity = msp.validity()?;

    if lower_parts.is_empty() {
        let values_type = DecimalType::try_from(msp.ptype())?;

        // At the canonical width, `from_stored_decimal` reuses the buffer without reading the
        // values, so a range pass would be wasted. A `None` range means that no value is valid, and
        // the scan in `from_stored_decimal` then finds no valid value either.
        let known_range = if values_type == decimal_dtype.canonical_values_type() {
            Precision::Absent
        } else {
            msp_range(&msp, exec_ctx)?.map_or(Precision::Absent, Precision::Exact)
        };

        let parts = DecimalDataParts {
            decimal_dtype,
            values: msp.buffer_handle().clone(),
            values_type,
            validity,
        };

        return WidenArray::from_stored_decimal(parts, known_range, exec_ctx);
    }

    let lower_parts = lower_parts
        .iter()
        .map(|part| part.clone().execute::<PrimitiveArray>(exec_ctx))
        .collect::<VortexResult<Vec<_>>>()?;
    let (values, values_type) = assemble_natural_width(&msp, &lower_parts);

    let parts = DecimalDataParts {
        decimal_dtype,
        values,
        values_type,
        validity,
    };

    WidenArray::from_stored_decimal(parts, Precision::Absent, exec_ctx)
}

/// Returns the range of the valid values of `msp`, or `None` when no value is valid.
///
/// [`min_max`] returns exact cached statistics without reading the values. Otherwise it makes one
/// vectorized pass over them and caches the result on `msp`. The decimal [`valid_range`] scan is
/// not vectorized, so it is not used here.
///
/// [`valid_range`]: vortex_array::aggregate_fn::fns::min_max::valid_range
fn msp_range(
    msp: &PrimitiveArray,
    exec_ctx: &mut ExecutionCtx,
) -> VortexResult<Option<RangeInclusive<i256>>> {
    let Some(extrema) = min_max(msp.as_ref(), exec_ctx, NumericalAggregateOpts::default())? else {
        return Ok(None);
    };

    let min = i64::try_from(&extrema.min)?;
    let max = i64::try_from(&extrema.max)?;

    Ok(Some(i256::from(min)..=i256::from(max)))
}

/// Assemble the MSP and one to three `u64` lower parts at their natural width: `i128` for one
/// lower part, and `i256` for two or three.
fn assemble_natural_width(
    msp: &PrimitiveArray,
    lower_parts: &[PrimitiveArray],
) -> (BufferHandle, DecimalType) {
    let (values, values_type) = match_each_signed_integer_ptype!(msp.ptype(), |Msp| {
        let msp = msp.as_slice::<Msp>();
        match lower_parts {
            [first] => (
                assemble_wide_decimal::<i128, Msp, 1>(
                    msp,
                    first.as_slice::<u64>().iter().map(|&word| [word]),
                )
                .into_byte_buffer(),
                DecimalType::I128,
            ),
            [first, second] => (
                assemble_wide_decimal::<i256, Msp, 2>(
                    msp,
                    first
                        .as_slice::<u64>()
                        .iter()
                        .zip(second.as_slice::<u64>())
                        .map(|(&a, &b)| [a, b]),
                )
                .into_byte_buffer(),
                DecimalType::I256,
            ),
            [first, second, third] => (
                assemble_wide_decimal::<i256, Msp, 3>(
                    msp,
                    first
                        .as_slice::<u64>()
                        .iter()
                        .zip(second.as_slice::<u64>())
                        .zip(third.as_slice::<u64>())
                        .map(|((&a, &b), &c)| [a, b, c]),
                )
                .into_byte_buffer(),
                DecimalType::I256,
            ),
            _ => vortex_panic!(
                "DecimalBytePartsData::validate bounds the lower parts to at most \
                 {MAX_LOWER_PARTS}, and assemble_decimal passes at least one, got {}",
                lower_parts.len()
            ),
        }
    });

    (BufferHandle::new_host(values), values_type)
}

/// Assemble a signed MSP slice and one array of `K` lower words per row.
///
/// The caller zips the `u64` lower-part slices into rows. The iterator must have the same
/// length as the MSP slice. MSP values are widened to `i64` as they are read.
pub fn assemble_wide_decimal<T, Msp, const K: usize>(
    msp: &[Msp],
    lower: impl TrustedLen<Item = [u64; K]>,
) -> Buffer<T>
where
    T: NativeDecimalType + From<i64> + From<u64> + Shl<usize, Output = T> + BitOr<Output = T>,
    Msp: Copy + Into<i64>,
{
    let mut out = BufferMut::<T>::with_capacity(msp.len());
    out.extend_trusted(
        msp.iter()
            .zip(lower)
            .map(|(&value, parts)| assemble_wide_decimal_value(value.into(), parts)),
    );
    out.freeze()
}

/// Reassemble a decimal's unscaled integer from its signed MSP and `K` lower words.
///
/// Sign-extend the MSP to `T`, then append each lower word by shifting left 64 bits and
/// filling the low bits. Lower words are ordered most significant first. Callers select
/// `i128` for one lower word and `i256` for two or three.
#[inline]
pub(crate) fn assemble_wide_decimal_value<T, const K: usize>(msp: i64, lower: [u64; K]) -> T
where
    T: NativeDecimalType + From<i64> + From<u64> + Shl<usize, Output = T> + BitOr<Output = T>,
{
    let mut value: T = msp.into();
    for part in lower {
        value = (value << LOWER_PART_BITS) | part.into();
    }
    value
}

#[cfg(test)]
mod tests;
