// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Reassembling decimal arrays and values from their parts.
//!
//! Assembly preserves the [part layout](crate) and returns values at the canonical width, as
//! [`assemble_decimal`] describes.

use std::ops::BitOr;
use std::ops::Shl;

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::decimal::copy_to_canonical_width;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::DecimalType;
use vortex_array::dtype::NativeDecimalType;
use vortex_array::dtype::i256;
use vortex_array::match_each_signed_integer_ptype;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_buffer::trusted_len::TrustedLen;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;
use vortex_mask::Mask;

use super::DecimalBytePartsData;
use super::LOWER_PART_BITS;
use super::MAX_LOWER_PARTS;
use super::stored_msp;

/// Reassemble validated parts into the canonical decimal array.
///
/// See [`DecimalBytePartsData::validate`] for the part requirements. Assembly reads the narrow
/// child of a [`Widen`] MSP, which avoids an intermediate widening copy. A lone MSP at the
/// canonical width reuses its executed buffer, and a narrower lone MSP gets one sign-extending
/// copy. Lower parts assemble straight into the canonical width. No value-range scan is needed, as
/// the [part layout](crate) explains.
///
/// # Errors
///
/// Returns an error if validation or part execution fails.
///
/// [`Widen`]: vortex_array::arrays::Widen
pub fn assemble_decimal(
    msp: &ArrayRef,
    lower_parts: &[ArrayRef],
    decimal_dtype: DecimalDType,
    exec_ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let dtype = DType::Decimal(decimal_dtype, msp.dtype().nullability());
    DecimalBytePartsData::validate(msp, lower_parts.iter(), decimal_dtype, &dtype, msp.len())?;

    // TODO(connor): Broadcast constant parts directly instead of materializing their buffers.
    let msp = stored_msp(msp).execute::<PrimitiveArray>(exec_ctx)?;
    let validity = msp.validity()?;

    let values = if lower_parts.is_empty() {
        msp_at_canonical_width(&msp, decimal_dtype)?
    } else {
        let lower_parts = lower_parts
            .iter()
            .map(|part| part.clone().execute::<PrimitiveArray>(exec_ctx))
            .collect::<VortexResult<Vec<_>>>()?;
        assemble_canonical_width(&msp, &lower_parts, decimal_dtype)
    };

    let canonical_type = decimal_dtype.canonical_values_type();
    Ok(DecimalArray::try_new_handle(values, canonical_type, decimal_dtype, validity)?.into_array())
}

/// Return the values of a lone MSP at the canonical width of `decimal_dtype`.
///
/// An MSP at the canonical width reuses its buffer. A narrower MSP gets a sign-extending copy.
fn msp_at_canonical_width(
    msp: &PrimitiveArray,
    decimal_dtype: DecimalDType,
) -> VortexResult<BufferHandle> {
    if DecimalType::try_from(msp.ptype())? == decimal_dtype.canonical_values_type() {
        return Ok(msp.buffer_handle().clone());
    }

    // NB: A stored MSP is never wider than the canonical width, so this copy only widens. A
    // widening copy checks no values and does not read the mask, so the all-true mask is safe even
    // when the MSP has nulls.
    let mask = Mask::new_true(msp.len());
    Ok(match_each_signed_integer_ptype!(msp.ptype(), |T| {
        copy_to_canonical_width(msp.as_slice::<T>(), &mask, decimal_dtype)?
    }))
}

/// Assemble the MSP and one to three `u64` lower parts at the canonical width of `decimal_dtype`.
///
/// [`DecimalBytePartsData::validate`] bounds the lower-part count by the precision, so the
/// canonical width is at least `i128` for one lower part and `i256` for two or three. One lower
/// part assembles into `i128` or `i256`, whichever is canonical.
fn assemble_canonical_width(
    msp: &PrimitiveArray,
    lower_parts: &[PrimitiveArray],
    decimal_dtype: DecimalDType,
) -> BufferHandle {
    let canonical_type = decimal_dtype.canonical_values_type();
    let values = match_each_signed_integer_ptype!(msp.ptype(), |Msp| {
        let msp = msp.as_slice::<Msp>();
        match lower_parts {
            [first] => {
                let words = first.as_slice::<u64>().iter().map(|&word| [word]);
                if canonical_type == DecimalType::I256 {
                    assemble_wide_decimal::<i256, Msp, 1>(msp, words).into_byte_buffer()
                } else {
                    assemble_wide_decimal::<i128, Msp, 1>(msp, words).into_byte_buffer()
                }
            }
            [first, second] => assemble_wide_decimal::<i256, Msp, 2>(
                msp,
                first
                    .as_slice::<u64>()
                    .iter()
                    .zip(second.as_slice::<u64>())
                    .map(|(&a, &b)| [a, b]),
            )
            .into_byte_buffer(),
            [first, second, third] => assemble_wide_decimal::<i256, Msp, 3>(
                msp,
                first
                    .as_slice::<u64>()
                    .iter()
                    .zip(second.as_slice::<u64>())
                    .zip(third.as_slice::<u64>())
                    .map(|((&a, &b), &c)| [a, b, c]),
            )
            .into_byte_buffer(),
            _ => vortex_panic!(
                "DecimalBytePartsData::validate bounds the lower parts to at most \
                 {MAX_LOWER_PARTS}, and assemble_decimal passes at least one, got {}",
                lower_parts.len()
            ),
        }
    });

    BufferHandle::new_host(values)
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
/// filling the low bits. Lower words are ordered most significant first. `T` must be at least
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
