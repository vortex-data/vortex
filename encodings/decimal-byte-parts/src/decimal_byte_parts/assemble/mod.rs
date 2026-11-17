// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Reassembling decimal arrays and values from their parts.

use std::ops::BitOr;
use std::ops::Shl;

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::NativeDecimalType;
use vortex_array::dtype::i256;
use vortex_array::match_each_signed_integer_ptype;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_buffer::trusted_len::TrustedLen;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use super::DecimalBytePartsData;
use super::LOWER_PART_BITS;
use super::MAX_LOWER_PARTS;

/// Reassemble decimal parts into a decimal array.
///
/// See [`DecimalBytePartsData::validate`] for the part requirements.
///
/// With no lower parts, the MSP buffer is reused as the decimal values. One lower part
/// assembles into `i128`. Two or three lower parts assemble into `i256`.
///
/// # Errors
///
/// Returns an error if validation or part execution fails.
pub fn assemble_decimal(
    msp: &ArrayRef,
    lower_parts: &[ArrayRef],
    decimal_dtype: DecimalDType,
    exec_ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let dtype = DType::Decimal(decimal_dtype, msp.dtype().nullability());
    DecimalBytePartsData::validate(msp, lower_parts.iter(), decimal_dtype, &dtype, msp.len())?;

    let validity = msp.validity()?;

    if lower_parts.is_empty() {
        return assemble_narrow_decimal(msp, validity, decimal_dtype, exec_ctx);
    }

    assemble_wide_decimal_from_arrays(msp, lower_parts, validity, decimal_dtype, exec_ctx)
}

fn assemble_narrow_decimal(
    msp: &ArrayRef,
    validity: Validity,
    decimal_dtype: DecimalDType,
    exec_ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    // TODO(mk): Broadcast a constant MSP directly instead of materializing its buffer.
    let msp = msp.clone().execute::<PrimitiveArray>(exec_ctx)?;
    Ok(match_each_signed_integer_ptype!(msp.ptype(), |P| {
        DecimalArray::new(msp.to_buffer::<P>(), decimal_dtype, validity).into_array()
    }))
}

/// Execute the MSP at its signed integer width and the `u64` lower parts before assembly.
/// The number of lower parts determines the decimal storage type: one produces `i128`, while
/// two or three produce `i256`.
fn assemble_wide_decimal_from_arrays(
    msp: &ArrayRef,
    lower_parts: &[ArrayRef],
    validity: Validity,
    decimal_dtype: DecimalDType,
    exec_ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    // TODO(mk): Broadcast constant parts directly instead of materializing their buffers.
    let msp = msp.clone().execute::<PrimitiveArray>(exec_ctx)?;
    let lower = lower_parts
        .iter()
        .map(|part| part.clone().execute::<PrimitiveArray>(exec_ctx))
        .collect::<VortexResult<Vec<_>>>()?;

    Ok(match_each_signed_integer_ptype!(msp.ptype(), |Msp| {
        let msp = msp.as_slice::<Msp>();
        match lower.as_slice() {
            [first] => DecimalArray::new(
                assemble_wide_decimal::<i128, Msp, 1>(
                    msp,
                    first.as_slice::<u64>().iter().map(|&word| [word]),
                ),
                decimal_dtype,
                validity,
            )
            .into_array(),
            [first, second] => DecimalArray::new(
                assemble_wide_decimal::<i256, Msp, 2>(
                    msp,
                    first
                        .as_slice::<u64>()
                        .iter()
                        .zip(second.as_slice::<u64>())
                        .map(|(&a, &b)| [a, b]),
                ),
                decimal_dtype,
                validity,
            )
            .into_array(),
            [first, second, third] => DecimalArray::new(
                assemble_wide_decimal::<i256, Msp, 3>(
                    msp,
                    first
                        .as_slice::<u64>()
                        .iter()
                        .zip(second.as_slice::<u64>())
                        .zip(third.as_slice::<u64>())
                        .map(|((&a, &b), &c)| [a, b, c]),
                ),
                decimal_dtype,
                validity,
            )
            .into_array(),
            _ => {
                vortex_bail!("expected between one and {MAX_LOWER_PARTS} lower parts")
            }
        }
    }))
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
