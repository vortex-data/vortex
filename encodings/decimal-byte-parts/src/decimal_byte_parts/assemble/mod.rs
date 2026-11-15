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
use vortex_array::builtins::ArrayBuiltins;
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
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;

use super::LOWER_PART_BITS;
use super::LOWER_PART_DTYPE;
use super::MAX_LOWER_PARTS;

/// Reassemble decimal parts into a decimal array.
///
/// The MSP must have a signed integer dtype, and every lower part must have a non-nullable
/// unsigned integer dtype. All parts must have the same length.
///
/// With no lower parts, the MSP buffer is reused as the decimal values. One lower part
/// assembles into `i128`. Two or three lower parts assemble into `i256`.
///
/// If there are lower parts, each part (including the MSP) is widened into a 64 bit array.
/// For example, parts consisting of a `i8` MSP and a single `u32` lower part is assembled into
/// a decimal array with `64 + 64 = 128` bit storage.
///
/// # Errors
///
/// Returns an error for invalid part dtypes, lengths, or counts, or if executing a part fails.
pub fn assemble_decimal(
    msp: &ArrayRef,
    lower_parts: &[ArrayRef],
    decimal_dtype: DecimalDType,
    exec_ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    vortex_ensure!(
        msp.dtype().is_signed_int(),
        "MSP must have a signed integer dtype"
    );

    let validity = msp.validity()?;

    if lower_parts.is_empty() {
        return assemble_narrow_decimal(msp, validity, decimal_dtype, exec_ctx);
    }

    vortex_ensure!(
        lower_parts.len() <= MAX_LOWER_PARTS,
        "at most {MAX_LOWER_PARTS} lower parts are supported, got {}",
        lower_parts.len()
    );
    let len = msp.len();
    for (idx, part) in lower_parts.iter().enumerate() {
        vortex_ensure!(
            part.dtype().is_unsigned_int() && !part.dtype().is_nullable(),
            "lower part {idx} must have a non-nullable unsigned integer dtype, got {}",
            part.dtype()
        );
        vortex_ensure_eq!(part.len(), len, "lower part {idx} length mismatch");
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

/// Execute the MSP at its signed integer width and cast lower parts to `u64` before assembly.
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
    // TODO(mk): Revisit dispatching on lower-part dtypes and widening values during assembly.
    // Casting narrowed parts allocates temporary buffers and adds passes over the data.
    // Nested dtype dispatch is significantly in benchmarks, but adds code and generic instantiations.
    let lower = lower_parts
        .iter()
        .map(|part| {
            part.cast(LOWER_PART_DTYPE)?
                .execute::<PrimitiveArray>(exec_ctx)
        })
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
