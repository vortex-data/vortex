// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Reassembling decimal arrays and values from their parts.
//!
//! Assembly preserves the [encoding contract](crate) and returns the canonical width. The natural
//! part width can be smaller, in which case the canonical-width copy sign-extends the values.

use std::ops::BitOr;
use std::ops::Shl;

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::Widen;
use vortex_array::arrays::decimal::copy_to_canonical_width;
use vortex_array::arrays::widen::WidenArraySlotsExt;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::DecimalType;
use vortex_array::dtype::NativeDecimalType;
use vortex_array::dtype::i256;
use vortex_array::match_each_decimal_value_type;
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

/// Reassemble validated parts into the canonical decimal array.
///
/// See [`DecimalBytePartsData::validate`] for the part requirements. A lone MSP stored at the
/// canonical width reuses its executed buffer. Narrow MSP storage assembles directly from the
/// integer [`Widen`]'s child, avoiding an intermediate widening copy. Other shapes assemble at
/// their natural width, then copy into the canonical width when it is wider. No value-range scan is
/// needed. See the [encoding contract](crate).
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

    // TODO(connor): Broadcast constant parts directly instead of materializing their buffers.
    let stored_msp = match msp.as_opt::<Widen>() {
        Some(widen) => widen.values().clone(),
        None => msp.clone(),
    };
    let msp = stored_msp.execute::<PrimitiveArray>(exec_ctx)?;
    let validity = msp.validity()?;

    let (mut values, values_type) = if lower_parts.is_empty() {
        (
            msp.buffer_handle().clone(),
            DecimalType::try_from(msp.ptype())?,
        )
    } else {
        let lower_parts = lower_parts
            .iter()
            .map(|part| part.clone().execute::<PrimitiveArray>(exec_ctx))
            .collect::<VortexResult<Vec<_>>>()?;
        assemble_natural_width(&msp, &lower_parts)
    };

    let canonical_type = decimal_dtype.canonical_values_type();
    if values_type < canonical_type {
        let mask = Mask::new_true(msp.len());
        values = match_each_decimal_value_type!(values_type, |T| {
            let values = Buffer::<T>::from_byte_buffer(values.as_host().clone());
            copy_to_canonical_width(&values, &mask, decimal_dtype)?
        });
    }

    Ok(DecimalArray::try_new_handle(values, canonical_type, decimal_dtype, validity)?.into_array())
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
