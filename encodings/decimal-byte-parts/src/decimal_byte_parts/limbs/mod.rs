// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Splitting decimal values into 64-bit parts and reassembling them.
//!
//! A `DecimalByteParts` array stores each value as a signed most significant part (MSP)
//! followed by `k` unsigned 64-bit lower parts ordered most significant first. The encoded
//! value is
//!
//! ```text
//! msp * 2^(64k) + Σ_{i<k} lower[i] * 2^(64 * (k - 1 - i))
//! ```
//!
//! This is exactly the two's complement bit pattern of the decimal value cut on 64-bit
//! boundaries.

use std::ops::BitOr;
use std::ops::Shl;

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::DecimalType;
use vortex_array::dtype::NativeDecimalType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::i256;
use vortex_array::match_each_signed_integer_ptype;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_mask::Mask;

/// The maximum number of lower parts an encoded decimal can carry. Each is 64 bits.
///
/// Since the MSP is at most 64 bits wide, three additional 64-bit parts saturates
/// the 256-bit maximum width of a Vortex decimal.
pub const MAX_LOWER_PARTS: usize = 3;

/// Number of bits stored in each lower part.
const LOWER_PART_BITS: usize = 64;

/// Every lower part is a non-nullable `u64` primitive, since the MSP carries the sign
/// and validity.
pub(crate) const LOWER_PART_DTYPE: DType = DType::Primitive(PType::U64, Nullability::NonNullable);

/// A decimal array decomposed into byte parts.
pub struct DecimalParts {
    /// The signed most significant part. This carries the validity of the whole array.
    pub msp: ArrayRef,
    /// The unsigned 64-bit lower parts, most significant first.
    pub lower_parts: Vec<ArrayRef>,
}

impl DecimalParts {
    /// Construct decimal parts from an MSP with no lower parts.
    fn from_msp<T: NativePType>(values: Buffer<T>, validity: Validity) -> Self {
        Self {
            msp: PrimitiveArray::new(values, validity).into_array(),
            lower_parts: Vec::new(),
        }
    }

    fn new(
        msp: Buffer<i64>,
        lower_parts: impl IntoIterator<Item = Buffer<u64>>,
        validity: Validity,
    ) -> Self {
        Self {
            msp: PrimitiveArray::new(msp, validity).into_array(),
            lower_parts: lower_parts
                .into_iter()
                .map(|part| PrimitiveArray::new(part, Validity::NonNullable).into_array())
                .collect(),
        }
    }
}

/// Split a canonical decimal array into a signed most significant part (MSP) and unsigned 64-bit
/// lower parts. The MSP is at most 64 bits.
///
/// Values narrower than 128 bits are already a single signed part, so they are returned
/// with no lower parts. `i128` values split into an `i64` MSP and one lower part. `i256`
/// values split into an `i64` MSP and three lower parts.
///
/// The MSP retains the decimal's validity while lower parts are non-nullable. Lower parts
/// are constructed with zeroes at null positions instead of invalid bytes.
///
/// # Errors
///
/// Returns an error if the array's validity cannot be derived or executed.
pub fn split_decimal(decimal: &DecimalArray, ctx: &mut ExecutionCtx) -> VortexResult<DecimalParts> {
    let validity = decimal.validity()?;
    Ok(match decimal.values_type() {
        DecimalType::I8 => DecimalParts::from_msp(decimal.buffer::<i8>(), validity),
        DecimalType::I16 => DecimalParts::from_msp(decimal.buffer::<i16>(), validity),
        DecimalType::I32 => DecimalParts::from_msp(decimal.buffer::<i32>(), validity),
        DecimalType::I64 => DecimalParts::from_msp(decimal.buffer::<i64>(), validity),
        DecimalType::I128 => {
            let mask = validity.execute_mask(decimal.len(), ctx)?;
            let (msp, lower) = split_wide(&decimal.buffer::<i128>(), &mask, i128_to_parts);
            DecimalParts::new(msp, lower, validity)
        }
        DecimalType::I256 => {
            let mask = validity.execute_mask(decimal.len(), ctx)?;
            let (msp, lower) = split_wide(&decimal.buffer::<i256>(), &mask, i256_to_parts);
            DecimalParts::new(msp, lower, validity)
        }
    })
}

/// Split wide integers into a signed MSP and `N` unsigned lower parts.
///
/// `to_parts` returns the MSP and lower words in most-significant-first order.
/// It is specialized for each input type: `i128` has one lower word and `i256`
/// has three. Null rows get zeros in every output buffer.
fn split_wide<T: Copy, const N: usize>(
    values: &Buffer<T>,
    validity: &Mask,
    to_parts: impl Fn(T) -> (i64, [u64; N]),
) -> (Buffer<i64>, [Buffer<u64>; N]) {
    let len = values.len();
    let mut msp = BufferMut::<i64>::with_capacity(len);
    let mut lower = std::array::from_fn::<_, N, _>(|_| BufferMut::<u64>::with_capacity(len));

    // Zero out all parts if all null
    if validity.all_false() {
        msp.push_n(0, len);
        for part in &mut lower {
            part.push_n(0, len);
        }
        return (msp.freeze(), lower.map(BufferMut::freeze));
    }

    // Allocate without zeroing, then initialize every part of each row together.
    let msp_out = &mut msp.spare_capacity_mut()[..len];
    let mut lower_out = lower
        .each_mut()
        .map(|part| &mut part.spare_capacity_mut()[..len]);

    match validity {
        Mask::AllTrue(_) => {
            for row in 0..len {
                let (high, words) = to_parts(values[row]);
                msp_out[row].write(high);
                for (part, word) in lower_out.iter_mut().zip(words) {
                    part[row].write(word);
                }
            }
        }
        Mask::Values(validity) => {
            // A shorter bitmap would leave output slots uninitialized before set_len.
            assert_eq!(
                validity.bit_buffer().len(),
                len,
                "values and validity must have the same length"
            );
            for (chunk_index, ((chunk, bits), msp)) in values
                .chunks(64)
                .zip(validity.bit_buffer().chunks().iter_padded())
                .zip(msp_out.chunks_mut(64))
                .enumerate()
            {
                for (i, (&value, msp)) in chunk.iter().zip(msp).enumerate() {
                    let mask = 0u64.wrapping_sub((bits >> i) & 1);
                    let (high, words) = to_parts(value);
                    msp.write(high & mask.cast_signed());
                    for (part, word) in lower_out.iter_mut().zip(words) {
                        part[chunk_index * 64 + i].write(word & mask);
                    }
                }
            }
        }
        Mask::AllFalse(_) => unreachable!("AllFalse case addressed above"),
    }

    // SAFETY: the input and all output slices have len elements. Both branches
    // initialize every slot, including null rows and the final partial chunk.
    // The bitmap length check prevents the masked iteration from ending early.
    unsafe {
        msp.set_len(len);
        for part in &mut lower {
            part.set_len(len);
        }
    }
    (msp.freeze(), lower.map(BufferMut::freeze))
}

/// Extract the high signed word and low unsigned word of an `i128`.
#[inline]
const fn i128_to_parts(value: i128) -> (i64, [u64; 1]) {
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "each cast preserves a 64-bit window of the original two's complement bits"
    )]
    ((value >> LOWER_PART_BITS) as i64, [value as u64])
}

/// Extract the signed MSP and three unsigned lower words of an `i256`.
#[inline]
const fn i256_to_parts(value: i256) -> (i64, [u64; MAX_LOWER_PARTS]) {
    let (low, high) = value.to_parts();
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "each cast preserves a 64-bit window of the original two's complement bits"
    )]
    (
        (high >> LOWER_PART_BITS) as i64,
        [high as u64, (low >> LOWER_PART_BITS) as u64, low as u64],
    )
}

/// Reassemble primitive arrays that constitute decimal byte parts into a canonical decimal array.
///
/// The MSP must be signed. There must be between zero and three (inclusive) `u64` lower parts, ordered
/// most significant first. The lower parts must be non-nullable. Every input array must have the same length.
///
/// With no lower parts, the MSP buffer is reused as the decimal values. One lower part
/// assembles into `i128`. Two or three lower parts assemble into `i256`.
///
/// # Errors
///
/// Returns an error if the parts do not describe a valid decimal, or if the MSP's validity
/// cannot be derived.
pub fn assemble_decimal(
    msp: &PrimitiveArray,
    lower_parts: &[PrimitiveArray],
    decimal_dtype: DecimalDType,
) -> VortexResult<DecimalArray> {
    let validity = msp.validity()?;
    vortex_ensure!(msp.dtype().as_ptype().is_signed_int());

    if lower_parts.is_empty() {
        return Ok(match_each_signed_integer_ptype!(msp.ptype(), |P| {
            // SAFETY: the buffer is typed by the array's own ptype, the decimal dtype is the
            // array's, and the validity is taken from the same array.
            unsafe { DecimalArray::new_unchecked(msp.to_buffer::<P>(), decimal_dtype, validity) }
        }));
    }

    let len = msp.len();
    let lower: Vec<&[u64]> = lower_parts
        .iter()
        .map(|part| {
            vortex_ensure!(
                part.dtype() == &LOWER_PART_DTYPE,
                "lower part must be non-nullable u64"
            );
            let part = part.as_slice::<u64>();
            vortex_ensure!(
                part.len() == len,
                "lower part has len {}, expected {len}",
                part.len()
            );
            Ok(part)
        })
        .collect::<VortexResult<_>>()?;

    Ok(match lower.as_slice() {
        [first] => DecimalArray::new(
            assemble_wide::<i128, 1>(msp, [first]),
            decimal_dtype,
            validity,
        ),
        [first, second] => DecimalArray::new(
            assemble_wide::<i256, 2>(msp, [first, second]),
            decimal_dtype,
            validity,
        ),
        [first, second, third] => DecimalArray::new(
            assemble_wide::<i256, 3>(msp, [first, second, third]),
            decimal_dtype,
            validity,
        ),
        _ => vortex_bail!(
            "at most {MAX_LOWER_PARTS} lower parts are supported, got {}",
            lower.len()
        ),
    })
}

/// Reassemble a signed MSP and `K` unsigned lower parts into wide integers.
///
/// Each row starts with the MSP sign-extended to `T`. Appending a lower word shifts the
/// accumulated value left by 64 bits and fills the low bits with that word. Lower parts
/// are appended most significant first.
///
/// The callers select `i128` for one lower part and `i256` for two or three. Since `K`
/// is constant, the compiler can unroll the loop that appends the lower words.
fn assemble_wide<T, const K: usize>(msp: &PrimitiveArray, lower: [&[u64]; K]) -> Buffer<T>
where
    T: NativeDecimalType + Shl<usize, Output = T> + BitOr<Output = T>,
{
    let mut out = BufferMut::<T>::with_capacity(msp.len());
    match_each_signed_integer_ptype!(msp.ptype(), |P| {
        out.extend_trusted(msp.as_slice::<P>().iter().enumerate().map(|(row, value)| {
            #[allow(
                clippy::useless_conversion,
                reason = "the widening to i64 is a no-op only for the i64 arm of the ptype match"
            )]
            let mut value = T::from(i64::from(*value)).vortex_expect("MSP fits in the output type");
            for part in lower {
                value = (value << LOWER_PART_BITS)
                    | T::from(part[row]).vortex_expect("lower word fits in the output type");
            }
            value
        }));
    });
    out.freeze()
}

#[cfg(test)]
mod tests;
