// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Splitting canonical decimal arrays into signed and unsigned parts.

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::DecimalType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::PType;
use vortex_array::dtype::i256;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use super::DecimalByteParts;
use super::DecimalBytePartsArray;
use super::LOWER_PART_BITS;
use super::MAX_I128_LOWER_PARTS;
use super::MAX_I256_LOWER_PARTS;

/// Create a [`DecimalBytePartsArray`] from a [`DecimalArray`] by splitting it into parts.
///
/// # Errors
///
/// Returns an error if the decimal cannot be split.
pub fn dbp_encode(
    decimal: &DecimalArray,
    exec_ctx: &mut ExecutionCtx,
) -> VortexResult<DecimalBytePartsArray> {
    let parts = split_decimal(decimal, exec_ctx)?;
    // SAFETY: splitting produces a signed MSP and zero, one, or three non-nullable u64 lower
    // parts, all with the decimal's length and in most-significant-first order. This also holds
    // for the constant parts used for empty and all-null inputs. The decimal dtype is preserved.
    Ok(unsafe {
        DecimalByteParts::new_unchecked(parts.msp, parts.lower_parts, decimal.decimal_dtype())
    })
}

/// A decimal array decomposed into byte parts.
pub struct DecimalParts {
    /// The signed most significant part. This carries the validity of the whole array.
    pub msp: ArrayRef,
    /// The unsigned 64-bit lower parts, most significant first.
    pub lower_parts: Vec<ArrayRef>,
}

impl DecimalParts {
    /// Construct decimal parts from the MSP buffer constituting a narrow decimal (`i64` or narrower).
    /// Narrow decimals have an MSP at most as wide as `i64` and no lower parts.
    fn from_narrow<T: NativePType>(values: Buffer<T>, validity: Validity) -> Self {
        Self {
            msp: PrimitiveArray::new(values, validity).into_array(),
            lower_parts: Vec::new(),
        }
    }

    /// Construct decimal parts arrays from the buffers constituting a wide decimal (`i128` or `i256`).
    /// Wide decimals have an `i64` MSP and up to [`super::MAX_LOWER_PARTS`] `u64` lower parts.
    fn from_wide(
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
/// Empty and all-null arrays use constant parts, preserving the part types and MSP's nullability.
///
/// # Errors
///
/// Returns an error if the array's validity cannot be derived or executed.
pub fn split_decimal(decimal: &DecimalArray, ctx: &mut ExecutionCtx) -> VortexResult<DecimalParts> {
    let validity = decimal.validity()?;
    let len = decimal.len();
    let mask = validity.execute_mask(len, ctx)?;

    if mask.all_false() || decimal.is_empty() {
        return Ok(split_no_valid_row(decimal, &validity));
    }

    Ok(match decimal.values_type() {
        DecimalType::I8 => DecimalParts::from_narrow(decimal.buffer::<i8>(), validity),
        DecimalType::I16 => DecimalParts::from_narrow(decimal.buffer::<i16>(), validity),
        DecimalType::I32 => DecimalParts::from_narrow(decimal.buffer::<i32>(), validity),
        DecimalType::I64 => DecimalParts::from_narrow(decimal.buffer::<i64>(), validity),
        DecimalType::I128 => {
            let (msp, lower) = split_wide(&decimal.buffer::<i128>(), &mask, i128_to_parts);
            DecimalParts::from_wide(msp, lower, validity)
        }
        DecimalType::I256 => {
            let (msp, lower) = split_wide(&decimal.buffer::<i256>(), &mask, i256_to_parts);
            DecimalParts::from_wide(msp, lower, validity)
        }
    })
}

/// Splits decimals with no valid rows (all null or empty) into constant decimal parts with the
/// corresponding nullability.
fn split_no_valid_row(decimal: &DecimalArray, validity: &Validity) -> DecimalParts {
    let (msp_ptype, lower_part_count) = match decimal.values_type() {
        DecimalType::I8 => (PType::I8, 0),
        DecimalType::I16 => (PType::I16, 0),
        DecimalType::I32 => (PType::I32, 0),
        DecimalType::I64 => (PType::I64, 0),
        DecimalType::I128 => (PType::I64, MAX_I128_LOWER_PARTS),
        DecimalType::I256 => (PType::I64, MAX_I256_LOWER_PARTS),
    };
    // Empty masks are also all-false. The default scalar is null for nullable inputs
    // and zero for non-nullable empty inputs, preserving the MSP's nullability.
    let msp = Scalar::default_value(&DType::Primitive(msp_ptype, validity.nullability()));
    let len = decimal.len();
    DecimalParts {
        msp: ConstantArray::new(msp, len).into_array(),
        lower_parts: vec![ConstantArray::new(0u64, len).into_array(); lower_part_count],
    }
}

/// Split wide integers into a signed MSP and `N` unsigned lower parts.
///
/// `to_parts` returns the MSP and lower words in most-significant-first order.
/// It is specialized for each input type: `i128` has one lower word and `i256`
/// has three. Null rows get zeros in every output buffer. The caller handles empty
/// and all-null arrays before calling this function.
pub fn split_wide<T: Copy, const N: usize>(
    values: &[T],
    validity: &Mask,
    to_parts: impl Fn(T) -> (i64, [u64; N]),
) -> (Buffer<i64>, [Buffer<u64>; N]) {
    let len = values.len();
    let mut msp = BufferMut::<i64>::with_capacity(len);
    let mut lower = std::array::from_fn::<_, N, _>(|_| BufferMut::<u64>::with_capacity(len));

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
        Mask::AllFalse(_) => unreachable!("all-null arrays are handled by split_decimal"),
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
pub const fn i128_to_parts(value: i128) -> (i64, [u64; MAX_I128_LOWER_PARTS]) {
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "each cast preserves a 64-bit window of the original two's complement bits"
    )]
    ((value >> LOWER_PART_BITS) as i64, [value as u64])
}

/// Extract the signed MSP and three unsigned lower words of an `i256`.
#[inline]
pub const fn i256_to_parts(value: i256) -> (i64, [u64; MAX_I256_LOWER_PARTS]) {
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

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;
    use vortex_array::arrays::DecimalArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DecimalDType;
    use vortex_array::dtype::PType;
    use vortex_array::dtype::i256;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use super::split_decimal;
    use crate::decimal_byte_parts::LOWER_PART_DTYPE;
    use crate::decimal_byte_parts::MAX_LOWER_PARTS;

    #[test]
    fn test_split_i256_part_count_and_types() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let decimal = DecimalArray::new(
            Buffer::from(vec![i256::from_i128(i128::MAX), i256::MIN]),
            DecimalDType::new(76, 0),
            Validity::NonNullable,
        );
        let parts = split_decimal(&decimal, &mut ctx)?;
        assert_eq!(parts.lower_parts.len(), MAX_LOWER_PARTS);
        assert_eq!(parts.msp.dtype().as_ptype(), PType::I64);
        for part in &parts.lower_parts {
            assert_eq!(part.dtype(), &LOWER_PART_DTYPE);
        }
        Ok(())
    }

    #[rstest]
    fn test_split_i256_part_order(
        #[values(Validity::NonNullable, Validity::from_iter([true, false, true]))]
        validity: Validity,
    ) -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let decimal = DecimalArray::new(
            buffer![
                i256::from_parts((2u128 << 64) | 3, (1i128 << 64) | 4),
                i256::ZERO,
                i256::from_parts((6u128 << 64) | 7, (-2i128 << 64) | 5),
            ],
            DecimalDType::new(76, 0),
            validity.clone(),
        );
        let parts = split_decimal(&decimal, &mut ctx)?;
        assert_arrays_eq!(
            PrimitiveArray::new(buffer![1i64, 0, -2], validity),
            parts.msp,
            &mut ctx
        );
        assert_eq!(parts.lower_parts.len(), 3);
        for (part, expected) in parts.lower_parts.into_iter().zip([
            buffer![4u64, 0, 5],
            buffer![2u64, 0, 6],
            buffer![3u64, 0, 7],
        ]) {
            assert_arrays_eq!(
                PrimitiveArray::new(expected, Validity::NonNullable),
                part,
                &mut ctx
            );
        }
        Ok(())
    }
}
