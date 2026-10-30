// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Splitting canonical decimal arrays into signed and unsigned parts.

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::decimal::converted_buffer;
use vortex_array::arrays::decimal::valid_range;
use vortex_array::dtype::BigCast;
use vortex_array::dtype::DType;
use vortex_array::dtype::DecimalType;
use vortex_array::dtype::NativeDecimalType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::PType;
use vortex_array::dtype::i256;
use vortex_array::match_each_decimal_value_type;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_mask::Mask;

use super::DecimalByteParts;
use super::DecimalBytePartsArray;
use super::LOWER_PART_BITS;
use super::MAX_I128_LOWER_PARTS;
use super::MAX_I256_LOWER_PARTS;

impl DecimalByteParts {
    /// Encode a [`DecimalArray`] as byte parts, with as many lower parts as its valid values need
    /// (see [`lower_part_count`]).
    ///
    /// # Errors
    ///
    /// Returns an error if the validity cannot be executed, or if a valid value needs more lower
    /// parts than the canonical width of the dtype allows (see
    /// [`super::DecimalBytePartsData::validate`]). Only a value outside the precision needs them.
    pub fn encode(
        decimal: &DecimalArray,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<DecimalBytePartsArray> {
        let parts = split_decimal(decimal, lower_part_count(decimal, exec_ctx)?, exec_ctx)?;
        Self::try_new_with_lower_parts(parts.msp, parts.lower_parts, decimal.decimal_dtype())
    }
}

/// A decimal array decomposed into byte parts.
pub struct DecimalParts {
    /// The signed most significant part. This carries the validity of the whole array.
    pub msp: ArrayRef,
    /// The unsigned 64-bit lower parts, most significant first.
    pub lower_parts: Vec<ArrayRef>,
}

impl DecimalParts {
    /// Construct decimal parts with no lower parts from an MSP buffer of `i64` or narrower values:
    /// either the decimal's own narrow storage, or a copy of wide values that all fit `i64`.
    fn from_narrow<T: NativePType>(values: Buffer<T>, validity: Validity) -> Self {
        Self {
            msp: PrimitiveArray::new(values, validity).into_array(),
            lower_parts: Vec::new(),
        }
    }

    /// Construct decimal parts arrays from the buffers constituting a wide decimal (`i128` or `i256`).
    /// Wide decimals have an `i64` MSP and up to `MAX_LOWER_PARTS` `u64` lower parts.
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

/// The number of lower parts that the valid values of `decimal` need: 0 when every valid value
/// fits `i64`, 1 when every valid value fits `i128`, and 3 otherwise.
///
/// Null slots do not count, so they can hold any value. An array without valid values needs no
/// lower parts. A storage width up to `i64` needs no lower parts either, so it needs no scan.
///
/// The count depends only on the stored values, not on the precision. A value outside the
/// precision can need more lower parts than the dtype allows, and the constructor then rejects
/// the parts. The scan does not use the decimal `min_max` aggregate, because that aggregate builds
/// decimal scalars, which panic on a value outside the precision.
///
/// # Errors
///
/// Returns an error if the validity cannot be executed.
pub fn lower_part_count(decimal: &DecimalArray, ctx: &mut ExecutionCtx) -> VortexResult<usize> {
    let values_type = decimal.values_type();
    if values_type <= DecimalType::I64 {
        return Ok(0);
    }

    let mask = decimal.validity()?.execute_mask(decimal.len(), ctx)?;

    match_each_decimal_value_type!(values_type, |T| {
        Ok(valid_values_lower_part_count(&decimal.buffer::<T>(), &mask))
    })
}

/// The number of lower parts that the valid values in `values` need, as [`lower_part_count`]
/// describes.
fn valid_values_lower_part_count<T: NativeDecimalType>(values: &[T], validity: &Mask) -> usize {
    let Some((min, max)) = valid_range(values, validity) else {
        return 0;
    };

    if range_fits::<i64, _>(min, max) {
        0
    } else if range_fits::<i128, _>(min, max) {
        MAX_I128_LOWER_PARTS
    } else {
        MAX_I256_LOWER_PARTS
    }
}

/// Whether `W` holds every value from `min` to `max`.
fn range_fits<W: BigCast, T: NativeDecimalType>(min: T, max: T) -> bool {
    <W as BigCast>::from(min).is_some() && <W as BigCast>::from(max).is_some()
}

/// Split a canonical decimal array into a signed most significant part (MSP) and
/// `lower_part_count` unsigned 64-bit lower parts. The MSP is at most 64 bits.
///
/// These shapes are supported:
///
/// - A storage width up to `i64` with no lower parts. The MSP reuses the values buffer.
/// - `i128` or `i256` values with no lower parts. The values are copied into an `i64` MSP.
/// - `i128` values with one lower part, or `i256` values with three lower parts. The values split
///   on 64-bit boundaries into an `i64` MSP and the lower parts.
/// - `i256` values with one lower part. The values are copied into `i128`, and then split as
///   `i128` values.
///
/// [`lower_part_count`] gives the smallest count that holds the valid values.
///
/// The MSP retains the decimal's validity while lower parts are non-nullable. Lower parts
/// are constructed with zeroes at null positions instead of invalid bytes.
/// Empty and all-null arrays use constant parts, preserving the part types and MSP's nullability.
///
/// # Errors
///
/// Returns an error if the array's validity cannot be derived or executed, if the storage width
/// and `lower_part_count` are not one of the shapes above, or if a valid value does not fit
/// `lower_part_count` lower parts.
pub fn split_decimal(
    decimal: &DecimalArray,
    lower_part_count: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<DecimalParts> {
    let shape = SplitShape::classify(decimal.values_type(), lower_part_count)?;

    let validity = decimal.validity()?;
    let len = decimal.len();
    let mask = validity.execute_mask(len, ctx)?;

    if mask.all_false() || decimal.is_empty() {
        return Ok(split_no_valid_row(len, shape, &validity));
    }

    Ok(match shape {
        SplitShape::ReuseI8 => DecimalParts::from_narrow(decimal.buffer::<i8>(), validity),
        SplitShape::ReuseI16 => DecimalParts::from_narrow(decimal.buffer::<i16>(), validity),
        SplitShape::ReuseI32 => DecimalParts::from_narrow(decimal.buffer::<i32>(), validity),
        SplitShape::ReuseI64 => DecimalParts::from_narrow(decimal.buffer::<i64>(), validity),
        SplitShape::IntoI64 => {
            DecimalParts::from_narrow(converted_buffer::<i64>(decimal, &mask)?, validity)
        }
        SplitShape::I128OneLowerPart => {
            let (msp, lower) = split_wide(&decimal.buffer::<i128>(), &mask, i128_to_parts);
            DecimalParts::from_wide(msp, lower, validity)
        }
        SplitShape::I256OneLowerPart => {
            let values = converted_buffer::<i128>(decimal, &mask)?;
            let (msp, lower) = split_wide(&values, &mask, i128_to_parts);
            DecimalParts::from_wide(msp, lower, validity)
        }
        SplitShape::I256ThreeLowerParts => {
            let (msp, lower) = split_wide(&decimal.buffer::<i256>(), &mask, i256_to_parts);
            DecimalParts::from_wide(msp, lower, validity)
        }
    })
}

/// One of the shapes that [`split_decimal`] lists, chosen from a storage width and a lower-part
/// count.
#[derive(Clone, Copy)]
enum SplitShape {
    ReuseI8,
    ReuseI16,
    ReuseI32,
    ReuseI64,
    /// `i128` or `i256` values.
    IntoI64,
    I128OneLowerPart,
    /// Copied into `i128` first.
    I256OneLowerPart,
    I256ThreeLowerParts,
}

impl SplitShape {
    /// The shape of splitting `values_type` values into `lower_part_count` lower parts, or an
    /// error if [`split_decimal`] does not support it.
    fn classify(values_type: DecimalType, lower_part_count: usize) -> VortexResult<Self> {
        Ok(match (values_type, lower_part_count) {
            (DecimalType::I8, 0) => Self::ReuseI8,
            (DecimalType::I16, 0) => Self::ReuseI16,
            (DecimalType::I32, 0) => Self::ReuseI32,
            (DecimalType::I64, 0) => Self::ReuseI64,
            (DecimalType::I128 | DecimalType::I256, 0) => Self::IntoI64,
            (DecimalType::I128, MAX_I128_LOWER_PARTS) => Self::I128OneLowerPart,
            (DecimalType::I256, MAX_I128_LOWER_PARTS) => Self::I256OneLowerPart,
            (DecimalType::I256, MAX_I256_LOWER_PARTS) => Self::I256ThreeLowerParts,
            _ => {
                let supported = match values_type {
                    DecimalType::I128 => "0 or 1",
                    DecimalType::I256 => "0, 1 or 3",
                    _ => "0",
                };
                vortex_bail!(
                    "{values_type} values split into {supported} lower parts, \
                     got {lower_part_count}"
                )
            }
        })
    }

    /// The type of the MSP.
    fn msp_ptype(self) -> PType {
        match self {
            Self::ReuseI8 => PType::I8,
            Self::ReuseI16 => PType::I16,
            Self::ReuseI32 => PType::I32,
            Self::ReuseI64
            | Self::IntoI64
            | Self::I128OneLowerPart
            | Self::I256OneLowerPart
            | Self::I256ThreeLowerParts => PType::I64,
        }
    }

    /// The number of `u64` lower parts.
    fn lower_part_count(self) -> usize {
        match self {
            Self::ReuseI8 | Self::ReuseI16 | Self::ReuseI32 | Self::ReuseI64 | Self::IntoI64 => 0,
            Self::I128OneLowerPart | Self::I256OneLowerPart => MAX_I128_LOWER_PARTS,
            Self::I256ThreeLowerParts => MAX_I256_LOWER_PARTS,
        }
    }
}

/// Splits decimals with no valid rows (all null or empty) into constant decimal parts with the
/// corresponding nullability.
fn split_no_valid_row(len: usize, shape: SplitShape, validity: &Validity) -> DecimalParts {
    // Empty masks are also all-false. The default scalar is null for nullable inputs
    // and zero for non-nullable empty inputs, preserving the MSP's nullability.
    let msp = Scalar::default_value(&DType::Primitive(shape.msp_ptype(), validity.nullability()));
    DecimalParts {
        msp: ConstantArray::new(msp, len).into_array(),
        lower_parts: vec![ConstantArray::new(0u64, len).into_array(); shape.lower_part_count()],
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
mod tests;
