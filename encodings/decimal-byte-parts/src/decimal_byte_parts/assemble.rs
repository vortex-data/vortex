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
        vortex_ensure!(
            part.len() == len,
            "lower part {idx} has len {}, expected {len}",
            part.len()
        );
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
            _ => vortex_bail!("expected between one and {MAX_LOWER_PARTS} lower parts"),
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
mod tests {
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::Constant;
    use vortex_array::arrays::DecimalArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::DecimalDType;
    use vortex_array::dtype::DecimalType;
    use vortex_array::dtype::NativeDecimalType;
    use vortex_array::dtype::PType;
    use vortex_array::dtype::i256;
    use vortex_array::match_each_decimal_value_type;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use super::assemble_decimal;
    use crate::decimal_byte_parts::split_decimal;

    #[rstest]
    #[case::empty_non_nullable(0, Validity::NonNullable)]
    #[case::empty_nullable(0, Validity::AllValid)]
    #[case::empty_all_null(0, Validity::AllInvalid)]
    #[case::all_null(3, Validity::AllInvalid)]
    #[case::all_null_array(3, Validity::Array(BoolArray::from_iter([false; 3]).into_array()))]
    fn test_split_without_valid_rows(
        #[case] len: usize,
        #[case] validity: Validity,
        #[values(
            DecimalType::I8,
            DecimalType::I16,
            DecimalType::I32,
            DecimalType::I64,
            DecimalType::I128,
            DecimalType::I256
        )]
        values_type: DecimalType,
    ) -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let decimal = match_each_decimal_value_type!(values_type, |T| {
            DecimalArray::new(
                Buffer::<T>::zeroed(len),
                DecimalDType::new(T::MAX_PRECISION, 0),
                validity,
            )
        });
        let parts = split_decimal(&decimal, &mut ctx)?;
        assert!(parts.msp.is::<Constant>());
        assert!(parts.lower_parts.iter().all(|part| part.is::<Constant>()));
        assert_eq!(parts.msp.len(), len);
        assert_eq!(
            parts.msp.dtype().nullability(),
            decimal.dtype().nullability()
        );
        let round_tripped = round_trip(decimal.clone())?;
        assert_eq!(round_tripped.values_type(), values_type);
        assert_arrays_eq!(decimal, round_tripped, &mut ctx);
        Ok(())
    }

    #[rstest]
    #[case::non_nullable(Validity::NonNullable)]
    #[case::all_valid(Validity::AllValid)]
    #[case::all_null(Validity::AllInvalid)]
    #[case::mixed(Validity::from_iter((0..263).map(|i| i % 3 != 1)))]
    #[case::sparse(Validity::from_iter((0..263).map(|i| i % 16 == 0)))]
    #[case::null_prefix_and_suffix(Validity::from_iter((0..263).map(|i| (67..196).contains(&i))))]
    fn test_split_zeroes_null_words(
        #[case] validity: Validity,
        #[values(false, true)] wide_256: bool,
        #[values(0, 1, 63, 64, 65, 257)] len: usize,
    ) -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let decimal = if wide_256 {
            DecimalArray::new(
                buffer![i256::from_i128(-1); 263],
                DecimalDType::new(76, 2),
                validity,
            )
        } else {
            DecimalArray::new(buffer![-1i128; 263], DecimalDType::new(38, 2), validity)
        };
        let decimal = decimal
            .slice(3..len + 3)?
            .execute::<DecimalArray>(&mut ctx)?;
        let mask = decimal.validity()?.execute_mask(len, &mut ctx)?;
        let expected = PrimitiveArray::new(
            mask.iter()
                .map(|valid| if valid { u64::MAX } else { 0 })
                .collect::<Buffer<_>>(),
            Validity::NonNullable,
        );
        let parts = split_decimal(&decimal, &mut ctx)?;
        assert_eq!(parts.lower_parts.len(), if wide_256 { 3 } else { 1 });
        assert_eq!(
            parts.msp.dtype(),
            &DType::Primitive(PType::I64, decimal.dtype().nullability())
        );
        for lower in parts.lower_parts {
            assert_arrays_eq!(expected.clone(), lower, &mut ctx);
        }
        assert_arrays_eq!(decimal.clone(), round_trip(decimal)?, &mut ctx);
        Ok(())
    }

    fn round_trip(decimal: DecimalArray) -> VortexResult<DecimalArray> {
        let mut ctx = array_session().create_execution_ctx();
        let parts = split_decimal(&decimal, &mut ctx)?;
        assemble_decimal(
            &parts.msp,
            &parts.lower_parts,
            decimal.decimal_dtype(),
            &mut ctx,
        )?
        .execute::<DecimalArray>(&mut ctx)
    }

    #[rstest]
    #[case::zero(0)]
    #[case::one(1)]
    #[case::minus_one(-1)]
    #[case::limb_boundary(1i128 << 64)]
    #[case::just_below_limb_boundary((1i128 << 64) - 1)]
    #[case::negative_limb_boundary(-(1i128 << 64))]
    #[case::max(i128::MAX)]
    #[case::min(i128::MIN)]
    fn test_split_assemble_i128(#[case] value: i128) -> VortexResult<()> {
        let decimal = DecimalArray::new(
            Buffer::from(vec![value]),
            DecimalDType::new(38, 2),
            Validity::NonNullable,
        );
        let round_tripped = round_trip(decimal)?;
        assert_eq!(round_tripped.buffer::<i128>().as_slice(), &[value]);
        Ok(())
    }

    #[rstest]
    #[case::zero(i256::ZERO)]
    #[case::one(i256::ONE)]
    #[case::minus_one(i256::ZERO - i256::ONE)]
    #[case::max(i256::MAX)]
    #[case::min(i256::MIN)]
    #[case::word_1(i256::from_parts(1u128 << 64, 0))]
    #[case::word_2(i256::from_parts(0, 1))]
    #[case::word_3(i256::from_parts(0, 1i128 << 64))]
    #[case::mixed(i256::from_parts(u128::MAX, -3))]
    fn test_split_assemble_i256(#[case] value: i256) -> VortexResult<()> {
        let decimal = DecimalArray::new(
            Buffer::from(vec![value]),
            DecimalDType::new(76, 2),
            Validity::NonNullable,
        );
        let round_tripped = round_trip(decimal)?;
        assert_eq!(round_tripped.buffer::<i256>().as_slice(), &[value]);
        Ok(())
    }

    #[rstest]
    fn test_split_narrow_decimal_reuses_values(
        #[values(Validity::NonNullable, Validity::from_iter([true, false, true]))]
        validity: Validity,
    ) -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let decimal = DecimalArray::new(buffer![1i32, 2, 3], DecimalDType::new(2, 0), validity);
        let parts = split_decimal(&decimal, &mut ctx)?;
        assert!(parts.lower_parts.is_empty());
        assert_eq!(parts.msp.dtype().as_ptype(), PType::I32);
        let msp = parts.msp.execute::<PrimitiveArray>(&mut ctx)?;
        assert_eq!(
            msp.as_slice::<i32>().as_ptr(),
            decimal.buffer::<i32>().as_ptr()
        );
        assert_arrays_eq!(decimal.clone(), round_trip(decimal)?, &mut ctx);
        Ok(())
    }

    #[rstest]
    #[case::signed(PrimitiveArray::new(buffer![0i64; 2], Validity::NonNullable))]
    #[case::float(PrimitiveArray::new(buffer![0f32; 2], Validity::NonNullable))]
    #[case::nullable_all_valid(PrimitiveArray::new(buffer![0u64; 2], Validity::AllValid))]
    #[case::nullable_all_null(PrimitiveArray::new(buffer![0u64; 2], Validity::AllInvalid))]
    #[case::nullable_mixed(PrimitiveArray::new(buffer![0u64; 2], Validity::from_iter([true, false])))]
    fn test_assemble_rejects_invalid_lower_dtype(
        #[case] invalid_lower: PrimitiveArray,
        #[values(1, 2, 3)] lower_count: usize,
    ) {
        let msp = PrimitiveArray::new(buffer![0i64; 2], Validity::NonNullable);
        let mut lower =
            vec![PrimitiveArray::new(buffer![0u64; 2], Validity::NonNullable); lower_count];
        lower[lower_count - 1] = invalid_lower;
        let dtype = DecimalDType::new(if lower_count == 1 { 38 } else { 76 }, 0);
        let lower = lower
            .into_iter()
            .map(IntoArray::into_array)
            .collect::<Vec<_>>();
        let mut ctx = array_session().create_execution_ctx();
        assert!(assemble_decimal(&msp.into_array(), &lower, dtype, &mut ctx).is_err());
    }

    #[rstest]
    fn test_assemble_rejects_mismatched_lower_lengths(
        #[values(1, 2, 3)] lower_count: usize,
        #[values(0, 1, 3)] lower_len: usize,
    ) {
        let msp = PrimitiveArray::new(buffer![0i64; 2], Validity::NonNullable);
        let mut lower =
            vec![PrimitiveArray::new(buffer![0u64; 2], Validity::NonNullable); lower_count];
        lower[lower_count - 1] =
            PrimitiveArray::new(buffer![0u64; lower_len], Validity::NonNullable);
        let dtype = DecimalDType::new(if lower_count == 1 { 38 } else { 76 }, 0);
        let lower = lower
            .into_iter()
            .map(IntoArray::into_array)
            .collect::<Vec<_>>();
        let mut ctx = array_session().create_execution_ctx();
        assert!(assemble_decimal(&msp.into_array(), &lower, dtype, &mut ctx).is_err());
    }

    #[rstest]
    fn test_assemble_i256_part_order_and_sign_extension(
        #[values(false, true)] narrow_msp: bool,
        #[values(2, 3)] lower_count: usize,
    ) -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let msp = if narrow_msp {
            PrimitiveArray::new(buffer![3i8, -3], Validity::NonNullable)
        } else {
            PrimitiveArray::new(buffer![3i64, -3], Validity::NonNullable)
        };
        let lower = [4u64, 1, 2].map(|word| buffer![word; 2].into_array());
        let dtype = DecimalDType::new(76, 0);
        let actual = assemble_decimal(
            &msp.into_array(),
            &lower[3 - lower_count..],
            dtype,
            &mut ctx,
        )?;
        let low = (1u128 << 64) | 2;
        let expected = if lower_count == 2 {
            buffer![i256::from_parts(low, 3), i256::from_parts(low, -3)]
        } else {
            buffer![
                i256::from_parts(low, (3i128 << 64) | 4),
                i256::from_parts(low, (-3i128 << 64) | 4),
            ]
        };
        assert_arrays_eq!(
            DecimalArray::new(expected, dtype, Validity::NonNullable),
            actual,
            &mut ctx
        );
        Ok(())
    }
}
