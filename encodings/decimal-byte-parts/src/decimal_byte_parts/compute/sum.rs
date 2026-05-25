// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::CheckedAdd;
use num_traits::CheckedMul;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::fns::sum::Sum;
use vortex_array::aggregate_fn::kernels::DynAggregateKernel;
use vortex_array::arrays::Constant;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::DecimalType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::i256;
use vortex_array::match_each_signed_integer_ptype;
use vortex_array::scalar::DecimalValue;
use vortex_array::scalar::Scalar;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::DecimalByteParts;
use crate::decimal_byte_parts::DecimalBytePartsArrayExt;
use crate::decimal_byte_parts::compute::compare::as_primitive;

/// Maximum decimal precision (the i256 width).
const MAX_PRECISION: u8 = 76;

/// Native `SUM` over a [`DecimalByteParts`] column.
///
/// Sums each limb column independently — the signed msp into an `i128` accumulator and every
/// unsigned `u64` lower part into a `u128` accumulator — then combines the partial sums into an
/// `i256` total with checked arithmetic. No value is ever reconstructed into a wide canonical
/// buffer. The result type, overflow handling, and precision saturation match the canonical
/// (Arrow-style) decimal sum: return precision is `min(76, input_precision + 10)`, and the result
/// is null on overflow or when it exceeds that precision.
#[derive(Debug)]
pub(crate) struct DecimalBytePartsSumKernel;

impl DynAggregateKernel for DecimalBytePartsSumKernel {
    fn aggregate(
        &self,
        aggregate_fn: &AggregateFnRef,
        batch: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Scalar>> {
        if !aggregate_fn.is::<Sum>() {
            return Ok(None);
        }
        let Some(arr) = batch.as_opt::<DecimalByteParts>() else {
            return Ok(None);
        };
        let k = arr.num_lower_parts();
        // Lower parts must be the standard unsigned 64-bit limbs.
        if (0..k).any(|i| PType::try_from(arr.lower_part(i).dtype()).ok() != Some(PType::U64)) {
            return Ok(None);
        }

        let in_dtype = *arr
            .dtype()
            .as_decimal_opt()
            .vortex_expect("byte-parts is always a decimal");
        let out_dtype = DecimalDType::new(
            u8::min(MAX_PRECISION, in_dtype.precision() + 10),
            in_dtype.scale(),
        );
        let null = || {
            Ok(Some(Scalar::null(DType::Decimal(
                out_dtype,
                Nullability::Nullable,
            ))))
        };

        // The msp carries the array validity. A constant msp is summed in O(1) (and its validity is
        // cheap); otherwise decode it exactly once and take both the validity mask and the sum from
        // that decoded copy — never decode the msp twice.
        let msp = arr.msp();
        let len = msp.len();
        let msp_const = msp
            .as_opt::<Constant>()
            .and_then(|c| c.scalar().as_primitive().typed_value::<i64>());
        let (mask, msp_sum) = match msp_const {
            Some(value) => {
                let mask = msp.validity()?.execute_mask(len, ctx)?;
                let sum = i128::from(value) * mask.true_count() as i128;
                (mask, sum)
            }
            None => {
                let prim = as_primitive(msp, ctx)?;
                let mask = prim.validity()?.execute_mask(len, ctx)?;
                let sum = sum_signed(&prim, &mask);
                (mask, sum)
            }
        };
        let valid_count = mask.true_count();

        // Combine the limb sums most-significant first: total = msp << 64k + Σ lower[j] << 64(k-1-j).
        let mut total = match i256::from_i128(msp_sum).checked_mul(&base(64 * k)) {
            Some(acc) => acc,
            None => return null(),
        };
        for idx in 0..k {
            let lower = limb_sum_unsigned(arr.lower_part(idx), &mask, valid_count, ctx)?;
            total = match i256::from_parts(lower, 0)
                .checked_mul(&base(64 * (k - 1 - idx)))
                .and_then(|shifted| total.checked_add(&shifted))
            {
                Some(acc) => acc,
                None => return null(),
            };
        }

        let value = DecimalValue::I256(total);
        if !value.fits_in_precision(out_dtype) {
            return null();
        }
        // Narrow to the return type's native width so the result matches the canonical sum exactly.
        let value = match DecimalType::smallest_decimal_value_type(&out_dtype) {
            DecimalType::I8 => DecimalValue::I8(narrow(value)),
            DecimalType::I16 => DecimalValue::I16(narrow(value)),
            DecimalType::I32 => DecimalValue::I32(narrow(value)),
            DecimalType::I64 => DecimalValue::I64(narrow(value)),
            DecimalType::I128 => DecimalValue::I128(narrow(value)),
            DecimalType::I256 => value,
        };
        Ok(Some(Scalar::decimal(
            value,
            out_dtype,
            Nullability::Nullable,
        )))
    }
}

/// `2^shift` as an `i256` (`shift <= 192`, so this never overflows).
fn base(shift: usize) -> i256 {
    i256::ONE << shift
}

/// Narrows an in-range [`DecimalValue`] to a concrete native width (the magnitude is already
/// bounded by `fits_in_precision`, so the cast cannot fail).
fn narrow<T: vortex_array::dtype::NativeDecimalType>(value: DecimalValue) -> T {
    value
        .cast::<T>()
        .vortex_expect("value validated to fit the return precision")
}

/// Sum of an unsigned `u64` limb over valid rows, into `u128`. A constant limb (the common shape
/// after compression — e.g. an all-zero limb, or a narrowed column) is summed in O(1) as
/// `value * valid_count` without decoding it; otherwise the limb is decoded and summed.
fn limb_sum_unsigned(
    part: &ArrayRef,
    mask: &Mask,
    valid_count: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<u128> {
    if let Some(constant) = part.as_opt::<Constant>()
        && let Some(value) = constant.scalar().as_primitive().typed_value::<u64>()
    {
        return Ok(u128::from(value) * valid_count as u128);
    }
    Ok(sum_unsigned(&as_primitive(part, ctx)?, mask))
}

/// Sum of the signed msp column over valid rows, into `i128` (cannot overflow for realistic
/// lengths: `len * i64::MAX < i128::MAX`).
fn sum_signed(prim: &PrimitiveArray, mask: &Mask) -> i128 {
    match_each_signed_integer_ptype!(prim.ptype(), |P| {
        let values = prim.as_slice::<P>();
        match mask {
            Mask::AllTrue(_) => sum_all_i128(values),
            Mask::AllFalse(_) => 0,
            Mask::Values(bits) => values
                .iter()
                .zip(bits.bit_buffer())
                .filter(|(_, valid)| *valid)
                .map(|(&val, _)| i128::from(val))
                .sum(),
        }
    })
}

/// Sum of an unsigned `u64` lower column over valid rows, into `u128`.
fn sum_unsigned(prim: &PrimitiveArray, mask: &Mask) -> u128 {
    let values = prim.as_slice::<u64>();
    match mask {
        Mask::AllTrue(_) => sum_all_u128(values),
        Mask::AllFalse(_) => 0,
        Mask::Values(bits) => values
            .iter()
            .zip(bits.bit_buffer())
            .filter(|(_, valid)| *valid)
            .map(|(&val, _)| u128::from(val))
            .sum(),
    }
}

// Wide (`i128`/`u128`) accumulation does not vectorize — there is no SIMD 128-bit add. Instead we
// sum each column as two 64-bit *limbs*: the low and high 32 bits of every element accumulated
// separately into 64-bit lanes, which LLVM lowers to 8-wide `vpaddq` reductions. A 64-bit limb
// accumulator would itself overflow after ~2^32 elements (each half-term is < 2^32), so we flush
// the limbs into the wide total every `SUM_BLOCK` elements — small enough that a block's limb sum
// stays below 2^63. Any realistic batch is a single block (still fully vectorized); larger inputs
// stay exact.

/// Block size bounding each 64-bit limb accumulator: `SUM_BLOCK * (2^32 - 1) < 2^63`.
const SUM_BLOCK: usize = 1 << 31;

/// SIMD-friendly widening sum of a signed-integer column into `i128`.
fn sum_all_i128<P: Into<i64> + Copy>(values: &[P]) -> i128 {
    sum_i128_blocked(values, SUM_BLOCK)
}

fn sum_i128_blocked<P: Into<i64> + Copy>(values: &[P], block: usize) -> i128 {
    let mut total: i128 = 0;
    for block in values.chunks(block) {
        let mut lo: i64 = 0;
        let mut hi: i64 = 0;
        for &value in block {
            let value: i64 = value.into();
            lo += value & 0xffff_ffff;
            hi += value >> 32;
        }
        total += (i128::from(hi) << 32) + i128::from(lo);
    }
    total
}

/// SIMD-friendly widening sum of a `u64` column into `u128`.
fn sum_all_u128(values: &[u64]) -> u128 {
    sum_u128_blocked(values, SUM_BLOCK)
}

fn sum_u128_blocked(values: &[u64], block: usize) -> u128 {
    let mut total: u128 = 0;
    for block in values.chunks(block) {
        let mut lo: u64 = 0;
        let mut hi: u64 = 0;
        for &value in block {
            lo += value & 0xffff_ffff;
            hi += value >> 32;
        }
        total += (u128::from(hi) << 32) + u128::from(lo);
    }
    total
}

#[cfg(test)]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap
)]
mod tests {
    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::aggregate_fn::fns::sum::sum;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::DecimalArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::dtype::DecimalDType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::i256;
    use vortex_array::scalar::Scalar;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::sum_i128_blocked;
    use super::sum_u128_blocked;
    use crate::DecimalByteParts;

    /// A constant lower limb (the common post-compression shape) must sum identically to the
    /// decoded form — exercising the O(1) `value * valid_count` fast path.
    #[test]
    fn constant_limb_matches_canonical() -> VortexResult<()> {
        let dtype = DecimalDType::new(38, 2);
        let msp_vals = [1i64, 2, -3, 0, 5];
        let lower_const = 7u64;
        let msp = PrimitiveArray::new(
            msp_vals.iter().copied().collect::<Buffer<i64>>(),
            Validity::NonNullable,
        )
        .into_array();
        let lower = ConstantArray::new(
            Scalar::primitive(lower_const, Nullability::NonNullable),
            msp_vals.len(),
        )
        .into_array();
        let bp = DecimalByteParts::try_new_parts(msp, vec![lower], dtype)?.into_array();
        let values: Vec<i128> = msp_vals
            .iter()
            .map(|&m| (i128::from(m) << 64) + i128::from(lower_const))
            .collect();
        let canon = DecimalArray::from_iter(values, dtype).into_array();
        check(bp, canon)
    }

    /// The block-flushed limb sums must equal a naive wide sum even when many blocks are crossed
    /// with limb values near `u32::MAX` (the case that would overflow a single 64-bit accumulator).
    #[test]
    fn blocked_limb_sum_is_exact_across_blocks() {
        let big: Vec<u64> = (0..50).map(|i| u64::MAX - i).collect();
        let expected: u128 = big.iter().map(|&v| u128::from(v)).sum();
        // Tiny block size forces repeated flushing.
        assert_eq!(sum_u128_blocked(&big, 4), expected);

        let signed: Vec<i64> = (0..50)
            .map(|i| {
                if i % 2 == 0 {
                    i64::MAX - i
                } else {
                    i64::MIN + i
                }
            })
            .collect();
        let expected: i128 = signed.iter().map(|&v| i128::from(v)).sum();
        assert_eq!(sum_i128_blocked(&signed, 4), expected);
    }

    fn validity(present: impl Iterator<Item = bool>) -> Validity {
        Validity::Array(BoolArray::from_iter(present).into_array())
    }

    fn bp_i128(values: &[Option<i128>], dtype: DecimalDType) -> ArrayRef {
        let vld = validity(values.iter().map(Option::is_some));
        let msp = PrimitiveArray::new(
            values
                .iter()
                .map(|x| (x.unwrap_or(0) >> 64) as i64)
                .collect::<Buffer<i64>>(),
            vld,
        );
        let low = PrimitiveArray::new(
            values
                .iter()
                .map(|x| x.unwrap_or(0) as u64)
                .collect::<Buffer<u64>>(),
            Validity::NonNullable,
        );
        DecimalByteParts::try_new_parts(msp.into_array(), vec![low.into_array()], dtype)
            .unwrap()
            .into_array()
    }

    fn bp_i256(values: &[Option<i256>], dtype: DecimalDType) -> ArrayRef {
        let vld = validity(values.iter().map(Option::is_some));
        let zero = i256::ZERO;
        let msp = PrimitiveArray::new(
            values
                .iter()
                .map(|x| (x.unwrap_or(zero).to_parts().1 >> 64) as i64)
                .collect::<Buffer<i64>>(),
            vld,
        );
        let mk = |f: fn(i256) -> u64| {
            PrimitiveArray::new(
                values
                    .iter()
                    .map(|x| f(x.unwrap_or(zero)))
                    .collect::<Buffer<u64>>(),
                Validity::NonNullable,
            )
            .into_array()
        };
        let lowers = vec![
            mk(|x| x.to_parts().1 as u64),
            mk(|x| (x.to_parts().0 >> 64) as u64),
            mk(|x| x.to_parts().0 as u64),
        ];
        DecimalByteParts::try_new_parts(msp.into_array(), lowers, dtype)
            .unwrap()
            .into_array()
    }

    fn session() -> VortexSession {
        use vortex_array::session::ArraySession;
        let session = VortexSession::empty().with::<ArraySession>();
        crate::initialize(&session);
        session
    }

    /// Native byte-parts sum must equal the canonical (Arrow-style) sum.
    fn check(byteparts: ArrayRef, canonical: ArrayRef) -> VortexResult<()> {
        let session = session();
        let native = sum(&byteparts, &mut session.create_execution_ctx())?;
        let canon = sum(&canonical, &mut session.create_execution_ctx())?;
        assert_eq!(native, canon, "byte-parts sum != canonical sum");
        Ok(())
    }

    #[rstest]
    #[case::all_valid(&[Some(10i128.pow(30)), Some(-5), Some(7), Some(10i128.pow(30))])]
    #[case::with_nulls(&[Some(10i128.pow(30)), None, Some(7), None])]
    #[case::all_null(&[None, None, None])]
    fn i128_sum_matches_canonical(#[case] values: &[Option<i128>]) -> VortexResult<()> {
        let dtype = DecimalDType::new(38, 2);
        let canon = DecimalArray::from_option_iter(values.iter().copied(), dtype).into_array();
        check(bp_i128(values, dtype), canon)
    }

    #[rstest]
    #[case::all_valid(&[Some(i256::from_i128(10i128.pow(35))), Some(i256::from_i128(-9)), Some(i256::from_parts(u128::MAX, 3))])]
    #[case::with_nulls(&[Some(i256::from_parts(u128::MAX, 1)), None, Some(i256::from_i128(42))])]
    fn i256_sum_matches_canonical(#[case] values: &[Option<i256>]) -> VortexResult<()> {
        let dtype = DecimalDType::new(60, 2);
        let canon = DecimalArray::from_option_iter(values.iter().copied(), dtype).into_array();
        check(bp_i256(values, dtype), canon)
    }

    #[test]
    fn i256_sum_overflow_is_null() -> VortexResult<()> {
        // Three i256::MAX values at precision 76: the sum exceeds precision 76 -> null on both paths.
        let dtype = DecimalDType::new(76, 0);
        let values = [Some(i256::MAX), Some(i256::MAX), Some(i256::MAX)];
        let canon = DecimalArray::from_option_iter(values.iter().copied(), dtype).into_array();
        check(bp_i256(&values, dtype), canon)
    }

    #[rstest]
    #[case::i32(DecimalDType::new(9, 2))]
    #[case::i64(DecimalDType::new(18, 2))]
    fn single_part_sum_matches_canonical(#[case] dtype: DecimalDType) -> VortexResult<()> {
        // Single-part byte-parts: msp is the value, zero lower parts.
        let values = [Some(1234i64), None, Some(-56), Some(78)];
        let vld = validity(values.iter().map(Option::is_some));
        let msp = PrimitiveArray::new(
            values
                .iter()
                .map(|x| x.unwrap_or(0))
                .collect::<Buffer<i64>>(),
            vld,
        );
        let byteparts = DecimalByteParts::try_new(msp.into_array(), dtype)?.into_array();
        let canon = DecimalArray::from_option_iter(values.iter().map(|x| x.map(i128::from)), dtype)
            .into_array();
        check(byteparts, canon)
    }
}
