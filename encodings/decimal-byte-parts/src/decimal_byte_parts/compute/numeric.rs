// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::ops::overflowing::OverflowingAdd;
use num_traits::ops::overflowing::OverflowingSub;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::ScalarFn;
use vortex_array::arrays::scalar_fn::ExactScalarFn;
use vortex_array::arrays::scalar_fn::ScalarFnArrayExt;
use vortex_array::arrays::scalar_fn::ScalarFnArrayView;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::DecimalType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::PType;
use vortex_array::kernel::ExecuteParentKernel;
use vortex_array::scalar_fn::fns::binary::Binary;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_array::validity::Validity;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::DecimalByteParts;
use crate::decimal_byte_parts::DecimalBytePartsArrayExt;
use crate::decimal_byte_parts::compute::compare::as_primitive;

#[derive(Clone, Copy)]
enum Op {
    Add,
    Sub,
}

/// Bridges decimal byte-parts addition/subtraction into the parent-kernel scheduler so a
/// `byteparts +/- byteparts` expression is evaluated limb-wise, in-place, and stays a
/// [`DecimalByteParts`] array instead of canonicalizing to a wide [`DecimalArray`].
#[derive(Debug)]
pub(crate) struct NumericExecuteAdaptor;

impl ExecuteParentKernel<DecimalByteParts> for NumericExecuteAdaptor {
    type Parent = ExactScalarFn<Binary>;

    fn execute_parent(
        &self,
        array: ArrayView<'_, DecimalByteParts>,
        parent: ScalarFnArrayView<'_, Binary>,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let op = match *parent.options {
            Operator::Add => Op::Add,
            Operator::Sub => Op::Sub,
            _ => return Ok(None),
        };

        let Some(scalar_fn_array) = parent.as_opt::<ScalarFn>() else {
            return Ok(None);
        };
        let other = match child_idx {
            0 => scalar_fn_array.get_child(1),
            1 => scalar_fn_array.get_child(0),
            _ => return Ok(None),
        };
        let Some(other) = other.as_opt::<DecimalByteParts>() else {
            return Ok(None);
        };

        if !array.dtype().eq_ignore_nullability(other.dtype()) {
            return Ok(None);
        }
        let in_dtype = *array
            .dtype()
            .as_decimal_opt()
            .vortex_expect("byte-parts is always a decimal");

        // Only push down when both operands are in the *natural* (un-narrowed) layout for the dtype,
        // and only for widths that keep the same physical storage under add/sub precision promotion
        // (i32/i64 single part, i128/i256 split). A column narrowed below its natural width, or an
        // i8/i16 decimal whose promoted precision crosses into a wider type, falls back to the
        // canonical path (which widens correctly) so the cheap fixed-width overflow check stays sound.
        let natural = DecimalType::smallest_decimal_value_type(&in_dtype);
        let (natural_k, msp_is_i64) = match natural {
            DecimalType::I32 => (0, false),
            DecimalType::I64 => (0, true),
            DecimalType::I128 => (1, true),
            DecimalType::I256 => (3, true),
            DecimalType::I8 | DecimalType::I16 => return Ok(None),
        };
        if !layout_matches(&array, natural_k, msp_is_i64)
            || !layout_matches(&other, natural_k, msp_is_i64)
        {
            return Ok(None);
        }

        // Compute `lhs OP rhs` matching the original expression. Add is commutative; Sub is not, so
        // when our array is the right-hand child we swap the operands.
        let (lhs, rhs) = match (op, child_idx) {
            (Op::Sub, 1) => (&other, &array),
            _ => (&array, &other),
        };
        let out_dtype = in_dtype.promote_add_sub();
        let result = match natural {
            DecimalType::I32 => add_sub_single::<i32>(lhs.msp(), rhs.msp(), op, out_dtype, ctx)?,
            DecimalType::I64 => add_sub_single::<i64>(lhs.msp(), rhs.msp(), op, out_dtype, ctx)?,
            _ => add_sub_multi(lhs, rhs, op, out_dtype, ctx)?,
        };
        Ok(Some(result))
    }
}

fn is_i32(part: &ArrayRef) -> bool {
    PType::try_from(part.dtype()).ok() == Some(PType::I32)
}

fn is_i64(part: &ArrayRef) -> bool {
    PType::try_from(part.dtype()).ok() == Some(PType::I64)
}

fn is_u64(part: &ArrayRef) -> bool {
    PType::try_from(part.dtype()).ok() == Some(PType::U64)
}

/// True if `v` is stored in the natural byte-parts layout for its dtype: `natural_k` unsigned u64
/// lower parts and a signed msp of the expected width.
fn layout_matches(v: &ArrayView<'_, DecimalByteParts>, natural_k: usize, msp_is_i64: bool) -> bool {
    v.num_lower_parts() == natural_k
        && (if msp_is_i64 {
            is_i64(v.msp())
        } else {
            is_i32(v.msp())
        })
        && (0..natural_k).all(|i| is_u64(v.lower_part(i)))
}

/// Single-part (`i32`/`i64`) decimal add/subtract: the msp *is* the value, so this is a plain
/// fixed-width wrapping op with overflow detection (vectorizable), staying a single-part byte-parts.
fn add_sub_single<T>(
    lhs_msp: &ArrayRef,
    rhs_msp: &ArrayRef,
    op: Op,
    out_dtype: DecimalDType,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef>
where
    T: NativePType + OverflowingAdd + OverflowingSub,
{
    let lhs = as_primitive(lhs_msp, ctx)?;
    let rhs = as_primitive(rhs_msp, ctx)?;
    let av = lhs.as_slice::<T>();
    let bv = rhs.as_slice::<T>();
    let n = av.len();
    let is_add = matches!(op, Op::Add);

    let mut out = BufferMut::<T>::zeroed(n);
    let dst = out.as_mut_slice();
    let mut overflow = 0u8;
    for i in 0..n {
        let (res, of) = if is_add {
            av[i].overflowing_add(&bv[i])
        } else {
            av[i].overflowing_sub(&bv[i])
        };
        dst[i] = res;
        overflow |= u8::from(of);
    }
    if overflow != 0 {
        vortex_bail!(
            "decimal {} overflowed precision {}",
            if is_add { "add" } else { "subtract" },
            out_dtype.precision()
        );
    }

    let validity = lhs.validity()?.and(rhs.validity()?)?;
    let msp = PrimitiveArray::new(out.freeze(), validity).into_array();
    Ok(DecimalByteParts::try_new(msp, out_dtype)?.into_array())
}

/// Limb-wise multi-precision add/subtract of two same-layout byte-parts columns.
///
/// Each limb column is processed least-significant-first so the carry (borrow) chain runs across
/// limbs while every per-limb pass stays a straight loop over rows that vectorizes across lanes.
fn add_sub_multi(
    lhs: &ArrayView<'_, DecimalByteParts>,
    rhs: &ArrayView<'_, DecimalByteParts>,
    op: Op,
    out_dtype: DecimalDType,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let a_msp = as_primitive(lhs.msp(), ctx)?;
    let b_msp = as_primitive(rhs.msp(), ctx)?;
    let n = a_msp.len();
    let k = lhs.num_lower_parts();

    let a_lo = (0..k)
        .map(|i| as_primitive(lhs.lower_part(i), ctx))
        .collect::<VortexResult<Vec<_>>>()?;
    let b_lo = (0..k)
        .map(|i| as_primitive(rhs.lower_part(i), ctx))
        .collect::<VortexResult<Vec<_>>>()?;

    let a0 = a_msp.as_slice::<i64>();
    let b0 = b_msp.as_slice::<i64>();
    let a_lo: Vec<&[u64]> = a_lo.iter().map(|p| p.as_slice::<u64>()).collect();
    let b_lo: Vec<&[u64]> = b_lo.iter().map(|p| p.as_slice::<u64>()).collect();

    let is_add = matches!(op, Op::Add);
    let mut carry = vec![0u64; n];
    let mut out_lo: Vec<BufferMut<u64>> = (0..k).map(|_| BufferMut::<u64>::zeroed(n)).collect();

    for limb in (0..k).rev() {
        let av = a_lo[limb];
        let bv = b_lo[limb];
        let out = out_lo[limb].as_mut_slice();
        if is_add {
            for i in 0..n {
                let (s1, c1) = av[i].overflowing_add(bv[i]);
                let (s2, c2) = s1.overflowing_add(carry[i]);
                out[i] = s2;
                carry[i] = u64::from(c1 | c2);
            }
        } else {
            for i in 0..n {
                let (d1, b1) = av[i].overflowing_sub(bv[i]);
                let (d2, b2) = d1.overflowing_sub(carry[i]);
                out[i] = d2;
                carry[i] = u64::from(b1 | b2);
            }
        }
    }

    // The msp is the signed high limb. Signed overflow here means the whole wide value overflowed
    // its i128/i256 storage; we OR it into a single accumulator (a vectorizable reduction) and, to
    // match Arrow's checked arithmetic, error if any row overflowed. This is an approximate match
    // for Arrow's precision check: it catches physical-width overflow exactly but not the narrow
    // band between `10^precision` and the storage max for precisions capped at the width.
    let mut out_msp = BufferMut::<i64>::zeroed(n);
    let msp = out_msp.as_mut_slice();
    let mut overflow = 0u8;
    if is_add {
        for i in 0..n {
            let (sum, o1) = a0[i].overflowing_add(b0[i]);
            let (res, o2) = sum.overflowing_add(carry[i] as i64);
            msp[i] = res;
            overflow |= u8::from(o1 | o2);
        }
    } else {
        for i in 0..n {
            let (diff, o1) = a0[i].overflowing_sub(b0[i]);
            let (res, o2) = diff.overflowing_sub(carry[i] as i64);
            msp[i] = res;
            overflow |= u8::from(o1 | o2);
        }
    }
    if overflow != 0 {
        vortex_bail!(
            "decimal {} overflowed precision {}",
            if is_add { "add" } else { "subtract" },
            out_dtype.precision()
        );
    }

    let validity = a_msp.validity()?.and(b_msp.validity()?)?;
    let msp_array = PrimitiveArray::new(out_msp.freeze(), validity).into_array();
    let lowers: Vec<ArrayRef> = out_lo
        .into_iter()
        .map(|buf| PrimitiveArray::new(buf.freeze(), Validity::NonNullable).into_array())
        .collect();

    Ok(DecimalByteParts::try_new_parts(msp_array, lowers, out_dtype)?.into_array())
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
    use vortex_array::LEGACY_SESSION;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::DecimalArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::builtins::ArrayBuiltins;
    use vortex_array::dtype::DecimalDType;
    use vortex_array::dtype::i256;
    use vortex_array::scalar_fn::fns::operators::Operator;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_error::VortexResult;

    use crate::DecimalByteParts;

    fn validity_of(present: impl Iterator<Item = bool>) -> Validity {
        Validity::Array(BoolArray::from_iter(present).into_array())
    }

    fn i128_pair(values: &[Option<i128>], dtype: DecimalDType) -> (ArrayRef, ArrayRef) {
        let validity = validity_of(values.iter().map(Option::is_some));
        let msp = PrimitiveArray::new(
            values
                .iter()
                .map(|v| (v.unwrap_or(0) >> 64) as i64)
                .collect::<Buffer<i64>>(),
            validity,
        );
        let low = PrimitiveArray::new(
            values
                .iter()
                .map(|v| v.unwrap_or(0) as u64)
                .collect::<Buffer<u64>>(),
            Validity::NonNullable,
        );
        let bp = DecimalByteParts::try_new_parts(msp.into_array(), vec![low.into_array()], dtype)
            .unwrap()
            .into_array();
        let canon = DecimalArray::from_option_iter(values.iter().copied(), dtype).into_array();
        (bp, canon)
    }

    fn i256_pair(values: &[Option<i256>], dtype: DecimalDType) -> (ArrayRef, ArrayRef) {
        let validity = validity_of(values.iter().map(Option::is_some));
        let z = i256::ZERO;
        let msp = PrimitiveArray::new(
            values
                .iter()
                .map(|v| (v.unwrap_or(z).to_parts().1 >> 64) as i64)
                .collect::<Buffer<i64>>(),
            validity,
        );
        let mk = |f: fn(i256) -> u64| {
            PrimitiveArray::new(
                values
                    .iter()
                    .map(|v| f(v.unwrap_or(z)))
                    .collect::<Buffer<u64>>(),
                Validity::NonNullable,
            )
            .into_array()
        };
        let lowers = vec![
            mk(|v| v.to_parts().1 as u64),
            mk(|v| (v.to_parts().0 >> 64) as u64),
            mk(|v| v.to_parts().0 as u64),
        ];
        let bp = DecimalByteParts::try_new_parts(msp.into_array(), lowers, dtype)
            .unwrap()
            .into_array();
        let canon = DecimalArray::from_option_iter(values.iter().copied(), dtype).into_array();
        (bp, canon)
    }

    /// Asserts that `bp_a OP bp_b` (1) stays a [`DecimalByteParts`] array (the op is pushed down,
    /// never canonicalized) and (2) produces the same values as the canonical/Arrow path.
    fn check(
        bp_a: ArrayRef,
        bp_b: ArrayRef,
        canon_a: ArrayRef,
        canon_b: ArrayRef,
        op: Operator,
    ) -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let native = bp_a.binary(bp_b, op)?.execute::<ArrayRef>(&mut ctx)?;
        assert!(
            native.as_opt::<DecimalByteParts>().is_some(),
            "{op:?} should stay byte-parts, got {}",
            native.encoding_id()
        );
        let native = native.execute::<DecimalArray>(&mut ctx)?.into_array();
        let canonical = canon_a
            .binary(canon_b, op)?
            .execute::<DecimalArray>(&mut ctx)?
            .into_array();
        assert_arrays_eq!(native, canonical);
        Ok(())
    }

    #[rstest]
    #[case(Operator::Add)]
    #[case(Operator::Sub)]
    fn i128_matches_canonical(#[case] op: Operator) -> VortexResult<()> {
        let dtype = DecimalDType::new(38, 2);
        let p = 10i128.pow(30);
        // Cases that exercise carry/borrow across the limb boundary, sign changes, and nulls.
        let a = [
            Some(p),
            Some(p),
            Some(-p),
            Some(u64::MAX as i128),
            None,
            Some(7),
        ];
        let b = [Some(p), Some(-p), Some(-p), Some(3), Some(5), None];
        let (bp_a, canon_a) = i128_pair(&a, dtype);
        let (bp_b, canon_b) = i128_pair(&b, dtype);
        check(bp_a, bp_b, canon_a, canon_b, op)
    }

    #[rstest]
    #[case(Operator::Add)]
    #[case(Operator::Sub)]
    fn i256_matches_canonical(#[case] op: Operator) -> VortexResult<()> {
        let dtype = DecimalDType::new(60, 2);
        // `from_parts(lower: u128, upper: i128)`. Keep `upper` small so values stay within precision
        // 60, but use large lower limbs to exercise carry/borrow across the 64-bit limb boundaries
        // and from the lower 128 bits up into the msp.
        let v = |upper: i128, lower: u128| i256::from_parts(lower, upper);
        let a = [
            Some(v(10, u128::MAX)),       // lower overflows -> carry into msp
            Some(v(3, u64::MAX as u128)), // low u64 overflows -> carry into next limb
            Some(v(-10, 5)),              // negative msp
            Some(v(7, 0)),
            None,
            Some(i256::from_i128(7)),
        ];
        let b = [
            Some(v(10, 1)),
            Some(v(3, 1)),
            Some(v(0, 1)),
            Some(v(-7, 9)),
            Some(i256::from_i128(5)),
            None,
        ];
        let (bp_a, canon_a) = i256_pair(&a, dtype);
        let (bp_b, canon_b) = i256_pair(&b, dtype);
        check(bp_a, bp_b, canon_a, canon_b, op)
    }

    /// Overflow past the storage width must error on both paths, matching Arrow's checked add.
    #[test]
    fn add_overflow_errors() -> VortexResult<()> {
        let dtype = DecimalDType::new(38, 2);
        let big = 10i128.pow(38) - 1; // 38 nines: valid at precision 38, but the sum overflows i128
        let (bp_a, canon_a) = i128_pair(&[Some(big)], dtype);
        let (bp_b, canon_b) = i128_pair(&[Some(big)], dtype);
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        assert!(
            bp_a.binary(bp_b, Operator::Add)?
                .execute::<ArrayRef>(&mut ctx)
                .is_err(),
            "byte-parts add should detect overflow"
        );
        assert!(
            canon_a
                .binary(canon_b, Operator::Add)?
                .execute::<ArrayRef>(&mut ctx)
                .is_err(),
            "canonical add should detect overflow"
        );
        Ok(())
    }

    fn single_i32(values: &[Option<i32>], dtype: DecimalDType) -> (ArrayRef, ArrayRef) {
        let validity = validity_of(values.iter().map(Option::is_some));
        let msp = PrimitiveArray::new(
            values
                .iter()
                .map(|v| v.unwrap_or(0))
                .collect::<Buffer<i32>>(),
            validity,
        );
        let bp = DecimalByteParts::try_new(msp.into_array(), dtype)
            .unwrap()
            .into_array();
        let canon = DecimalArray::from_option_iter(values.iter().copied(), dtype).into_array();
        (bp, canon)
    }

    fn single_i64(values: &[Option<i64>], dtype: DecimalDType) -> (ArrayRef, ArrayRef) {
        let validity = validity_of(values.iter().map(Option::is_some));
        let msp = PrimitiveArray::new(
            values
                .iter()
                .map(|v| v.unwrap_or(0))
                .collect::<Buffer<i64>>(),
            validity,
        );
        let bp = DecimalByteParts::try_new(msp.into_array(), dtype)
            .unwrap()
            .into_array();
        let canon = DecimalArray::from_option_iter(values.iter().copied(), dtype).into_array();
        (bp, canon)
    }

    #[rstest]
    #[case(Operator::Add)]
    #[case(Operator::Sub)]
    fn i32_matches_canonical(#[case] op: Operator) -> VortexResult<()> {
        let dtype = DecimalDType::new(9, 2);
        let a = [Some(123_456_789), Some(-100), None, Some(0), Some(50)];
        let b = [Some(1), Some(2_000), Some(7), None, Some(-50)];
        let (bp_a, canon_a) = single_i32(&a, dtype);
        let (bp_b, canon_b) = single_i32(&b, dtype);
        check(bp_a, bp_b, canon_a, canon_b, op)
    }

    #[rstest]
    #[case(Operator::Add)]
    #[case(Operator::Sub)]
    fn i64_matches_canonical(#[case] op: Operator) -> VortexResult<()> {
        let dtype = DecimalDType::new(18, 2);
        let a = [Some(1_000_000_000_000i64), Some(-5), None, Some(0)];
        let b = [Some(2i64), Some(3), Some(9), None];
        let (bp_a, canon_a) = single_i64(&a, dtype);
        let (bp_b, canon_b) = single_i64(&b, dtype);
        check(bp_a, bp_b, canon_a, canon_b, op)
    }

    /// A `Decimal(38,2)` whose values fit `i64` is narrowed to a single `i64` part. Its natural
    /// storage is `i128`, so add/sub must fall back to the canonical path (not push down) — and must
    /// not false-overflow at the `i64` boundary when the true sum exceeds `i64::MAX` but fits the
    /// decimal precision.
    #[test]
    fn narrowed_wide_decimal_is_correct() -> VortexResult<()> {
        let dtype = DecimalDType::new(38, 2);
        let big = 8_000_000_000_000_000_000i64; // ~8e18; sum 1.6e19 overflows i64, fits Decimal(38,2)
        let bp = |x: i64| {
            DecimalByteParts::try_new(
                PrimitiveArray::new(
                    [x].into_iter().collect::<Buffer<i64>>(),
                    Validity::NonNullable,
                )
                .into_array(),
                dtype,
            )
            .unwrap()
            .into_array()
        };
        let canon = |x: i64| DecimalArray::from_iter([i128::from(x)], dtype).into_array();

        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let native = bp(big)
            .binary(bp(big), Operator::Add)?
            .execute::<DecimalArray>(&mut ctx)?
            .into_array();
        let expected = canon(big)
            .binary(canon(big), Operator::Add)?
            .execute::<DecimalArray>(&mut ctx)?
            .into_array();
        assert_arrays_eq!(native, expected);
        Ok(())
    }

    /// Differential check around the `Decimal(38,2)` overflow boundary: byte-parts add must agree
    /// with the canonical (Arrow-backed) path either by producing equal arrays or by both erroring.
    #[test]
    fn overflow_semantics_match_arrow() -> VortexResult<()> {
        let dtype = DecimalDType::new(38, 2);
        let xs = [
            10i128.pow(36),     // sum 2e36, well inside
            6 * 10i128.pow(37), // sum 1.2e38, exceeds 38 digits but fits i128
            9 * 10i128.pow(37), // sum 1.8e38, overflows i128
            10i128.pow(38) - 1, // 38 nines, sum overflows i128
        ];
        for x in xs {
            let (bp_a, canon_a) = i128_pair(&[Some(x)], dtype);
            let (bp_b, canon_b) = i128_pair(&[Some(x)], dtype);
            let mut ctx = LEGACY_SESSION.create_execution_ctx();
            let bp = bp_a
                .binary(bp_b, Operator::Add)?
                .execute::<DecimalArray>(&mut ctx);
            let canon = canon_a
                .binary(canon_b, Operator::Add)?
                .execute::<DecimalArray>(&mut ctx);
            assert_eq!(
                bp.is_ok(),
                canon.is_ok(),
                "byte-parts and canonical disagree on ok/err for x={x}"
            );
            if let (Ok(bp), Ok(canon)) = (bp, canon) {
                // Compare raw i128 buffers directly: the precision-band results are valid i128 values
                // but exceed 38 digits, so building decimal scalars (as `assert_arrays_eq` does) would
                // fail — yet both paths still produce the same bits.
                assert_eq!(
                    bp.buffer::<i128>().as_slice(),
                    canon.buffer::<i128>().as_slice(),
                    "byte-parts and canonical produced different values for x={x}"
                );
            }
        }
        Ok(())
    }
}
