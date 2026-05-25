// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::ScalarFn;
use vortex_array::arrays::scalar_fn::ExactScalarFn;
use vortex_array::arrays::scalar_fn::ScalarFnArrayExt;
use vortex_array::arrays::scalar_fn::ScalarFnArrayView;
use vortex_array::dtype::PType;
use vortex_array::kernel::ExecuteParentKernel;
use vortex_array::scalar_fn::fns::binary::Binary;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_array::validity::Validity;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

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

        // Both operands must share the standard wide layout: signed i64 msp + unsigned u64 limbs of
        // matching arity. Otherwise fall back to the canonical (Arrow) path.
        let k = array.num_lower_parts();
        if other.num_lower_parts() != k
            || !array.dtype().eq_ignore_nullability(other.dtype())
            || !is_i64(array.msp())
            || !is_i64(other.msp())
            || (0..k).any(|i| !is_u64(array.lower_part(i)) || !is_u64(other.lower_part(i)))
        {
            return Ok(None);
        }

        // Compute `lhs OP rhs` matching the original expression. Add is commutative; Sub is not, so
        // when our array is the right-hand child we swap the operands.
        let (lhs, rhs) = match (op, child_idx) {
            (Op::Sub, 1) => (&other, &array),
            _ => (&array, &other),
        };
        add_sub(lhs, rhs, op, ctx).map(Some)
    }
}

fn is_i64(part: &ArrayRef) -> bool {
    PType::try_from(part.dtype()).ok() == Some(PType::I64)
}

fn is_u64(part: &ArrayRef) -> bool {
    PType::try_from(part.dtype()).ok() == Some(PType::U64)
}

/// Limb-wise multi-precision add/subtract of two same-layout byte-parts columns.
///
/// Each limb column is processed least-significant-first so the carry (borrow) chain runs across
/// limbs while every per-limb pass stays a straight loop over rows that vectorizes across lanes.
fn add_sub(
    lhs: &ArrayView<'_, DecimalByteParts>,
    rhs: &ArrayView<'_, DecimalByteParts>,
    op: Op,
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

    // The msp carries the sign, but two's-complement makes the high-limb add/sub identical to the
    // unsigned operation reinterpreted as signed.
    let mut out_msp = BufferMut::<i64>::zeroed(n);
    let msp = out_msp.as_mut_slice();
    if is_add {
        for i in 0..n {
            msp[i] = (a0[i] as u64)
                .wrapping_add(b0[i] as u64)
                .wrapping_add(carry[i]) as i64;
        }
    } else {
        for i in 0..n {
            msp[i] = (a0[i] as u64)
                .wrapping_sub(b0[i] as u64)
                .wrapping_sub(carry[i]) as i64;
        }
    }

    let validity = a_msp.validity()?.and(b_msp.validity()?)?;
    let msp_array = PrimitiveArray::new(out_msp.freeze(), validity).into_array();
    let lowers: Vec<ArrayRef> = out_lo
        .into_iter()
        .map(|buf| PrimitiveArray::new(buf.freeze(), Validity::NonNullable).into_array())
        .collect();

    let dtype = *lhs
        .dtype()
        .as_decimal_opt()
        .vortex_expect("operands validated as decimal");
    Ok(DecimalByteParts::try_new_parts(msp_array, lowers, dtype)?.into_array())
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
}
