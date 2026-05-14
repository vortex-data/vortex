// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Fast-path `Eq` / `NotEq` comparison against a constant.
//!
//! When the constant cannot fit in the packable range `[0, 2^bit_width - 1]`, no value
//! stored in the packed buffer can equal it, so:
//!
//! * `Eq`    → every position is `false` (modulo patches/validity).
//! * `NotEq` → every position is `true`  (modulo patches/validity).
//!
//! Detecting this is an `O(1)` range check on the constant — strictly cheaper than
//! encoding `c` into the bit-packed representation. The check is layout-agnostic and
//! does not touch the packed buffer.
//!
//! In-range constants and ordering operators (`Lt`/`Lte`/`Gt`/`Gte`) currently fall
//! through to the canonical decompress + Arrow compare path.

use num_traits::ToPrimitive;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::primitive::NativeValue;
use vortex_array::dtype::IntegerPType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
use vortex_array::match_each_integer_ptype;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::fns::binary::CompareKernel;
use vortex_array::scalar_fn::fns::operators::CompareOperator;
use vortex_buffer::BitBufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::BitPacked;
use crate::BitPackedArrayExt;

impl CompareKernel for BitPacked {
    fn compare(
        lhs: ArrayView<'_, Self>,
        rhs: &ArrayRef,
        operator: CompareOperator,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        // Only `Eq` / `NotEq` are accelerated here. Ordering operators (`Lt`, `Lte`, `Gt`,
        // `Gte`) need either a SWAR less-than over the packed bytes or unpack-then-compare;
        // both are out of scope for this commit and fall through to the canonical path.
        if !matches!(operator, CompareOperator::Eq | CompareOperator::NotEq) {
            return Ok(None);
        }

        let Some(constant) = rhs.as_constant() else {
            return Ok(None);
        };
        let Some(constant) = constant.as_primitive_opt() else {
            return Ok(None);
        };

        match_each_integer_ptype!(constant.ptype(), |T| {
            compare_eq_constant::<T>(
                lhs,
                constant
                    .typed_value::<T>()
                    .vortex_expect("null scalar handled in adaptor"),
                rhs.dtype().nullability(),
                operator,
                ctx,
            )
        })
    }
}

/// Returns `true` if `constant` cannot fit in the packable range `[0, 2^bit_width - 1]`.
///
/// `O(1)` check on the constant; never inspects the packed buffer.
#[inline]
fn constant_out_of_packable_range<T>(constant: T, bit_width: u8) -> bool
where
    T: NativePType + ToPrimitive,
{
    let Some(c) = constant.to_i128() else {
        return false;
    };
    let max = (1i128 << bit_width) - 1;
    c < 0 || c > max
}

fn compare_eq_constant<T>(
    lhs: ArrayView<'_, BitPacked>,
    constant: T,
    rhs_nullability: Nullability,
    operator: CompareOperator,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>>
where
    T: NativePType + ToPrimitive,
{
    if !constant_out_of_packable_range(constant, lhs.bit_width()) {
        // Constant fits in the packable range, so at least some packed lanes could match
        // it. The fast path doesn't apply.
        return Ok(None);
    }

    // Every packed lane disagrees with `constant`. `Eq` is `false` everywhere, `NotEq` is
    // `true` everywhere — modulo patches (which carry the real value) and validity.
    let packed_lane_result = matches!(operator, CompareOperator::NotEq);
    let len = lhs.len();
    let validity = lhs.validity()?;
    let patches = lhs.patches();
    let result_nullability = lhs.dtype().nullability() | rhs_nullability;

    // Hot path: no patches, no nulls — every position has the same boolean result, so we
    // return a `ConstantArray<bool>` in `O(1)`.
    if patches.is_none() && validity.no_nulls() {
        return Ok(Some(
            ConstantArray::new(Scalar::bool(packed_lane_result, result_nullability), len)
                .into_array(),
        ));
    }

    let mut bits = BitBufferMut::full(packed_lane_result, len);

    if let Some(patches) = patches {
        let indices = patches.indices().clone().execute::<PrimitiveArray>(ctx)?;
        let values = patches.values().clone().execute::<PrimitiveArray>(ctx)?;
        let patches_offset = patches.offset();

        match_each_unsigned_integer_ptype!(indices.ptype(), |I| {
            apply_eq_patches::<T, I>(
                &mut bits,
                indices.as_slice::<I>(),
                values.as_slice::<T>(),
                patches_offset,
                operator,
                constant,
            );
        });
    }

    let validity = validity.union_nullability(rhs_nullability);
    Ok(Some(BoolArray::new(bits.freeze(), validity).into_array()))
}

fn apply_eq_patches<T, I>(
    bits: &mut BitBufferMut,
    indices: &[I],
    values: &[T],
    indices_offset: usize,
    operator: CompareOperator,
    constant: T,
) where
    T: NativePType,
    I: IntegerPType,
{
    // Only Eq/NotEq reach this point (see `CompareKernel::compare`).
    let cmp: fn(T, T) -> bool = match operator {
        CompareOperator::Eq => |l, r| NativeValue(l) == NativeValue(r),
        CompareOperator::NotEq => |l, r| NativeValue(l) != NativeValue(r),
        _ => unreachable!("only Eq/NotEq reach the bitpacked compare-constant fast path"),
    };

    let len = bits.len();
    for (&raw_idx, &value) in indices.iter().zip(values.iter()) {
        let i: usize = raw_idx.as_();
        if i < indices_offset {
            continue;
        }
        let pos = i - indices_offset;
        if pos >= len {
            break;
        }
        if cmp(value, constant) {
            bits.set(pos);
        } else {
            bits.unset(pos);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::builtins::ArrayBuiltins;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::scalar::Scalar;
    use vortex_array::scalar_fn::fns::operators::Operator;
    use vortex_array::session::ArraySession;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::BitPackedArrayExt;
    use crate::BitPackedData;
    use crate::bitpacking::compute::compare::constant_out_of_packable_range;

    static SESSION: LazyLock<VortexSession> =
        LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

    #[test]
    fn range_check_is_o1() {
        // 8-bit packable range is [0, 255].
        assert!(constant_out_of_packable_range::<i32>(256, 8));
        assert!(constant_out_of_packable_range::<i32>(-1, 8));
        assert!(!constant_out_of_packable_range::<i32>(255, 8));
        assert!(!constant_out_of_packable_range::<i32>(0, 8));
    }

    #[rstest]
    #[case(Operator::Eq, false)]
    #[case(Operator::NotEq, true)]
    fn eq_above_range_no_patches(#[case] op: Operator, #[case] expected: bool) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        // 999 is above the 8-bit packable range; no packed lane matches.
        let packed = BitPackedData::encode(
            &PrimitiveArray::from_iter([1u32, 2, 3, 250, 100]).into_array(),
            8,
            &mut ctx,
        )?;
        let result = packed
            .into_array()
            .binary(ConstantArray::new(999u32, 5).into_array(), op)?
            .execute::<BoolArray>(&mut ctx)?;
        assert_arrays_eq!(result, BoolArray::from_iter([expected; 5]));
        Ok(())
    }

    #[rstest]
    #[case(Operator::Eq)]
    #[case(Operator::NotEq)]
    fn eq_above_range_with_patches(#[case] op: Operator) -> VortexResult<()> {
        // bit_width=4 packable range is [0, 15]; out-of-range values become patches.
        let mut ctx = SESSION.create_execution_ctx();
        let values = buffer![1u32, 5, 1000, 7, 1000, 14];
        let constant = 1000u32;

        let packed = BitPackedData::encode(&values.clone().into_array(), 4, &mut ctx)?;
        assert!(packed.patches().is_some());

        let result = packed
            .into_array()
            .binary(ConstantArray::new(constant, values.len()).into_array(), op)?
            .execute::<BoolArray>(&mut ctx)?;

        let expected: Vec<bool> = values
            .iter()
            .map(|v| match op {
                Operator::Eq => *v == constant,
                Operator::NotEq => *v != constant,
                _ => unreachable!(),
            })
            .collect();
        assert_arrays_eq!(result, BoolArray::from_iter(expected));
        Ok(())
    }

    #[test]
    fn ordering_falls_through() -> VortexResult<()> {
        // Ordering ops aren't accelerated yet; they go through the canonical path and
        // must still return a correct answer.
        let mut ctx = SESSION.create_execution_ctx();
        let values = [1u32, 2, 3, 250, 100];
        let packed =
            BitPackedData::encode(&PrimitiveArray::from_iter(values).into_array(), 8, &mut ctx)?;
        let result = packed
            .into_array()
            .binary(ConstantArray::new(999u32, 5).into_array(), Operator::Lt)?
            .execute::<BoolArray>(&mut ctx)?;
        assert_arrays_eq!(
            result,
            BoolArray::from_iter(values.iter().map(|v| *v < 999))
        );
        Ok(())
    }

    #[test]
    fn eq_in_range_falls_through() -> VortexResult<()> {
        // In-range constants must defer to the canonical path.
        let mut ctx = SESSION.create_execution_ctx();
        let values = [1u32, 2, 3, 250, 100];
        let packed =
            BitPackedData::encode(&PrimitiveArray::from_iter(values).into_array(), 8, &mut ctx)?;
        let result = packed
            .into_array()
            .binary(ConstantArray::new(100u32, 5).into_array(), Operator::Eq)?
            .execute::<BoolArray>(&mut ctx)?;
        assert_arrays_eq!(
            result,
            BoolArray::from_iter(values.iter().map(|v| *v == 100))
        );
        Ok(())
    }

    #[test]
    fn eq_nullable_constant() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let packed = BitPackedData::encode(
            &PrimitiveArray::from_iter([1u32, 2, 3]).into_array(),
            4,
            &mut ctx,
        )?;
        let rhs = ConstantArray::new(Scalar::primitive(999u32, Nullability::Nullable), 3);
        let result = packed
            .into_array()
            .binary(rhs.into_array(), Operator::Eq)?
            .execute::<BoolArray>(&mut ctx)?;
        assert_eq!(result.dtype(), &DType::Bool(Nullability::Nullable));
        Ok(())
    }
}
