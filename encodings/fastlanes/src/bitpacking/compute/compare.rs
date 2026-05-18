// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Fast-path comparison against a constant for bit-packed arrays.
//!
//! A bit-packed lane holds values in `[0, 2^bit_width - 1]`. When the RHS constant sits
//! outside that range, every packed lane has the same `Ordering` relative to `c`:
//!
//! * `c > 2^bit_width - 1` (above range) → every packed lane is `< c`
//! * `c < 0` (below range) → every packed lane is `> c` (packed values are non-negative)
//!
//! That collapses each of the six comparison operators to a constant boolean (modulo
//! patches and validity), so the result is either a `ConstantArray<bool>` (`O(1)`) or a
//! `BitBuffer` filled with that constant and overlaid with per-position results at any
//! patched indices.
//!
//! Detecting whether the constant falls in the packable range is an `O(1)` `i128` check
//! on the constant alone — strictly cheaper than encoding `c` into the bit-packed
//! representation, and layout-agnostic.
//!
//! **In-range constants** (those that could match a packed lane) fall through to the
//! canonical decompress + Arrow compare path. See `docs/inrange_compare_plan.md` for the
//! plan to accelerate that case for ordering operators.

use std::cmp::Ordering;

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
use vortex_array::dtype::PType;
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
        let Some(constant) = rhs.as_constant() else {
            return Ok(None);
        };
        let Some(constant) = constant.as_primitive_opt() else {
            return Ok(None);
        };

        match_each_integer_ptype!(constant.ptype(), |T| {
            compare_constant::<T>(
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

/// Ordering of every packed lane vs `constant` when `constant` is outside the packable
/// range. Returns `None` when `constant` itself fits in the range (no fast path applies).
///
/// `O(1)` check on the constant; never inspects the packed buffer.
#[inline]
fn constant_relation_to_packed<T>(constant: T, bit_width: u8) -> Option<Ordering>
where
    T: NativePType + ToPrimitive,
{
    let c = constant.to_i128()?;
    if c < 0 {
        return Some(Ordering::Greater);
    }
    let max = (1i128 << bit_width) - 1;
    if c > max {
        return Some(Ordering::Less);
    }
    None
}

/// Reduce `lane op constant` to a constant boolean when every packed lane has the same
/// ordering relation to `constant`.
#[inline]
fn reduce_constant(relation: Ordering, operator: CompareOperator) -> bool {
    match (operator, relation) {
        (CompareOperator::Eq, _) => false,
        (CompareOperator::NotEq, _) => true,
        (CompareOperator::Lt, Ordering::Less) => true,
        (CompareOperator::Lt, _) => false,
        (CompareOperator::Lte, Ordering::Less | Ordering::Equal) => true,
        (CompareOperator::Lte, _) => false,
        (CompareOperator::Gt, Ordering::Greater) => true,
        (CompareOperator::Gt, _) => false,
        (CompareOperator::Gte, Ordering::Greater | Ordering::Equal) => true,
        (CompareOperator::Gte, _) => false,
    }
}

fn compare_constant<T>(
    lhs: ArrayView<'_, BitPacked>,
    constant: T,
    rhs_nullability: Nullability,
    operator: CompareOperator,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>>
where
    T: NativePType + ToPrimitive,
{
    let Some(relation) = constant_relation_to_packed(constant, lhs.bit_width()) else {
        // In-range. Try the SWAR fast path for the supported width/storage; fall through
        // otherwise.
        return compare_in_range_swar::<T>(lhs, constant, rhs_nullability, operator);
    };

    let packed_lane_result = reduce_constant(relation, operator);
    let len = lhs.len();
    let validity = lhs.validity()?;
    let patches = lhs.patches();
    let result_nullability = lhs.dtype().nullability() | rhs_nullability;

    // Hot path: no patches, no nulls — every position has the same boolean result.
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
            apply_patches::<T, I>(
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

fn apply_patches<T, I>(
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
    let cmp: fn(T, T) -> bool = match operator {
        CompareOperator::Eq => |l, r| NativeValue(l) == NativeValue(r),
        CompareOperator::NotEq => |l, r| NativeValue(l) != NativeValue(r),
        CompareOperator::Lt => |l, r| NativeValue(l) < NativeValue(r),
        CompareOperator::Lte => |l, r| NativeValue(l) <= NativeValue(r),
        CompareOperator::Gt => |l, r| NativeValue(l) > NativeValue(r),
        CompareOperator::Gte => |l, r| NativeValue(l) >= NativeValue(r),
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

/// In-range SWAR / Knuth-broadword fast path. Currently scoped to power-of-two
/// `bit_width ∈ {1, 2, 4, 8, 16}` on `u32` / `i32` storage; everything else returns
/// `Ok(None)` and lets the canonical decompress + Arrow compare path run.
///
/// The kernel walks each 1024-chunk of the packed buffer, runs a Knuth-style broadword
/// comparison against the slot-replicated constant (no decompress, no unpacked
/// materialisation), writes per-element result bits directly into a `[u64; 16]`
/// element-order chunk bitmap, then pushes those 16 u64s into the result buffer.
fn compare_in_range_swar<T>(
    lhs: ArrayView<'_, BitPacked>,
    constant: T,
    rhs_nullability: Nullability,
    operator: CompareOperator,
) -> VortexResult<Option<ArrayRef>>
where
    T: NativePType + ToPrimitive,
{
    use super::compare_block::block_eq_u32;
    use super::compare_block::block_lt_u32;
    use super::compare_block::new_block;
    use super::compare_eq_w1::swar_eq_w1_u32;
    use super::compare_eq_w1::swar_lt_w1_u32;
    use super::compare_eq_w2::swar_eq_w2_u32;
    use super::compare_eq_w2::swar_lt_w2_u32;
    use super::compare_eq_w4::swar_eq_w4_u32;
    use super::compare_eq_w4::swar_lt_w4_u32;
    use super::compare_eq_w8::swar_eq_w8_u32;
    use super::compare_eq_w8::swar_lt_w8_u32;

    let w = lhs.bit_width();
    // Supported widths: 1..=16. Anything else returns Ok(None) → canonical.
    if !(1..=16).contains(&w) {
        return Ok(None);
    }
    if !matches!(T::PTYPE, PType::U32 | PType::I32) {
        return Ok(None);
    }
    if lhs.offset() != 0 || lhs.patches().is_some() {
        return Ok(None);
    }

    let len = lhs.len();
    let validity = lhs.validity()?;
    // Safe: `constant_relation_to_packed` returned `None`, so `0 <= c <= 2^W - 1 <= 65535`.
    let c_u32: u32 = constant
        .to_i128()
        .vortex_expect("integer constant fits in i128") as u32;

    // Two SWAR primitives: Eq and Lt. The other four operators derive from them:
    //   NotEq = !Eq            Lte = Lt | Eq
    //   Gt    = !(Lt | Eq)     Gte = !Lt

    let packed = lhs.packed_slice::<u32>();
    // packed_len per chunk = 32 (lanes) * W (words per lane).
    let elems_per_chunk: usize = 32 * (w as usize);
    let num_chunks = len.div_ceil(1024);

    let need_eq = matches!(
        operator,
        CompareOperator::Eq | CompareOperator::NotEq | CompareOperator::Lte | CompareOperator::Gt
    );
    let need_lt = matches!(
        operator,
        CompareOperator::Lt | CompareOperator::Lte | CompareOperator::Gt | CompareOperator::Gte
    );

    let n_words = len.div_ceil(64);
    let mut words = vortex_buffer::BufferMut::<u64>::with_capacity(n_words);

    let mut eq_bits = [0u64; 16];
    let mut lt_bits = [0u64; 16];
    // Reusable 4 KB unpack buffer for the block-decompress fallback (non-pow2 widths).
    // Allocated unconditionally — the stack cost is negligible and keeps the dispatch
    // branchless.
    let mut block = new_block();

    for chunk_idx in 0..num_chunks {
        let chunk = &packed[chunk_idx * elems_per_chunk..][..elems_per_chunk];
        eq_bits.fill(0);
        lt_bits.fill(0);

        // Only widths with explicit AVX2 kernels run here. The other widths bail this
        // path (`Ok(None)` higher up after we measure) so the canonical Arrow-compare
        // path runs — its SIMD `cmp::eq`/`cmp::lt` already wins over block-decompress
        // once the build has AVX2 enabled.
        if need_eq {
            match w {
                1 => swar_eq_w1_u32(chunk, c_u32 as u8, &mut eq_bits),
                2 => swar_eq_w2_u32(chunk, c_u32 as u8, &mut eq_bits),
                4 => swar_eq_w4_u32(chunk, c_u32 as u8, &mut eq_bits),
                8 => swar_eq_w8_u32(chunk, c_u32 as u8, &mut eq_bits),
                _ => block_eq_u32(chunk, w as usize, c_u32, &mut block, &mut eq_bits),
            }
        }
        if need_lt {
            match w {
                1 => swar_lt_w1_u32(chunk, c_u32 as u8, &mut lt_bits),
                2 => swar_lt_w2_u32(chunk, c_u32 as u8, &mut lt_bits),
                4 => swar_lt_w4_u32(chunk, c_u32 as u8, &mut lt_bits),
                8 => swar_lt_w8_u32(chunk, c_u32 as u8, &mut lt_bits),
                _ => block_lt_u32(chunk, w as usize, c_u32, &mut block, &mut lt_bits),
            }
        }

        let chunk_u64_start = chunk_idx * 16;
        let chunk_u64_end = (chunk_u64_start + 16).min(n_words);
        for i in chunk_u64_start..chunk_u64_end {
            let local = i - chunk_u64_start;
            let bit = match operator {
                CompareOperator::Eq => eq_bits[local],
                CompareOperator::NotEq => !eq_bits[local],
                CompareOperator::Lt => lt_bits[local],
                CompareOperator::Lte => lt_bits[local] | eq_bits[local],
                CompareOperator::Gt => !(lt_bits[local] | eq_bits[local]),
                CompareOperator::Gte => !lt_bits[local],
            };
            words.push(bit);
        }
    }

    let tail_bits = len % 64;
    if tail_bits != 0 {
        let last_idx = words.len() - 1;
        let mask = (1u64 << tail_bits) - 1;
        words[last_idx] &= mask;
    }

    let bits = vortex_buffer::BitBuffer::new_with_offset(words.freeze().into_byte_buffer(), len, 0);
    let validity = validity.union_nullability(rhs_nullability);
    Ok(Some(BoolArray::new(bits, validity).into_array()))
}

#[cfg(test)]
mod tests {
    use std::cmp::Ordering;
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
    use crate::bitpacking::compute::compare::constant_relation_to_packed;

    static SESSION: LazyLock<VortexSession> =
        LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

    #[test]
    fn range_check_is_o1() {
        // For an 8-bit packable range of [0, 255]:
        assert_eq!(
            constant_relation_to_packed::<i32>(256, 8),
            Some(Ordering::Less)
        );
        assert_eq!(
            constant_relation_to_packed::<i32>(-1, 8),
            Some(Ordering::Greater)
        );
        assert_eq!(constant_relation_to_packed::<i32>(255, 8), None);
        assert_eq!(constant_relation_to_packed::<i32>(0, 8), None);
    }

    #[rstest]
    #[case(Operator::Eq, false)]
    #[case(Operator::NotEq, true)]
    #[case(Operator::Lt, true)]
    #[case(Operator::Lte, true)]
    #[case(Operator::Gt, false)]
    #[case(Operator::Gte, false)]
    fn above_range_no_patches(#[case] op: Operator, #[case] expected: bool) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        // 999 is above the 8-bit packable range; every packed lane is < 999.
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
    #[case(Operator::Lt)]
    fn above_range_with_patches(#[case] op: Operator) -> VortexResult<()> {
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

        let cmp: fn(u32, u32) -> bool = match op {
            Operator::Eq => |l, r| l == r,
            Operator::Lt => |l, r| l < r,
            _ => unreachable!(),
        };
        assert_arrays_eq!(
            result,
            BoolArray::from_iter(values.iter().map(|v| cmp(*v, constant)))
        );
        Ok(())
    }

    #[test]
    fn below_range_signed() -> VortexResult<()> {
        // Packed signed values are non-negative, so -5 is always less than every lane.
        let mut ctx = SESSION.create_execution_ctx();
        let packed = BitPackedData::encode(
            &PrimitiveArray::from_iter([0i32, 7, 15, 3, 12]).into_array(),
            4,
            &mut ctx,
        )?;
        let len = packed.len();
        let result = packed
            .into_array()
            .binary(ConstantArray::new(-5i32, len).into_array(), Operator::Gt)?
            .execute::<BoolArray>(&mut ctx)?;
        assert_arrays_eq!(result, BoolArray::from_iter([true; 5]));
        Ok(())
    }

    #[test]
    fn in_range_falls_through() -> VortexResult<()> {
        // 100 is in the 8-bit packable range; fall through to the canonical path.
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
    fn nullable_constant() -> VortexResult<()> {
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
