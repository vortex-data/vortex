// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compare kernel for [`BitPackedArray`](crate::BitPackedArray) against a constant.
//!
//! Two complementary strategies, picked per call:
//!
//! * **Out-of-range fast path** — a bit-packed lane holds values in `[0, 2^bit_width - 1]`.
//!   When the RHS constant `c` sits outside that range, every packed lane has the same
//!   [`Ordering`] relative to `c`:
//!
//!   * `c > 2^bit_width - 1` (above range) → every packed lane is `< c`
//!   * `c < 0` (below range) → every packed lane is `> c` (packed values are non-negative)
//!
//!   so each operator collapses to a constant boolean (modulo patches and validity). With
//!   no patches and no nulls the result is an `O(1)` [`ConstantArray`]`<bool>`; otherwise a
//!   [`BitBuffer`](vortex_buffer::BitBuffer) is filled with that constant and the
//!   per-position outcome is overlaid at each patched index. Detecting the range is an
//!   `O(1)` `i128` check on the constant alone — strictly cheaper than encoding `c` into
//!   the bit-packed representation, and layout-agnostic.
//!
//! * **Streaming fallback** — in-range constants are evaluated by [`stream_compare`], which
//!   walks the array one 1024-element FastLanes block at a time and uses the fused
//!   unpack-and-compare kernel ([`fastlanes::BitPackingCompare`]) to compare each value
//!   in-register, folding the result straight into a `BitBuffer` without ever materialising
//!   the unpacked primitive.

use std::cmp::Ordering;

use fastlanes::BitPackingCompare;
use fastlanes::FastLanesComparable;
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
use vortex_array::dtype::PhysicalPType;
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
use crate::bitpacking::compute::stream_predicate::stream_compare;
use crate::unpack_iter::BitPacked as BitPackedIter;

impl CompareKernel for BitPacked {
    fn compare(
        lhs: ArrayView<'_, Self>,
        rhs: &ArrayRef,
        operator: CompareOperator,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        // Only accelerate compare-against-constant.
        let Some(constant) = rhs.as_constant() else {
            return Ok(None);
        };
        let Some(constant_prim) = constant.as_primitive_opt() else {
            return Ok(None);
        };

        // Adaptor strips null-constant RHS, and the binary scalar-fn coerce_args step has
        // already promoted both sides to a common ptype.
        let nullability = lhs.dtype().nullability() | rhs.dtype().nullability();
        let lhs_ptype = lhs.dtype().as_ptype();
        if constant_prim.ptype() != lhs_ptype {
            return Ok(None);
        }

        let result = match_each_integer_ptype!(lhs_ptype, |T| {
            let rhs: T = constant_prim
                .typed_value::<T>()
                .vortex_expect("compare adaptor strips null constants");
            compare_constant_typed::<T>(lhs, rhs, operator, nullability, ctx)?
        });
        Ok(Some(result))
    }
}

fn compare_constant_typed<T>(
    lhs: ArrayView<'_, BitPacked>,
    rhs: T,
    operator: CompareOperator,
    nullability: Nullability,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef>
where
    T: NativePType
        + Copy
        + ToPrimitive
        + BitPackedIter
        + FastLanesComparable<Bitpacked = <T as PhysicalPType>::Physical>,
    <T as PhysicalPType>::Physical: BitPackingCompare,
{
    // `O(1)` fast path: a constant outside the packable range compares identically against
    // every packed lane, so the answer is a constant boolean modulo patches and validity.
    if let Some(relation) = constant_relation_to_packed(rhs, lhs.bit_width()) {
        return compare_out_of_range::<T>(lhs, rhs, relation, operator, nullability, ctx);
    }

    // In-range: fused unpack-and-compare over the packed blocks. `NativePType::is_eq` /
    // `is_lt` etc. provide total comparison; `NotEq` has no direct method, so use `!is_eq`.
    match operator {
        CompareOperator::Eq => {
            stream_compare::<T, _>(lhs, rhs, |a, b| a.is_eq(b), nullability, ctx)
        }
        CompareOperator::NotEq => {
            stream_compare::<T, _>(lhs, rhs, |a, b| !a.is_eq(b), nullability, ctx)
        }
        CompareOperator::Lt => {
            stream_compare::<T, _>(lhs, rhs, |a, b| a.is_lt(b), nullability, ctx)
        }
        CompareOperator::Lte => {
            stream_compare::<T, _>(lhs, rhs, |a, b| a.is_le(b), nullability, ctx)
        }
        CompareOperator::Gt => {
            stream_compare::<T, _>(lhs, rhs, |a, b| a.is_gt(b), nullability, ctx)
        }
        CompareOperator::Gte => {
            stream_compare::<T, _>(lhs, rhs, |a, b| a.is_ge(b), nullability, ctx)
        }
    }
}

/// Build the comparison result for a constant that lies outside the packable range, given
/// the [`Ordering`] every packed lane has relative to it.
fn compare_out_of_range<T>(
    lhs: ArrayView<'_, BitPacked>,
    constant: T,
    relation: Ordering,
    operator: CompareOperator,
    nullability: Nullability,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef>
where
    T: NativePType,
{
    let packed_lane_result = reduce_constant(relation, operator);
    let len = lhs.len();
    let validity = lhs.validity()?;
    let patches = lhs.patches();

    // Hot path: no patches, no nulls — every position has the same boolean result.
    if patches.is_none() && validity.no_nulls() {
        return Ok(
            ConstantArray::new(Scalar::bool(packed_lane_result, nullability), len).into_array(),
        );
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

    let validity = validity.union_nullability(nullability);
    Ok(BoolArray::new(bits.freeze(), validity).into_array())
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

#[cfg(test)]
mod tests {
    use std::cmp::Ordering;

    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::LEGACY_SESSION;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::builtins::ArrayBuiltins;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::scalar::Scalar;
    use vortex_array::scalar_fn::fns::binary::CompareKernel;
    use vortex_array::scalar_fn::fns::operators::CompareOperator;
    use vortex_array::scalar_fn::fns::operators::Operator;
    use vortex_error::VortexResult;

    use crate::BitPacked;
    use crate::BitPackedArrayExt;
    use crate::BitPackedData;
    use crate::bitpacking::compute::compare::constant_relation_to_packed;

    /// All six operators on a small in-range input (streaming path).
    #[rstest]
    #[case(Operator::Eq, vec![false, false, false, true, false, false, true])]
    #[case(Operator::NotEq, vec![true, true, true, false, true, true, false])]
    #[case(Operator::Lt, vec![true, true, true, false, false, false, false])]
    #[case(Operator::Lte, vec![true, true, true, true, false, false, true])]
    #[case(Operator::Gt, vec![false, false, false, false, true, true, false])]
    #[case(Operator::Gte, vec![false, false, false, true, true, true, true])]
    fn small(#[case] op: Operator, #[case] expected: Vec<bool>) {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let values = PrimitiveArray::from_iter([0u32, 1, 2, 3, 4, 5, 3]);
        let packed = BitPackedData::encode(&values.into_array(), 3, &mut ctx).unwrap();
        let rhs = ConstantArray::new(3u32, packed.len()).into_array();
        let result = packed
            .into_array()
            .binary(rhs, op)
            .unwrap()
            .execute::<BoolArray>(&mut ctx)
            .unwrap();
        assert_arrays_eq!(result, BoolArray::from_iter(expected));
    }

    /// Sweep every native int type across several bit-widths. 2048 elements spans two
    /// FastLanes blocks, exercising the per-type monomorphised inner loop. The kernel is
    /// invoked *directly* and asserted `Some`, proving the streaming path engages (rather
    /// than silently falling back to the Arrow compare), and its output is checked against
    /// the Primitive fallback.
    macro_rules! sweep {
        ($name:ident, $T:ty, $($bw:expr),+) => {
            #[test]
            fn $name() -> VortexResult<()> {
                let mut ctx = LEGACY_SESSION.create_execution_ctx();
                for bw in [$($bw),+] {
                    let cap: u128 = 1u128 << bw;
                    let values: Vec<$T> = (0..2048u128).map(|i| (i % cap) as $T).collect();
                    let prim = PrimitiveArray::from_iter(values);
                    let packed = BitPackedData::encode(&prim.clone().into_array(), bw, &mut ctx)?;
                    let rhs_val = (cap.min(2048) / 2) as $T;
                    let rhs = ConstantArray::new(rhs_val, prim.len()).into_array();
                    for op in [CompareOperator::Eq, CompareOperator::Lt, CompareOperator::Gte] {
                        let got = <BitPacked as CompareKernel>::compare(
                            packed.as_view(), &rhs, op, &mut ctx,
                        )?
                        .expect("streaming compare kernel must engage")
                        .execute::<BoolArray>(&mut ctx)?;
                        let want = prim
                            .clone()
                            .into_array()
                            .binary(rhs.clone(), Operator::from(op))?
                            .execute::<BoolArray>(&mut ctx)?;
                        assert_arrays_eq!(got, want);
                    }
                }
                Ok(())
            }
        };
    }

    sweep!(sweep_u8, u8, 1, 4, 7);
    sweep!(sweep_u16, u16, 1, 8, 15);
    sweep!(sweep_u32, u32, 1, 16, 31);
    sweep!(sweep_u64, u64, 1, 32, 63);
    sweep!(sweep_i8, i8, 1, 4, 7);
    sweep!(sweep_i16, i16, 1, 8, 15);
    sweep!(sweep_i32, i32, 1, 16, 31);
    sweep!(sweep_i64, i64, 1, 32, 63);

    /// Inline-patch path: encode signed i32 values that exceed the bit-width range so they
    /// end up in `Patches`. The streaming kernel must splice the patches in before the
    /// predicate runs.
    #[test]
    fn signed_with_patches_matches_primitive() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let values: Vec<i32> = (0..1500)
            .map(|i| if i % 73 == 0 { 100_000 + i } else { i % 100 })
            .collect();
        let prim = PrimitiveArray::from_iter(values);
        let packed = BitPackedData::encode(&prim.clone().into_array(), 7, &mut ctx)?;
        assert!(packed.patches().is_some(), "test setup expects patches");
        let rhs = ConstantArray::new(50i32, prim.len()).into_array();
        let expected = prim
            .into_array()
            .binary(rhs.clone(), Operator::Eq)?
            .execute::<BoolArray>(&mut ctx)?;
        let actual = packed
            .into_array()
            .binary(rhs, Operator::Eq)?
            .execute::<BoolArray>(&mut ctx)?;
        assert_arrays_eq!(actual, expected);
        Ok(())
    }

    /// Nullable input — the result must carry the array's validity.
    #[test]
    fn nullable_propagates_validity() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let prim = PrimitiveArray::from_option_iter([Some(1u32), None, Some(3), Some(4), None]);
        let packed = BitPackedData::encode(&prim.clone().into_array(), 3, &mut ctx)?;
        let rhs = ConstantArray::new(3u32, packed.len()).into_array();
        let actual = packed
            .into_array()
            .binary(rhs.clone(), Operator::Eq)?
            .execute::<BoolArray>(&mut ctx)?;
        let expected = prim
            .into_array()
            .binary(rhs, Operator::Eq)?
            .execute::<BoolArray>(&mut ctx)?;
        assert_arrays_eq!(actual, expected);
        Ok(())
    }

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

    /// Out-of-range constant with no patches/nulls collapses to an `O(1)` `ConstantArray`.
    #[test]
    fn above_range_returns_constant_array() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let packed = BitPackedData::encode(
            &PrimitiveArray::from_iter([1u32, 2, 3, 250, 100]).into_array(),
            8,
            &mut ctx,
        )?;
        // 999 is above the 8-bit packable range; the fast path must fire and return a
        // ConstantArray rather than a materialised BoolArray.
        let rhs = ConstantArray::new(999u32, packed.len()).into_array();
        let result = <BitPacked as CompareKernel>::compare(
            packed.as_view(),
            &rhs,
            CompareOperator::Eq,
            &mut ctx,
        )?
        .expect("out-of-range fast path must engage");
        assert!(
            result.as_constant().is_some(),
            "out-of-range, no-patch, no-null compare must be O(1) ConstantArray"
        );
        let result = result.execute::<BoolArray>(&mut ctx)?;
        assert_arrays_eq!(result, BoolArray::from_iter([false; 5]));
        Ok(())
    }

    #[rstest]
    #[case(Operator::Eq, false)]
    #[case(Operator::NotEq, true)]
    #[case(Operator::Lt, true)]
    #[case(Operator::Lte, true)]
    #[case(Operator::Gt, false)]
    #[case(Operator::Gte, false)]
    fn above_range_no_patches(#[case] op: Operator, #[case] expected: bool) -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
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
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let values = [1u32, 5, 1000, 7, 1000, 14];
        let constant = 1000u32;

        let packed =
            BitPackedData::encode(&PrimitiveArray::from_iter(values).into_array(), 4, &mut ctx)?;
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
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
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
    fn nullable_constant() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
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
