// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Block-streaming compare kernel for [`BitPackedArray`] against a constant.
//!
//! Avoids materialising the full primitive: the array is walked one 1024-element FastLanes
//! block at a time through a reusable scratch buffer, and a per-element bool is folded into
//! a [`BitBuffer`]. Patches are re-applied at the end by overwriting bits at the patched
//! indices with `predicate(patch_value)`.
//!
//! [`BitPackedArray`]: crate::BitPackedArray
//! [`BitBuffer`]: vortex_buffer::BitBuffer

use fastlanes::BitPacking;
use fastlanes::BitPackingCompare;
use fastlanes::FastLanesComparable;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PhysicalPType;
use vortex_array::match_each_integer_ptype;
use vortex_array::scalar_fn::fns::binary::CompareKernel;
use vortex_array::scalar_fn::fns::operators::CompareOperator;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::BitPacked;
use crate::bitpacking::compute::compare_fused::stream_compare_fused;
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

/// Compare every value against the constant via the fused FastLanes `unpack_cmp` kernel.
///
/// `NativePType::is_eq` / `is_lt` etc. provide total comparison (matching the primitive between
/// kernel's dispatch shape). `NotEq` has no direct method, so use `!is_eq`.
fn compare_constant_typed<T>(
    lhs: ArrayView<'_, BitPacked>,
    rhs: T,
    operator: CompareOperator,
    nullability: Nullability,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef>
where
    T: NativePType
        + BitPackedIter
        + FastLanesComparable<Bitpacked = <T as PhysicalPType>::Physical>,
    <T as PhysicalPType>::Physical: BitPacking + NativePType + BitPackingCompare,
{
    match operator {
        CompareOperator::Eq => {
            stream_compare_fused::<T, _>(lhs, rhs, nullability, |a, b| a.is_eq(b), ctx)
        }
        CompareOperator::NotEq => {
            stream_compare_fused::<T, _>(lhs, rhs, nullability, |a, b| !a.is_eq(b), ctx)
        }
        CompareOperator::Lt => {
            stream_compare_fused::<T, _>(lhs, rhs, nullability, |a, b| a.is_lt(b), ctx)
        }
        CompareOperator::Lte => {
            stream_compare_fused::<T, _>(lhs, rhs, nullability, |a, b| a.is_le(b), ctx)
        }
        CompareOperator::Gt => {
            stream_compare_fused::<T, _>(lhs, rhs, nullability, |a, b| a.is_gt(b), ctx)
        }
        CompareOperator::Gte => {
            stream_compare_fused::<T, _>(lhs, rhs, nullability, |a, b| a.is_ge(b), ctx)
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
    use vortex_array::arrays::slice::SliceKernel;
    use vortex_array::assert_arrays_eq;
    use vortex_array::builtins::ArrayBuiltins;
    use vortex_array::scalar_fn::fns::binary::CompareKernel;
    use vortex_array::scalar_fn::fns::operators::CompareOperator;
    use vortex_array::scalar_fn::fns::operators::Operator;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::BitPacked;
    use crate::BitPackedArrayExt;
    use crate::BitPackedData;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(vortex_array::array_session);

    /// All six operators on a small in-range input.
    #[rstest]
    #[case(Operator::Eq, vec![false, false, false, true, false, false, true])]
    #[case(Operator::NotEq, vec![true, true, true, false, true, true, false])]
    #[case(Operator::Lt, vec![true, true, true, false, false, false, false])]
    #[case(Operator::Lte, vec![true, true, true, true, false, false, true])]
    #[case(Operator::Gt, vec![false, false, false, false, true, true, false])]
    #[case(Operator::Gte, vec![false, false, false, true, true, true, true])]
    fn small(#[case] op: Operator, #[case] expected: Vec<bool>) {
        let mut ctx = SESSION.create_execution_ctx();
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
                let mut ctx = SESSION.create_execution_ctx();
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
        let mut ctx = SESSION.create_execution_ctx();
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

    /// Sliced inputs: a non-zero block offset (and a length spanning several blocks) must still go
    /// through the fused kernel and agree with the primitive fallback. Sweeps slice starts that
    /// land both inside the first block and past it, with lengths that end mid-block and on a block
    /// boundary.
    #[rstest]
    #[case(1, 4000)] // start mid-first-block, multi-block length
    #[case(1023, 2)] // start at the last row of the first block
    #[case(1024, 1024)] // start exactly on a block boundary, exactly one block long
    #[case(1500, 1000)] // start mid-second-block
    #[case(3, 1021)] // ends exactly on the first block boundary
    fn sliced_matches_primitive(
        #[case] start: usize,
        #[case] slice_len: usize,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values: Vec<u32> = (0..5000u32).map(|i| i % 128).collect();
        let prim = PrimitiveArray::from_iter(values);
        let packed = BitPackedData::encode(&prim.clone().into_array(), 7, &mut ctx)?;

        let sliced = packed.into_array().slice(start..start + slice_len)?;
        let rhs = ConstantArray::new(50u32, slice_len).into_array();
        for op in [
            CompareOperator::Eq,
            CompareOperator::Lt,
            CompareOperator::Gte,
        ] {
            let got = <BitPacked as CompareKernel>::compare(
                sliced.as_::<BitPacked>(),
                &rhs,
                op,
                &mut ctx,
            )?
            .expect("fused compare kernel must engage for sliced arrays")
            .execute::<BoolArray>(&mut ctx)?;
            let want = prim
                .clone()
                .into_array()
                .slice(start..start + slice_len)?
                .binary(rhs.clone(), Operator::from(op))?
                .execute::<BoolArray>(&mut ctx)?;
            assert_arrays_eq!(got, want);
        }
        Ok(())
    }

    /// Sliced *and* patched: combine a non-zero offset with out-of-range values that land in
    /// `Patches`, exercising the `offset + (global - p_off)` patch-position math.
    #[test]
    fn sliced_with_patches_matches_primitive() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values: Vec<i32> = (0..4096)
            .map(|i| if i % 91 == 0 { 100_000 + i } else { i % 100 })
            .collect();
        let prim = PrimitiveArray::from_iter(values);
        let packed = BitPackedData::encode(&prim.clone().into_array(), 7, &mut ctx)?;
        assert!(packed.patches().is_some(), "test setup expects patches");

        let (start, end) = (700usize, 3500usize);
        // `ArrayRef::slice` leaves a lazy `SliceArray` over a patched `BitPacked` (the
        // `SliceReduce` path bails when patches are present), so go through the `SliceKernel`,
        // which reads the buffers and produces a sliced `BitPacked` with sliced patches.
        let sliced = <BitPacked as SliceKernel>::slice(packed.as_view(), start..end, &mut ctx)?
            .expect("slice kernel produces a sliced bitpacked array");
        let rhs = ConstantArray::new(50i32, end - start).into_array();
        let got = <BitPacked as CompareKernel>::compare(
            sliced.as_::<BitPacked>(),
            &rhs,
            CompareOperator::Eq,
            &mut ctx,
        )?
        .expect("fused compare kernel must engage for sliced arrays with patches")
        .execute::<BoolArray>(&mut ctx)?;
        let want = prim
            .into_array()
            .slice(start..end)?
            .binary(rhs, Operator::Eq)?
            .execute::<BoolArray>(&mut ctx)?;
        assert_arrays_eq!(got, want);
        Ok(())
    }

    /// Nullable input — the result must carry the array's validity.
    #[test]
    fn nullable_propagates_validity() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
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
}
