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
use num_traits::AsPrimitive;
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
use crate::BitPackedArrayExt;
use crate::BitWidthsView;
use crate::bitpacking::compute::compare_fused::stream_compare_fused;
use crate::bitpacking::compute::packed_range::constant_result;
use crate::bitpacking::compute::packed_range::packed_range;
use crate::unpack_iter::BitPacked as BitPackedIter;

impl CompareKernel for BitPacked {
    fn compare(
        lhs: ArrayView<'_, Self>,
        rhs: &ArrayRef,
        operator: CompareOperator,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let BitWidthsView::Global(bit_width) = lhs.bit_widths() else {
            return Ok(None);
        };
        // Only accelerate compare-against-constant.
        let Some(constant) = rhs.as_constant() else {
            return Ok(None);
        };
        let Some(constant_prim) = constant.as_primitive_opt() else {
            return Ok(None);
        };

        // The adaptor strips a null-constant RHS, and binary expressions require both argument
        // types to match before dispatch.
        let nullability = lhs.dtype().nullability() | rhs.dtype().nullability();
        let lhs_ptype = lhs.dtype().as_ptype();
        if constant_prim.ptype() != lhs_ptype {
            return Ok(None);
        }

        let result = match_each_integer_ptype!(lhs_ptype, |T| {
            let rhs: T = constant_prim
                .typed_value::<T>()
                .vortex_expect("compare adaptor strips null constants");
            compare_constant_typed::<T>(lhs, bit_width, rhs, operator, nullability, ctx)?
        });
        Ok(Some(result))
    }
}

/// Compare every value against the constant via the fused FastLanes `unpack_cmp` kernel.
///
/// `NativePType::is_eq` / `is_lt` etc. provide total comparison (matching the primitive between
/// kernel's dispatch shape).
fn compare_constant_typed<T>(
    lhs: ArrayView<'_, BitPacked>,
    bit_width: u8,
    rhs: T,
    operator: CompareOperator,
    nullability: Nullability,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef>
where
    T: NativePType
        + BitPackedIter
        + FastLanesComparable<Bitpacked = <T as PhysicalPType>::Physical>
        + AsPrimitive<<T as PhysicalPType>::Physical>,
    <T as PhysicalPType>::Physical: BitPacking
        + NativePType
        + BitPackingCompare
        + FastLanesComparable<Bitpacked = <T as PhysicalPType>::Physical>,
{
    // `NotEq`, `Gte` and `Lte` invert the result of `Eq`, `Lt` and `Gt`, so each pair shares one
    // set of fused kernels.
    let (base, negate) = match operator {
        CompareOperator::Eq => (BaseCompare::Eq, false),
        CompareOperator::NotEq => (BaseCompare::Eq, true),
        CompareOperator::Lt => (BaseCompare::Lt, false),
        CompareOperator::Gte => (BaseCompare::Lt, true),
        CompareOperator::Gt => (BaseCompare::Gt, false),
        CompareOperator::Lte => (BaseCompare::Gt, true),
    };
    let cmp = move |value: T, other: T| base.apply(value, other);
    let exact = move |value: T| cmp(value, rhs) != negate;

    // The encoder rejects negative integers, so packed values are never negative. A negative
    // constant then compares the same against every packed value as against zero.
    if rhs.is_lt(T::zero()) {
        return constant_result(lhs, exact(T::zero()), nullability, &exact, ctx);
    }
    if let Some(value) = decided_by_bit_width(bit_width, rhs, operator, cmp) {
        return constant_result(lhs, value != negate, nullability, &exact, ctx);
    }

    // Both sides are now non-negative, so the unsigned physical values compare the same as the
    // values of `T`. The fused kernels take function items of the physical type, so signed and
    // unsigned arrays of one width share them.
    let rhs = rhs.as_();
    match base {
        BaseCompare::Eq => stream_compare_fused::<T, _, _>(
            lhs,
            bit_width,
            rhs,
            nullability,
            <T as PhysicalPType>::Physical::is_eq,
            negate,
            &exact,
            ctx,
        ),
        BaseCompare::Lt => stream_compare_fused::<T, _, _>(
            lhs,
            bit_width,
            rhs,
            nullability,
            <T as PhysicalPType>::Physical::is_lt,
            negate,
            &exact,
            ctx,
        ),
        BaseCompare::Gt => stream_compare_fused::<T, _, _>(
            lhs,
            bit_width,
            rhs,
            nullability,
            <T as PhysicalPType>::Physical::is_gt,
            negate,
            &exact,
            ctx,
        ),
    }
}

/// The comparison that the fused kernel runs. The other operators are its negations.
#[derive(Clone, Copy)]
enum BaseCompare {
    Eq,
    Lt,
    Gt,
}

impl BaseCompare {
    fn apply<T: NativePType>(self, value: T, other: T) -> bool {
        match self {
            Self::Eq => value.is_eq(other),
            Self::Lt => value.is_lt(other),
            Self::Gt => value.is_gt(other),
        }
    }
}

/// The result for every packed value, if the bit width alone decides it.
///
/// Orderings are decided when both ends of the packed range agree, equality only by a constant
/// outside the range or by W = 0.
fn decided_by_bit_width<T: NativePType>(
    bit_width: u8,
    rhs: T,
    operator: CompareOperator,
    cmp: impl Fn(T, T) -> bool,
) -> Option<bool> {
    let (lo, hi) = packed_range::<T>(bit_width)?;
    let at_lo = cmp(lo, rhs);
    let decided = match operator {
        CompareOperator::Eq | CompareOperator::NotEq => {
            rhs.is_lt(lo) || rhs.is_gt(hi) || lo.is_eq(hi)
        }
        _ => at_lo == cmp(hi, rhs),
    };
    decided.then_some(at_lo)
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
    use vortex_array::optimizer::ArrayOptimizer;
    use vortex_array::scalar_fn::fns::binary::CompareKernel;
    use vortex_array::scalar_fn::fns::operators::CompareOperator;
    use vortex_array::scalar_fn::fns::operators::Operator;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::BitPacked;
    use crate::BitPackedArrayExt;
    use crate::BitPackedData;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();
        crate::initialize(&session);
        session
    });

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
        let result = packed.into_array().binary(rhs, op).unwrap();
        assert_arrays_eq!(result, BoolArray::from_iter(expected), &mut ctx);
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
                    for op in [
                        CompareOperator::Eq,
                        CompareOperator::NotEq,
                        CompareOperator::Lt,
                        CompareOperator::Lte,
                        CompareOperator::Gt,
                        CompareOperator::Gte,
                    ] {
                        let got = <BitPacked as CompareKernel>::compare(
                            packed.as_view(), &rhs, op, &mut ctx,
                        )?
                        .expect("streaming compare kernel must engage");
                        let want = prim
                            .clone()
                            .into_array()
                            .binary(rhs.clone(), Operator::from(op))?;
                        assert_arrays_eq!(got, want, &mut SESSION.create_execution_ctx());
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

    /// Signed arrays use the unsigned physical kernels. A negative constant compares the same
    /// against every packed value, and at the widest packing the packed values still never set
    /// the sign bit.
    #[rstest]
    #[case::negative_constant(7, -3)]
    #[case::negative_constant_widest(31, -3)]
    #[case::positive_constant_widest(31, 1_000_000)]
    #[case::min_constant(7, i32::MIN)]
    fn signed_compare_matches_primitive(
        #[case] bit_width: u8,
        #[case] rhs_value: i32,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values: Vec<i32> = (0..2048)
            .map(|i| if i % 7 == 0 { i32::MAX - i } else { i % 100 })
            .collect();
        let prim = PrimitiveArray::from_iter(values);
        let packed = BitPackedData::encode(&prim.clone().into_array(), bit_width, &mut ctx)?;
        let rhs = ConstantArray::new(rhs_value, prim.len()).into_array();
        for op in [
            CompareOperator::Eq,
            CompareOperator::NotEq,
            CompareOperator::Lt,
            CompareOperator::Lte,
            CompareOperator::Gt,
            CompareOperator::Gte,
        ] {
            let got = <BitPacked as CompareKernel>::compare(packed.as_view(), &rhs, op, &mut ctx)?
                .expect("bit-packed compare kernel must engage");
            let want = prim
                .clone()
                .into_array()
                .binary(rhs.clone(), Operator::from(op))?;
            assert_arrays_eq!(got, want, &mut SESSION.create_execution_ctx());
        }
        Ok(())
    }

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
        let expected = prim.into_array().binary(rhs.clone(), Operator::Eq)?;
        let actual = packed.into_array().binary(rhs, Operator::Eq)?;
        assert_arrays_eq!(actual, expected, &mut ctx);
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

        let sliced = packed
            .into_array()
            .slice(start..start + slice_len)?
            .optimize(&SESSION)?;
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
            .expect("fused compare kernel must engage for sliced arrays");
            let want = prim
                .clone()
                .into_array()
                .slice(start..start + slice_len)?
                .binary(rhs.clone(), Operator::from(op))?;
            assert_arrays_eq!(got, want, &mut ctx);
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
        .expect("fused compare kernel must engage for sliced arrays with patches");
        let want = prim
            .into_array()
            .slice(start..end)?
            .binary(rhs, Operator::Eq)?;
        assert_arrays_eq!(got, want, &mut ctx);
        Ok(())
    }

    /// Nullable input — the result must carry the array's validity.
    #[test]
    fn nullable_propagates_validity() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let prim = PrimitiveArray::from_option_iter([Some(1u32), None, Some(3), Some(4), None]);
        let packed = BitPackedData::encode(&prim.clone().into_array(), 3, &mut ctx)?;
        let rhs = ConstantArray::new(3u32, packed.len()).into_array();
        let actual = packed.into_array().binary(rhs.clone(), Operator::Eq)?;
        let expected = prim.into_array().binary(rhs, Operator::Eq)?;
        assert_arrays_eq!(actual, expected, &mut ctx);
        Ok(())
    }

    const ALL_OPERATORS: [CompareOperator; 6] = [
        CompareOperator::Eq,
        CompareOperator::NotEq,
        CompareOperator::Lt,
        CompareOperator::Lte,
        CompareOperator::Gt,
        CompareOperator::Gte,
    ];

    /// A constant outside or on the edge of `[0, 2^W - 1]` gives a constant, patches and nulls
    /// still exact.
    #[rstest]
    fn decided_by_bit_width_matches_primitive(
        #[values(i32::MIN, -5, 0, 1, 127, 128, 1000, i32::MAX)] rhs: i32,
        #[values(false, true)] patched: bool,
        #[values(false, true)] nullable: bool,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = (0..3000i32).map(|i| match i % 97 {
            0 if patched => 100_000 + i,
            _ => i % 128,
        });
        let prim = if nullable {
            PrimitiveArray::from_option_iter(
                values.enumerate().map(|(i, v)| (i % 11 != 0).then_some(v)),
            )
        } else {
            PrimitiveArray::from_iter(values)
        };
        let packed = BitPackedData::encode(&prim.clone().into_array(), 7, &mut ctx)?;
        assert_eq!(packed.patches().is_some(), patched);
        let rhs_array = ConstantArray::new(rhs, prim.len()).into_array();
        for op in ALL_OPERATORS {
            let got =
                <BitPacked as CompareKernel>::compare(packed.as_view(), &rhs_array, op, &mut ctx)?
                    .expect("compare kernel must engage");
            // Packed values span [0, 127].
            let decided = match op {
                CompareOperator::Eq | CompareOperator::NotEq => !(0..=127).contains(&rhs),
                CompareOperator::Lt | CompareOperator::Gte => !(1..=127).contains(&rhs),
                CompareOperator::Lte | CompareOperator::Gt => !(0..127).contains(&rhs),
            };
            if !patched && !nullable {
                assert_eq!(got.as_constant().is_some(), decided, "{op:?} {rhs}");
            }
            let want = prim
                .clone()
                .into_array()
                .binary(rhs_array.clone(), Operator::from(op))?;
            assert_arrays_eq!(got, want, &mut ctx);
        }
        Ok(())
    }

    /// Width 0 is always decided.
    #[rstest]
    fn zero_bit_width(#[values(0u32, 1, u32::MAX)] rhs: u32) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let prim = PrimitiveArray::from_iter(std::iter::repeat_n(0u32, 1500));
        let packed = BitPackedData::encode(&prim.clone().into_array(), 0, &mut ctx)?;
        let rhs_array = ConstantArray::new(rhs, prim.len()).into_array();
        for op in ALL_OPERATORS {
            let got =
                <BitPacked as CompareKernel>::compare(packed.as_view(), &rhs_array, op, &mut ctx)?
                    .expect("compare kernel must engage");
            assert!(
                got.as_constant().is_some(),
                "{op:?} {rhs} should stay constant"
            );
            let want = prim
                .clone()
                .into_array()
                .binary(rhs_array.clone(), Operator::from(op))?;
            assert_arrays_eq!(got, want, &mut ctx);
        }
        Ok(())
    }

    /// Patches past the packed width compare at full width.
    #[rstest]
    fn wide_patches_compare_exactly(
        #[values(0u64, 1, 1 << 26, (1 << 27) - 1)] rhs: u64,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = (0..5000u64).map(|i| match i % 89 {
            0 => (1 << 40) + i,
            1 => u64::MAX - i,
            _ => (i * 7919) % (1 << 27),
        });
        let prim = PrimitiveArray::from_iter(values);
        let packed = BitPackedData::encode(&prim.clone().into_array(), 27, &mut ctx)?;
        assert!(packed.patches().is_some(), "test setup expects patches");
        let rhs_array = ConstantArray::new(rhs, prim.len()).into_array();
        for op in ALL_OPERATORS {
            let got =
                <BitPacked as CompareKernel>::compare(packed.as_view(), &rhs_array, op, &mut ctx)?
                    .expect("compare kernel must engage");
            let want = prim
                .clone()
                .into_array()
                .binary(rhs_array.clone(), Operator::from(op))?;
            assert_arrays_eq!(got, want, &mut ctx);
        }
        Ok(())
    }
}
