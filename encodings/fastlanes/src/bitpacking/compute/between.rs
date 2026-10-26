// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Block-streaming between kernel for [`BitPackedArray`] against constant bounds.
//!
//! Reuses the same single-block scratch buffer as the compare kernel and folds a
//! `lower op_l v op_u upper` predicate per element, so the full primitive never
//! materialises.
//!
//! [`BitPackedArray`]: crate::BitPackedArray

use fastlanes::BitPacking;
use fastlanes::BitPackingCompare;
use fastlanes::FastLanesComparable;
use num_traits::AsPrimitive;
use num_traits::WrappingSub;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PhysicalPType;
use vortex_array::match_each_integer_ptype;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_functions::between::BetweenKernel;
use vortex_functions::between::BetweenOptions;
use vortex_functions::between::StrictComparison;

use crate::BitPacked;
use crate::BitPackedArrayExt;
use crate::BitWidthsView;
use crate::bitpacking::compute::compare_fused::stream_compare_fused;
use crate::bitpacking::compute::packed_range::constant_result;
use crate::bitpacking::compute::packed_range::packed_range;
use crate::bitpacking::compute::stream_predicate::stream_predicate;
use crate::unpack_iter::BitPacked as BitPackedIter;

impl BetweenKernel for BitPacked {
    fn between(
        array: ArrayView<'_, Self>,
        lower: &ArrayRef,
        upper: &ArrayRef,
        options: &BetweenOptions,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        if !array.bit_widths().is_global() {
            return Ok(None);
        }
        // Only accelerate constant-bounds between; vary-by-row bounds fall through to the
        // default `compare + and` pipeline.
        let (Some(lower_const), Some(upper_const)) = (lower.as_constant(), upper.as_constant())
        else {
            return Ok(None);
        };
        let (Some(lower_prim), Some(upper_prim)) = (
            lower_const.as_primitive_opt(),
            upper_const.as_primitive_opt(),
        ) else {
            return Ok(None);
        };

        let nullability =
            array.dtype().nullability() | lower.dtype().nullability() | upper.dtype().nullability();
        let arr_ptype = array.dtype().as_ptype();
        if lower_prim.ptype() != arr_ptype || upper_prim.ptype() != arr_ptype {
            return Ok(None);
        }

        let result = match_each_integer_ptype!(arr_ptype, |T| {
            let lo: T = lower_prim
                .typed_value::<T>()
                .vortex_expect("the between short circuit strips a null lower bound");
            let up: T = upper_prim
                .typed_value::<T>()
                .vortex_expect("the between short circuit strips a null upper bound");
            between_constant_typed::<T>(array, lo, up, options, nullability, ctx)?
        });
        Ok(Some(result))
    }
}

fn between_constant_typed<T>(
    array: ArrayView<'_, BitPacked>,
    lower: T,
    upper: T,
    options: &BetweenOptions,
    nullability: Nullability,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef>
where
    T: NativePType
        + BitPackedIter
        + FastLanesComparable<Bitpacked = <T as PhysicalPType>::Physical>
        + AsPrimitive<<T as PhysicalPType>::Physical>,
    <T as PhysicalPType>::Physical: BitPacking + NativePType + BitPackingCompare + WrappingSub,
{
    // Branch on strictness once at the top so each call into `between_impl` monomorphises
    // a single tight predicate — same shape as `Primitive::between` in `vortex-array`.
    match (options.lower_strict, options.upper_strict) {
        (StrictComparison::Strict, StrictComparison::Strict) => between_impl(
            array,
            lower,
            NativePType::is_lt,
            upper,
            NativePType::is_lt,
            nullability,
            ctx,
        ),
        (StrictComparison::Strict, StrictComparison::NonStrict) => between_impl(
            array,
            lower,
            NativePType::is_lt,
            upper,
            NativePType::is_le,
            nullability,
            ctx,
        ),
        (StrictComparison::NonStrict, StrictComparison::Strict) => between_impl(
            array,
            lower,
            NativePType::is_le,
            upper,
            NativePType::is_lt,
            nullability,
            ctx,
        ),
        (StrictComparison::NonStrict, StrictComparison::NonStrict) => between_impl(
            array,
            lower,
            NativePType::is_le,
            upper,
            NativePType::is_le,
            nullability,
            ctx,
        ),
    }
}

fn between_impl<T, Lo, Up>(
    array: ArrayView<'_, BitPacked>,
    lower: T,
    lower_fn: Lo,
    upper: T,
    upper_fn: Up,
    nullability: Nullability,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef>
where
    T: NativePType
        + BitPackedIter
        + FastLanesComparable<Bitpacked = <T as PhysicalPType>::Physical>
        + AsPrimitive<<T as PhysicalPType>::Physical>,
    <T as PhysicalPType>::Physical: BitPacking + NativePType + BitPackingCompare + WrappingSub,
    Lo: Fn(T, T) -> bool + Copy,
    Up: Fn(T, T) -> bool + Copy,
{
    let predicate = move |v| lower_fn(lower, v) & upper_fn(v, upper);
    let BitWidthsView::Global(bit_width) = array.bit_widths() else {
        return stream_predicate::<T, _>(array, nullability, predicate, ctx);
    };
    let Some((lo, hi)) = packed_range::<T>(bit_width) else {
        return stream_predicate::<T, _>(array, nullability, predicate, ctx);
    };

    // Both tests are monotone, so the ends of the packed range decide them.
    if lower_fn(lower, lo) && upper_fn(hi, upper) {
        return constant_result(array, true, nullability, &predicate, ctx);
    }
    if !lower_fn(lower, hi) || !upper_fn(lo, upper) {
        return constant_result(array, false, nullability, &predicate, ctx);
    }

    // Inclusive bounds clamped to the packed range, so one kernel serves every strictness.
    // A strict bound at the end of `T` returned above, so the step cannot overflow.
    let first = if lower_fn(lower, lower) {
        lower
    } else {
        lower + T::one()
    };
    let first = if first.is_lt(lo) { lo } else { first };
    let last = if upper_fn(upper, upper) {
        upper
    } else {
        upper - T::one()
    };
    let last = if last.is_gt(hi) { hi } else { last };
    if first.is_gt(last) {
        return constant_result(array, false, nullability, &predicate, ctx);
    }

    let kernel = offset_within::<T>(first, last);
    stream_compare_fused(
        array,
        bit_width,
        lower,
        nullability,
        kernel,
        &predicate,
        ctx,
    )
}

/// `first <= v <= last` as one unsigned compare: values below `first` wrap around.
fn offset_within<T>(first: T, last: T) -> impl Fn(T, T) -> bool + Copy
where
    T: NativePType + PhysicalPType + AsPrimitive<<T as PhysicalPType>::Physical>,
    <T as PhysicalPType>::Physical: NativePType + WrappingSub + PartialOrd,
{
    let span: <T as PhysicalPType>::Physical = (last - first).as_();
    let first: <T as PhysicalPType>::Physical = first.as_();
    move |v: T, _| {
        AsPrimitive::<<T as PhysicalPType>::Physical>::as_(v).wrapping_sub(&first) <= span
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::validity::Validity;
    use vortex_buffer::BufferMut;
    use vortex_error::VortexResult;
    use vortex_functions::between::BetweenKernel;
    use vortex_functions::between::BetweenOptions;
    use vortex_functions::between::StrictComparison;
    use vortex_functions::builtins::ArrayFunctions;
    use vortex_session::VortexSession;

    use crate::BitPacked;
    use crate::BitPackedArrayExt;
    use crate::BitPackedData;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();
        crate::initialize(&session);
        session
    });

    fn opts(lower: StrictComparison, upper: StrictComparison) -> BetweenOptions {
        BetweenOptions {
            lower_strict: lower,
            upper_strict: upper,
        }
    }

    #[rstest]
    #[case(StrictComparison::NonStrict, StrictComparison::NonStrict)]
    #[case(StrictComparison::Strict, StrictComparison::NonStrict)]
    #[case(StrictComparison::NonStrict, StrictComparison::Strict)]
    #[case(StrictComparison::Strict, StrictComparison::Strict)]
    fn multi_chunk_against_primitive_baseline(
        #[case] lower_strict: StrictComparison,
        #[case] upper_strict: StrictComparison,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values: BufferMut<u32> = (0..3000u32).map(|i| i % 257).collect();
        let prim = PrimitiveArray::new(values.freeze(), Validity::NonNullable);
        let packed = BitPackedData::encode(&prim.clone().into_array(), 9, &mut ctx)?;

        let lower = ConstantArray::new(40u32, prim.len()).into_array();
        let upper = ConstantArray::new(200u32, prim.len()).into_array();
        let options = opts(lower_strict, upper_strict);

        let expected = prim
            .into_array()
            .between(lower.clone(), upper.clone(), options.clone())?;
        let actual = packed.into_array().between(lower, upper, options)?;

        assert_arrays_eq!(actual, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn signed_with_patches_against_primitive_baseline() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values: Vec<i32> = (0..1500)
            .map(|i| if i % 73 == 0 { 100_000 + i } else { i % 100 })
            .collect();
        let prim = PrimitiveArray::from_iter(values);
        let packed = BitPackedData::encode(&prim.clone().into_array(), 7, &mut ctx)?;
        assert!(packed.patches().is_some(), "test setup expects patches");

        let lower = ConstantArray::new(20i32, prim.len()).into_array();
        let upper = ConstantArray::new(80i32, prim.len()).into_array();
        let options = opts(StrictComparison::NonStrict, StrictComparison::NonStrict);

        let expected = prim
            .into_array()
            .between(lower.clone(), upper.clone(), options.clone())?;
        let actual = packed.into_array().between(lower, upper, options)?;

        assert_arrays_eq!(actual, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn nullable_propagates_validity() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let prim =
            PrimitiveArray::from_option_iter([Some(1u32), None, Some(3), Some(4), None, Some(6)]);
        let packed = BitPackedData::encode(&prim.clone().into_array(), 3, &mut ctx)?;

        let lower = ConstantArray::new(2u32, packed.len()).into_array();
        let upper = ConstantArray::new(5u32, packed.len()).into_array();
        let options = opts(StrictComparison::NonStrict, StrictComparison::NonStrict);

        let actual = packed
            .into_array()
            .between(lower.clone(), upper.clone(), options.clone())?;
        let expected = prim.into_array().between(lower, upper, options)?;
        assert_arrays_eq!(actual, expected, &mut ctx);
        Ok(())
    }

    /// Bounds covering or missing `[0, 2^W - 1]` give a constant, patches and nulls still exact.
    #[rstest]
    fn decided_by_bit_width_matches_primitive(
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
        let bounds = [i32::MIN, -5, 0, 50, 127, 128, 1000];
        let strictness = [StrictComparison::Strict, StrictComparison::NonStrict];
        for (&lo, &up) in bounds
            .iter()
            .flat_map(|l| bounds.iter().map(move |u| (l, u)))
        {
            for (&lower_strict, &upper_strict) in strictness
                .iter()
                .flat_map(|l| strictness.iter().map(move |u| (l, u)))
            {
                let options = opts(lower_strict, upper_strict);
                let lower = ConstantArray::new(lo, prim.len()).into_array();
                let upper = ConstantArray::new(up, prim.len()).into_array();
                let got = <BitPacked as BetweenKernel>::between(
                    packed.as_view(),
                    &lower,
                    &upper,
                    &options,
                    &mut ctx,
                )?
                .expect("between kernel must engage");
                // Packed values span [0, 127].
                let inside = |v: i32| {
                    let above = match lower_strict {
                        StrictComparison::Strict => lo < v,
                        StrictComparison::NonStrict => lo <= v,
                    };
                    let below = match upper_strict {
                        StrictComparison::Strict => v < up,
                        StrictComparison::NonStrict => v <= up,
                    };
                    above && below
                };
                let decided = (0..=127).all(|v| inside(v) == inside(0));
                if !patched && !nullable {
                    assert_eq!(got.as_constant().is_some(), decided, "{lo} {up}");
                }
                let want = prim.clone().into_array().between(lower, upper, options)?;
                assert_arrays_eq!(got, want, &mut ctx);
            }
        }
        Ok(())
    }

    /// Patches past the packed width compare at full width.
    #[rstest]
    #[case(0, 1000)]
    #[case(1000, 1 << 26)]
    #[case(5, (1 << 27) - 1)]
    fn wide_patches_compare_exactly(#[case] lo: u64, #[case] up: u64) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = (0..5000u64).map(|i| match i % 89 {
            0 => (1 << 40) + i,
            1 => u64::MAX - i,
            2 => (1 << 32) + lo,
            _ => (i * 7919) % (1 << 27),
        });
        let prim = PrimitiveArray::from_iter(values);
        let packed = BitPackedData::encode(&prim.clone().into_array(), 27, &mut ctx)?;
        assert!(packed.patches().is_some(), "test setup expects patches");
        let lower = ConstantArray::new(lo, prim.len()).into_array();
        let upper = ConstantArray::new(up, prim.len()).into_array();
        for options in [
            opts(StrictComparison::NonStrict, StrictComparison::NonStrict),
            opts(StrictComparison::Strict, StrictComparison::Strict),
        ] {
            let got = <BitPacked as BetweenKernel>::between(
                packed.as_view(),
                &lower,
                &upper,
                &options,
                &mut ctx,
            )?
            .expect("between kernel must engage");
            let want = prim
                .clone()
                .into_array()
                .between(lower.clone(), upper.clone(), options)?;
            assert_arrays_eq!(got, want, &mut ctx);
        }
        Ok(())
    }
}
