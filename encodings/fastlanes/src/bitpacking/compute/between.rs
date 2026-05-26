// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Block-streaming between kernel for [`BitPackedArray`] against constant bounds.
//!
//! Reuses the same single-block scratch buffer as the compare kernel and folds a
//! `lower op_l v op_u upper` predicate per element, so the full primitive never
//! materialises.

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
use vortex_array::match_each_integer_ptype;
use vortex_array::scalar_fn::fns::between::BetweenKernel;
use vortex_array::scalar_fn::fns::between::BetweenOptions;
use vortex_array::scalar_fn::fns::between::StrictComparison;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::BitPacked;
use crate::bitpacking::compute::stream_predicate::stream_predicate;

impl BetweenKernel for BitPacked {
    fn between(
        array: ArrayView<'_, Self>,
        lower: &ArrayRef,
        upper: &ArrayRef,
        options: &BetweenOptions,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
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
                .vortex_expect("between precondition strips null lower");
            let up: T = upper_prim
                .typed_value::<T>()
                .vortex_expect("between precondition strips null upper");
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
    T: NativePType + Copy + crate::unpack_iter::BitPacked,
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
    T: NativePType + Copy + crate::unpack_iter::BitPacked,
    Lo: Fn(T, T) -> bool,
    Up: Fn(T, T) -> bool,
{
    stream_predicate::<T, _>(
        array,
        nullability,
        |v| lower_fn(lower, v) & upper_fn(v, upper),
        ctx,
    )
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::LEGACY_SESSION;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::builtins::ArrayBuiltins;
    use vortex_array::scalar_fn::fns::between::BetweenOptions;
    use vortex_array::scalar_fn::fns::between::StrictComparison;
    use vortex_array::validity::Validity;
    use vortex_buffer::BufferMut;
    use vortex_error::VortexResult;

    use crate::BitPackedArrayExt;
    use crate::BitPackedData;

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
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let values: BufferMut<u32> = (0..3000u32).map(|i| i % 257).collect();
        let prim = PrimitiveArray::new(values.freeze(), Validity::NonNullable);
        let packed = BitPackedData::encode(&prim.clone().into_array(), 9, &mut ctx)?;

        let lower = ConstantArray::new(40u32, prim.len()).into_array();
        let upper = ConstantArray::new(200u32, prim.len()).into_array();
        let options = opts(lower_strict, upper_strict);

        let expected = prim
            .into_array()
            .between(lower.clone(), upper.clone(), options.clone())?
            .execute::<BoolArray>(&mut ctx)?;
        let actual = packed
            .into_array()
            .between(lower, upper, options)?
            .execute::<BoolArray>(&mut ctx)?;

        assert_arrays_eq!(actual, expected);
        Ok(())
    }

    #[test]
    fn signed_with_patches_against_primitive_baseline() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
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
            .between(lower.clone(), upper.clone(), options.clone())?
            .execute::<BoolArray>(&mut ctx)?;
        let actual = packed
            .into_array()
            .between(lower, upper, options)?
            .execute::<BoolArray>(&mut ctx)?;

        assert_arrays_eq!(actual, expected);
        Ok(())
    }

    #[test]
    fn nullable_propagates_validity() -> VortexResult<()> {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let prim =
            PrimitiveArray::from_option_iter([Some(1u32), None, Some(3), Some(4), None, Some(6)]);
        let packed = BitPackedData::encode(&prim.clone().into_array(), 3, &mut ctx)?;

        let lower = ConstantArray::new(2u32, packed.len()).into_array();
        let upper = ConstantArray::new(5u32, packed.len()).into_array();
        let options = opts(StrictComparison::NonStrict, StrictComparison::NonStrict);

        let actual = packed
            .into_array()
            .between(lower.clone(), upper.clone(), options.clone())?
            .execute::<BoolArray>(&mut ctx)?;
        let expected = prim
            .into_array()
            .between(lower, upper, options)?
            .execute::<BoolArray>(&mut ctx)?;
        assert_arrays_eq!(actual, expected);
        Ok(())
    }
}
