// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Block-streaming compare kernel for [`DeltaArray`] against a constant.
//!
//! Avoids materialising the full primitive: the array is decompressed one 1024-element FastLanes
//! chunk at a time through reusable stack scratch, and a per-element bool is folded into a
//! [`vortex_buffer::BitBuffer`]. See [`super::stream_predicate`] for the streaming machinery.

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
use vortex_array::match_each_integer_ptype;
use vortex_array::scalar_fn::fns::binary::CompareKernel;
use vortex_array::scalar_fn::fns::operators::CompareOperator;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::Delta;
use crate::delta::compute::stream_predicate::stream_predicate;

impl CompareKernel for Delta {
    fn compare(
        lhs: ArrayView<'_, Self>,
        rhs: &ArrayRef,
        operator: CompareOperator,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        // Only accelerate compare-against-constant; everything else falls through to the
        // default decompress + Arrow compare pipeline.
        let Some(constant) = rhs.as_constant() else {
            return Ok(None);
        };
        let Some(constant_prim) = constant.as_primitive_opt() else {
            return Ok(None);
        };

        // The binary scalar-fn coerce step promotes both sides to a common ptype, and the
        // adaptor strips null-constant RHS.
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
    lhs: ArrayView<'_, Delta>,
    rhs: T,
    operator: CompareOperator,
    nullability: Nullability,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef>
where
    T: NativePType + Copy,
{
    // `NativePType::is_eq` / `is_lt` etc. provide total comparison. `NotEq` has no direct method,
    // so use `!is_eq`.
    match operator {
        CompareOperator::Eq => stream_predicate::<T, _>(lhs, nullability, |v| v.is_eq(rhs), ctx),
        CompareOperator::NotEq => {
            stream_predicate::<T, _>(lhs, nullability, |v| !v.is_eq(rhs), ctx)
        }
        CompareOperator::Lt => stream_predicate::<T, _>(lhs, nullability, |v| v.is_lt(rhs), ctx),
        CompareOperator::Lte => stream_predicate::<T, _>(lhs, nullability, |v| v.is_le(rhs), ctx),
        CompareOperator::Gt => stream_predicate::<T, _>(lhs, nullability, |v| v.is_gt(rhs), ctx),
        CompareOperator::Gte => stream_predicate::<T, _>(lhs, nullability, |v| v.is_ge(rhs), ctx),
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
    use vortex_array::scalar_fn::fns::binary::CompareKernel;
    use vortex_array::scalar_fn::fns::operators::CompareOperator;
    use vortex_array::scalar_fn::fns::operators::Operator;
    use vortex_array::session::ArraySession;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::Delta;

    static SESSION: LazyLock<VortexSession> =
        LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

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
        let delta = Delta::try_from_primitive_array(&values, &mut ctx).unwrap();
        let rhs = ConstantArray::new(3u32, delta.len()).into_array();
        let result = delta
            .into_array()
            .binary(rhs, op)
            .unwrap()
            .execute::<BoolArray>(&mut ctx)
            .unwrap();
        assert_arrays_eq!(result, BoolArray::from_iter(expected));
    }

    /// Sweep every native int type across several monotone delta sequences. 2048 elements span
    /// two FastLanes chunks. The kernel is invoked *directly* and asserted `Some`, proving the
    /// streaming path engages rather than silently falling back, and its output is checked
    /// against the Primitive fallback.
    macro_rules! sweep {
        ($name:ident, $T:ty) => {
            #[test]
            fn $name() -> VortexResult<()> {
                let mut ctx = SESSION.create_execution_ctx();
                let values: Vec<$T> = (0..2048i128).map(|i| (i % 97) as $T).collect();
                let prim = PrimitiveArray::from_iter(values);
                let delta = Delta::try_from_primitive_array(&prim, &mut ctx)?;
                let rhs = ConstantArray::new(40 as $T, prim.len()).into_array();
                for op in [
                    CompareOperator::Eq,
                    CompareOperator::Lt,
                    CompareOperator::Gte,
                ] {
                    let got =
                        <Delta as CompareKernel>::compare(delta.as_view(), &rhs, op, &mut ctx)?
                            .expect("streaming compare kernel must engage")
                            .execute::<BoolArray>(&mut ctx)?;
                    let want = prim
                        .clone()
                        .into_array()
                        .binary(rhs.clone(), Operator::from(op))?
                        .execute::<BoolArray>(&mut ctx)?;
                    assert_arrays_eq!(got, want);
                }
                Ok(())
            }
        };
    }

    sweep!(sweep_u8, u8);
    sweep!(sweep_u16, u16);
    sweep!(sweep_u32, u32);
    sweep!(sweep_u64, u64);
    sweep!(sweep_i8, i8);
    sweep!(sweep_i16, i16);
    sweep!(sweep_i32, i32);
    sweep!(sweep_i64, i64);

    /// Signed sequence crossing zero, sliced to exercise the offset/trailer windowing.
    #[test]
    fn signed_sliced_matches_primitive() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let prim: PrimitiveArray = (-1500i32..1500).collect();
        let delta = Delta::try_from_primitive_array(&prim, &mut ctx)?;
        let sliced = delta.slice(10..2500)?;
        let prim_sliced = prim.into_array().slice(10..2500)?;
        let rhs = ConstantArray::new(0i32, sliced.len()).into_array();
        let actual = sliced
            .binary(rhs.clone(), Operator::Lt)?
            .execute::<BoolArray>(&mut ctx)?;
        let expected = prim_sliced
            .binary(rhs, Operator::Lt)?
            .execute::<BoolArray>(&mut ctx)?;
        assert_arrays_eq!(actual, expected);
        Ok(())
    }

    /// Nullable input — the result must carry the array's validity.
    #[test]
    fn nullable_propagates_validity() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let prim =
            PrimitiveArray::from_option_iter([Some(1u32), None, Some(3), Some(4), None, Some(6)]);
        let delta = Delta::try_from_primitive_array(&prim, &mut ctx)?;
        let rhs = ConstantArray::new(3u32, delta.len()).into_array();
        let actual = delta
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
