// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ArrayView;
use crate::ExecutionCtx;
use crate::arrays::Primitive;
use crate::dtype::IntegerPType;
use crate::dtype::PType;
use crate::match_each_integer_ptype;
use crate::scalar_fn::fns::list_contains::IntegerMembership;
use crate::scalar_fn::fns::list_contains::ListContainsElementKernel;
use crate::scalar_fn::fns::list_contains::constant_list_scalar_contains;

/// Returns the source-member count where Primitive integer membership uses binary search.
#[doc(hidden)]
pub fn integer_membership_binary_search_min(ptype: PType) -> usize {
    // The generic implementation evaluates one equality expression per source member. Use the
    // prepared set once binary search becomes faster than the expression tree.
    match ptype.bit_width() {
        8 | 16 => 10,
        32 => 11,
        64 => 13,
        _ => 13,
    }
}

impl ListContainsElementKernel for Primitive {
    fn list_contains(
        list: &ArrayRef,
        element: ArrayView<'_, Self>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        evaluate_constant_list_membership(list, element, ctx)
    }
}

fn evaluate_constant_list_membership(
    list: &ArrayRef,
    element: ArrayView<'_, Primitive>,
    _ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    if !element.ptype().is_int() {
        return Ok(None);
    }

    let nullability = list.dtype().nullability() | element.dtype().nullability();

    match_each_integer_ptype!(element.ptype(), |T| {
        evaluate_integer_membership::<T>(list, element, nullability)
    })
}

fn evaluate_integer_membership<T: IntegerPType>(
    list: &ArrayRef,
    element: ArrayView<'_, Primitive>,
    nullability: crate::dtype::Nullability,
) -> VortexResult<Option<ArrayRef>> {
    let Some(membership) = IntegerMembership::<T>::try_from_constant_list(list, element.dtype())?
    else {
        return Ok(None);
    };
    evaluate_prepared_integer_membership(membership, element, nullability).map(Some)
}

/// Evaluates a prepared integer set against a Primitive array.
#[doc(hidden)]
pub fn evaluate_prepared_integer_membership<T: IntegerPType>(
    membership: IntegerMembership<T>,
    element: ArrayView<'_, Primitive>,
    nullability: crate::dtype::Nullability,
) -> VortexResult<ArrayRef> {
    if membership.members().len() > 4
        && membership.non_null_source_len() < integer_membership_binary_search_min(element.ptype())
    {
        return constant_list_scalar_contains(
            &membership.source_list().as_list(),
            element.array(),
            nullability,
        );
    }
    membership.evaluate_primitive(element, nullability)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use rstest::rstest;
    use vortex_error::VortexExpect;

    use super::*;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::BoolArray;
    use crate::arrays::Constant;
    use crate::arrays::ConstantArray;
    use crate::arrays::PrimitiveArray;
    use crate::assert_arrays_eq;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType::F32;
    use crate::dtype::PType::I32;
    use crate::dtype::PType::I64;
    use crate::expr::list_contains;
    use crate::expr::lit;
    use crate::expr::root;
    use crate::scalar::Scalar;
    #[cfg(not(codspeed))]
    use crate::test_harness::trace::trace_op;

    fn list(values: impl IntoIterator<Item = i32>, len: usize) -> ArrayRef {
        ConstantArray::new(
            Scalar::list(
                Arc::new(DType::Primitive(I32, Nullability::NonNullable)),
                values
                    .into_iter()
                    .map(|value| Scalar::primitive(value, Nullability::NonNullable))
                    .collect(),
                Nullability::NonNullable,
            ),
            len,
        )
        .into_array()
    }

    #[rstest]
    #[case::one(vec![3])]
    #[case::two(vec![3, 7])]
    #[case::three(vec![3, 7, 11])]
    #[case::four(vec![3, 7, 11, 15])]
    #[case::five((0..5).map(|value| value * 3).collect())]
    #[case::eleven((0..11).map(|value| value * 3).collect())]
    #[case::many((0..32).map(|value| value * 3).collect())]
    #[case::duplicate_heavy((0..32).map(|value| value % 5).collect())]
    fn test_membership_plans(#[case] members: Vec<i32>) -> VortexResult<()> {
        let mut ctx = crate::array_session().create_execution_ctx();
        let values = [0, 3, 7, 15, 31, 90_000, 310_000];
        let element = PrimitiveArray::from_iter(values);
        let expected = BoolArray::from_iter(values.map(|value| members.contains(&value)));

        let actual = <Primitive as ListContainsElementKernel>::list_contains(
            &list(members, element.len()),
            element.as_view(),
            &mut ctx,
        )?
        .vortex_expect("integer constant-list membership is supported");

        assert_arrays_eq!(actual, expected, &mut ctx);
        Ok(())
    }

    #[rstest]
    #[case::small(5)]
    #[case::many(13)]
    fn test_i64_membership(#[case] member_count: usize) -> VortexResult<()> {
        let mut ctx = crate::array_session().create_execution_ctx();
        let members = (0..member_count)
            .map(|value| i64::try_from(value).vortex_expect("member count fits i64"))
            .collect::<Vec<_>>();
        let values = [0i64, 11, 99];
        let element = PrimitiveArray::from_iter(values);
        let list = ConstantArray::new(
            Scalar::list(
                Arc::new(DType::Primitive(I64, Nullability::NonNullable)),
                members.iter().copied().map(Scalar::from).collect(),
                Nullability::NonNullable,
            ),
            element.len(),
        )
        .into_array();

        let actual = <Primitive as ListContainsElementKernel>::list_contains(
            &list,
            element.as_view(),
            &mut ctx,
        )?
        .vortex_expect("integer constant-list membership is supported");
        let expected = BoolArray::from_iter(values.map(|value| members.contains(&value)));
        assert_arrays_eq!(actual, expected, &mut ctx);
        Ok(())
    }

    #[test]
    #[cfg(not(codspeed))]
    fn test_registered_kernel_executes_through_expression() -> VortexResult<()> {
        let mut ctx = crate::array_session().create_execution_ctx();
        let values = [0i32, 1, 2, 3];
        let element = PrimitiveArray::from_iter(values);
        let members = [1, 3];
        let contains = element.into_array().apply(&list_contains(
            lit(list(members, values.len())
                .as_constant()
                .vortex_expect("constant list")),
            root(),
        ))?;

        let traced = trace_op(|| contains.execute::<BoolArray>(&mut ctx))?;
        let trace = traced.trace.to_string();
        let applied = trace
            .lines()
            .filter(|line| {
                line.contains("child_execute_parent session[")
                    && line.contains("slot=1")
                    && line.contains("parent=vortex.list.contains")
                    && line.contains("child=vortex.primitive")
            })
            .collect::<Vec<_>>();
        // A silent fallback preserves values but loses the membership optimization.
        assert_eq!(applied.len(), 1, "{trace}");

        let expected = BoolArray::from_iter(values.map(|value| members.contains(&value)));
        assert_arrays_eq!(traced.output, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_float_falls_back_through_expression() -> VortexResult<()> {
        let mut ctx = crate::array_session().create_execution_ctx();
        let values = [1.5f32, 2.5, 3.5];
        let element = PrimitiveArray::from_iter(values);
        let members = [1.5f32, 3.5];
        let list = ConstantArray::new(
            Scalar::list(
                Arc::new(DType::Primitive(F32, Nullability::NonNullable)),
                members.into_iter().map(Scalar::from).collect(),
                Nullability::NonNullable,
            ),
            element.len(),
        )
        .into_array();
        let list_scalar = list.as_constant().vortex_expect("list is constant");

        let actual = element
            .into_array()
            .apply(&list_contains(lit(list_scalar), root()))?
            .execute::<BoolArray>(&mut ctx)?;
        let expected = BoolArray::from_iter(values.map(|value| members.contains(&value)));

        assert_arrays_eq!(actual, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_null_needles() -> VortexResult<()> {
        let mut ctx = crate::array_session().create_execution_ctx();
        let element = PrimitiveArray::from_option_iter([Some(1), None, Some(2)]);
        let expected = BoolArray::from_iter([Some(true), None, Some(false)]);

        let actual = <Primitive as ListContainsElementKernel>::list_contains(
            &list([1, 3], element.len()),
            element.as_view(),
            &mut ctx,
        )?
        .vortex_expect("integer constant-list membership is supported");

        assert_arrays_eq!(actual, expected, &mut ctx);
        Ok(())
    }

    #[rstest]
    #[case::null_list(true)]
    #[case::empty_list(false)]
    fn test_constant_list_adaptor(#[case] null_list: bool) -> VortexResult<()> {
        let member_dtype = DType::Primitive(I32, Nullability::NonNullable);
        let list = if null_list {
            Scalar::null(DType::List(Arc::new(member_dtype), Nullability::Nullable))
        } else {
            Scalar::list(Arc::new(member_dtype), vec![], Nullability::NonNullable)
        };
        let needles = PrimitiveArray::from_option_iter([Some(1i32), None, Some(3)]).into_array();

        let mut ctx = crate::array_session().create_execution_ctx();
        let contains = needles
            .apply(&list_contains(lit(list), root()))?
            .execute::<ArrayRef>(&mut ctx)?;
        let expected = if null_list {
            BoolArray::from_iter([None, None, None])
        } else {
            BoolArray::from_iter([Some(false), Some(false), Some(false)])
        };

        assert!(contains.is::<Constant>());
        assert_arrays_eq!(contains, expected, &mut ctx);
        Ok(())
    }

    #[rstest]
    #[case::mixed(
        vec![Some(1), None, Some(3)],
        [Some(true), None, Some(true)]
    )]
    #[case::all_null(vec![None, None], [Some(false), None, Some(false)])]
    fn test_nullable_members(
        #[case] members: Vec<Option<i32>>,
        #[case] expected: [Option<bool>; 3],
    ) -> VortexResult<()> {
        let mut ctx = crate::array_session().create_execution_ctx();
        let member_dtype = DType::Primitive(I32, Nullability::Nullable);
        let list = ConstantArray::new(
            Scalar::list(
                Arc::new(member_dtype.clone()),
                members
                    .into_iter()
                    .map(|member| {
                        member
                            .map(|value| Scalar::primitive(value, Nullability::Nullable))
                            .unwrap_or_else(|| Scalar::null(member_dtype.clone()))
                    })
                    .collect(),
                Nullability::NonNullable,
            ),
            3,
        )
        .into_array();
        let element = PrimitiveArray::from_option_iter([Some(1), None, Some(3)]);

        let actual = <Primitive as ListContainsElementKernel>::list_contains(
            &list,
            element.as_view(),
            &mut ctx,
        )?
        .vortex_expect("integer constant-list membership is supported");

        assert_arrays_eq!(actual, BoolArray::from_iter(expected), &mut ctx);
        Ok(())
    }
}
