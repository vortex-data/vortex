// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ArrayView;
use crate::ExecutionCtx;
use crate::arrays::Primitive;
use crate::dtype::DType;
use crate::match_each_integer_ptype;
use crate::scalar_fn::fns::list_contains::IntegerMembership;
use crate::scalar_fn::fns::list_contains::ListContainsElementKernel;

impl ListContainsElementKernel for Primitive {
    fn list_contains(
        list: &ArrayRef,
        element: ArrayView<'_, Self>,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(list_scalar) = list.as_constant() else {
            return Ok(None);
        };
        let DType::List(member_dtype, _) = list.dtype() else {
            return Ok(None);
        };
        if !member_dtype.eq_ignore_nullability(element.dtype()) || !element.ptype().is_int() {
            return Ok(None);
        }

        let nullability = list.dtype().nullability() | element.dtype().nullability();
        let Some(elements) = list_scalar.as_list().elements() else {
            return Ok(None);
        };
        if elements.is_empty() {
            return Ok(None);
        }

        let result = match_each_integer_ptype!(element.ptype(), |T| {
            let members = elements
                .iter()
                .map(|value| {
                    value
                        .as_primitive_opt()
                        .vortex_expect("list dtype was checked before member extraction")
                        .try_typed_value::<T>()
                })
                .collect::<VortexResult<Vec<Option<T>>>>()?
                .into_iter()
                .flatten()
                .collect::<Vec<_>>();

            IntegerMembership::new(members).evaluate_primitive(element, nullability)?
        });

        Ok(Some(result))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use rstest::rstest;

    use super::*;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::BoolArray;
    use crate::arrays::ConstantArray;
    use crate::arrays::PrimitiveArray;
    use crate::assert_arrays_eq;
    use crate::dtype::Nullability;
    use crate::dtype::PType::I32;
    #[cfg(not(codspeed))]
    use crate::expr::list_contains;
    #[cfg(not(codspeed))]
    use crate::expr::lit;
    #[cfg(not(codspeed))]
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
    #[case::dense((0..32).map(|value| value * 3).collect())]
    #[case::sparse((0..32).map(|value| value * 10_000).collect())]
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
        assert_eq!(applied.len(), 1, "{trace}");

        let expected = BoolArray::from_iter(values.map(|value| members.contains(&value)));
        assert_arrays_eq!(traced.output, expected, &mut ctx);
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
