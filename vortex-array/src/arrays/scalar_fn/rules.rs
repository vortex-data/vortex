// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Constant;
use crate::arrays::ConstantArray;
use crate::arrays::Filter;
use crate::arrays::ScalarFn;
use crate::arrays::ScalarFnArray;
use crate::arrays::Slice;
use crate::arrays::StructArray;
use crate::arrays::filter::prepare_mask_for_reuse;
use crate::arrays::scalar_fn::ScalarFnArrayExt;
use crate::optimizer::rules::ArrayParentReduceRule;
use crate::optimizer::rules::ArrayReduceRule;
use crate::optimizer::rules::ParentRuleSet;
use crate::optimizer::rules::ReduceRuleSet;
use crate::scalar_fn::ArrayReduceNode;
use crate::scalar_fn::fns::pack::Pack;
use crate::validity::Validity;

pub(super) const RULES: ReduceRuleSet<ScalarFn> =
    ReduceRuleSet::new(&[&ScalarFnPackToStructRule, &ScalarFnAbstractReduceRule]);

pub(super) const PARENT_RULES: ParentRuleSet<ScalarFn> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&ScalarFilterPushdownRule),
    ParentRuleSet::lift(&ScalarFnSliceReduceRule),
]);

/// Converts a ScalarFnArray with Pack into a StructArray directly.
#[derive(Debug)]
struct ScalarFnPackToStructRule;
impl ArrayReduceRule<ScalarFn> for ScalarFnPackToStructRule {
    fn reduce(&self, array: ArrayView<'_, ScalarFn>) -> VortexResult<Option<ArrayRef>> {
        let Some(pack_options) = array.scalar_fn().as_opt::<Pack>() else {
            return Ok(None);
        };

        let validity = match pack_options.nullability {
            crate::dtype::Nullability::NonNullable => Validity::NonNullable,
            crate::dtype::Nullability::Nullable => Validity::AllValid,
        };

        Ok(Some(
            StructArray::try_new(
                pack_options.names.clone(),
                array.children(),
                array.len(),
                validity,
            )?
            .into_array(),
        ))
    }
}

#[derive(Debug)]
struct ScalarFnSliceReduceRule;
impl ArrayParentReduceRule<ScalarFn> for ScalarFnSliceReduceRule {
    type Parent = Slice;

    fn reduce_parent(
        &self,
        array: ArrayView<'_, ScalarFn>,
        parent: ArrayView<'_, Slice>,
        _child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        let range = parent.slice_range();

        let children: Vec<_> = array
            .iter_children()
            .map(|c| c.slice(range.clone()))
            .collect::<VortexResult<_>>()?;

        Ok(Some(
            ScalarFnArray::try_new_with_len(array.scalar_fn().clone(), children, range.len())?
                .into_array(),
        ))
    }
}

#[derive(Debug)]
struct ScalarFnAbstractReduceRule;
impl ArrayReduceRule<ScalarFn> for ScalarFnAbstractReduceRule {
    fn reduce(&self, array: ArrayView<'_, ScalarFn>) -> VortexResult<Option<ArrayRef>> {
        let node = ArrayReduceNode::new(array.as_ref());
        if let Some(reduced) = array.scalar_fn().reduce_array(&node)? {
            return Ok(Some(reduced.into_array()));
        }
        Ok(None)
    }
}

#[derive(Debug)]
struct ScalarFilterPushdownRule;

impl ArrayParentReduceRule<ScalarFn> for ScalarFilterPushdownRule {
    type Parent = Filter;

    fn reduce_parent(
        &self,
        child: ArrayView<'_, ScalarFn>,
        parent: ArrayView<'_, Filter>,
        _child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        let nchildren = child
            .iter_children()
            .filter(|c| !c.is::<Constant>())
            .count();
        if nchildren > 1
            && let Some(values) = parent.filter_mask().values()
        {
            prepare_mask_for_reuse(values, nchildren);
        }

        let new_children: Vec<_> = child
            .iter_children()
            .map(|c| match c.as_opt::<Constant>() {
                Some(array) => {
                    Ok(ConstantArray::new(array.scalar().clone(), parent.len()).into_array())
                }
                None => c.filter(parent.filter_mask().clone()),
            })
            .try_collect()?;

        Ok(Some(
            ScalarFnArray::try_new_with_len(child.scalar_fn().clone(), new_children, parent.len())?
                .into_array(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;
    use vortex_mask::Mask;

    use super::ScalarFilterPushdownRule;
    use crate::VortexSessionExecute;
    use crate::array::IntoArray;
    use crate::array_session;
    use crate::arrays::ChunkedArray;
    use crate::arrays::Constant;
    use crate::arrays::Filter;
    use crate::arrays::FilterArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::ScalarFn;
    use crate::arrays::ScalarFnArray;
    use crate::arrays::scalar_fn::ScalarFnArrayExt;
    use crate::arrays::scalar_fn::rules::ConstantArray;
    use crate::assert_arrays_eq;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::expr::cast;
    use crate::expr::is_null;
    use crate::expr::root;
    use crate::optimizer::rules::ArrayParentReduceRule;
    use crate::scalar::Scalar;
    use crate::scalar_fn::TypedScalarFnInstance;
    use crate::scalar_fn::fns::binary::Binary;
    use crate::scalar_fn::fns::literal::Literal;
    use crate::scalar_fn::fns::operators::Operator;

    #[rstest]
    #[case(0, [14, 14, 14, 14, 14])]
    #[case(1, [7, 27, 47, 67, 87])]
    #[case(2, [0, 40, 80, 120, 160])]
    fn test_filter_pushdown(
        #[case] nchildren: usize,
        #[case] expected: [i32; 5],
    ) -> VortexResult<()> {
        let children = (0..2)
            .map(|index| {
                if index < nchildren {
                    PrimitiveArray::from_iter(0i32..100).into_array()
                } else {
                    ConstantArray::new(7i32, 100).into_array()
                }
            })
            .collect();
        let array = ScalarFnArray::try_new(
            TypedScalarFnInstance::new(Binary, Operator::Add).erased(),
            children,
        )?;
        let mask = Mask::from_iter((0..100).map(|index| index % 20 == 0));
        let values = mask
            .values()
            .ok_or_else(|| vortex_err!("expected mask values"))?;
        assert!(values.cached_indices().is_none());
        let parent = FilterArray::try_new(array.clone().into_array(), mask.clone())?;

        let result = ScalarFilterPushdownRule
            .reduce_parent(array.as_view(), parent.as_view(), 0)?
            .ok_or_else(|| vortex_err!("expected filter pushdown"))?;

        assert_eq!(values.cached_indices().is_some(), nchildren > 1);
        let scalar_fn = result.as_::<ScalarFn>();
        assert_eq!(scalar_fn.len(), 5);
        for (index, child) in scalar_fn.iter_children().enumerate() {
            assert_eq!(child.len(), 5);
            if index < nchildren {
                assert!(child.is::<Filter>());
            } else {
                assert!(child.is::<Constant>());
            }
        }
        let expected = PrimitiveArray::from_iter(expected);
        assert_arrays_eq!(
            result,
            expected,
            &mut array_session().create_execution_ctx()
        );
        Ok(())
    }

    #[test]
    fn test_filter_pushdown_without_children() -> VortexResult<()> {
        let array = ScalarFnArray::try_new_with_len(
            TypedScalarFnInstance::new(Literal, Scalar::from(7i32)).erased(),
            vec![],
            3,
        )?;
        let parent = FilterArray::try_new(
            array.clone().into_array(),
            Mask::from_iter([true, false, true]),
        )?;

        let result = ScalarFilterPushdownRule
            .reduce_parent(array.as_view(), parent.as_view(), 0)?
            .ok_or_else(|| vortex_err!("expected filter pushdown"))?;

        assert_eq!(result.as_::<ScalarFn>().nchildren(), 0);
        assert_eq!(result.len(), 2);
        assert_arrays_eq!(
            result,
            ConstantArray::new(7i32, 2),
            &mut array_session().create_execution_ctx()
        );
        Ok(())
    }

    #[test]
    fn test_empty_constants() {
        let array = ChunkedArray::try_new(
            vec![
                ConstantArray::new(Some(1u64), 0).into_array(),
                PrimitiveArray::from_iter(vec![2u64])
                    .into_array()
                    .apply(&cast(
                        root(),
                        DType::Primitive(PType::U64, Nullability::Nullable),
                    ))
                    .vortex_expect("casted"),
            ],
            DType::Primitive(PType::U64, Nullability::Nullable),
        )
        .vortex_expect("construction")
        .into_array();

        let expr = is_null(root());
        array.apply(&expr).vortex_expect("expr evaluation");
    }
}
