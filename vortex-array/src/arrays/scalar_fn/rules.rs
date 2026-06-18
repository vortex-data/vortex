// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use itertools::Itertools;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::array::ParentView;
use crate::arrays::Constant;
use crate::arrays::ConstantArray;
use crate::arrays::Filter;
use crate::arrays::ScalarFn;
use crate::arrays::ScalarFnArray;
use crate::arrays::Slice;
use crate::arrays::StructArray;
use crate::arrays::scalar_fn::ScalarFnArrayExt;
use crate::dtype::DType;
use crate::expr::ExpressionReduceNode;
use crate::optimizer::rules::ArrayParentReduceRule;
use crate::optimizer::rules::ArrayReduceRule;
use crate::optimizer::rules::ParentRuleSet;
use crate::optimizer::rules::ReduceRuleSet;
use crate::scalar_fn::ReduceCtx;
use crate::scalar_fn::ReduceNode;
use crate::scalar_fn::ReduceNodeRef;
use crate::scalar_fn::ScalarFnRef;
use crate::scalar_fn::fns::pack::Pack;
use crate::validity::Validity;

pub(super) const RULES: ReduceRuleSet<ScalarFn> =
    ReduceRuleSet::new(&[&ScalarFnPackToStructRule, &ScalarFnAbstractReduceRule]);

pub(super) const PARENT_RULES: ParentRuleSet<ScalarFn> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&ScalarFnUnaryFilterPushDownRule),
    ParentRuleSet::lift(&ScalarFnSliceReduceRule),
]);

/// Converts a ScalarFnArray with Pack into a StructArray directly.
#[derive(Debug)]
struct ScalarFnPackToStructRule;
impl ArrayReduceRule<ScalarFn> for ScalarFnPackToStructRule {
    fn reduce(&self, array: ParentView<'_, ScalarFn>) -> VortexResult<Option<ArrayRef>> {
        let Some(pack_options) = ScalarFnArrayExt::scalar_fn(&array).as_opt::<Pack>() else {
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
        parent: ParentView<'_, Slice>,
        _child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        let range = parent.slice_range();

        let children: Vec<_> = array
            .iter_children()
            .map(|c| c.slice(range.clone()))
            .collect::<VortexResult<_>>()?;

        Ok(Some(
            ScalarFnArray::try_new_with_len(
                ScalarFnArrayExt::scalar_fn(&array).clone(),
                children,
                range.len(),
            )?
            .into_array(),
        ))
    }
}

#[derive(Debug)]
struct ScalarFnAbstractReduceRule;
impl ArrayReduceRule<ScalarFn> for ScalarFnAbstractReduceRule {
    fn reduce(&self, array: ParentView<'_, ScalarFn>) -> VortexResult<Option<ArrayRef>> {
        if let Some(reduced) = ScalarFnArrayExt::scalar_fn(&array)
            .reduce(&array, &ArrayReduceCtx { len: array.len() })?
        {
            return Ok(Some(reduced.as_array()));
        }
        Ok(None)
    }
}

impl ReduceNode for ParentView<'_, ScalarFn> {
    fn as_array(&self) -> ArrayRef {
        self.materialize_array_ref().clone()
    }

    fn as_expression(&self) -> ExpressionReduceNode {
        vortex_panic!("Cannot convert ArrayView to ExpressionReduceNode")
    }

    fn node_dtype(&self) -> VortexResult<DType> {
        Ok(self.dtype().clone())
    }

    fn scalar_fn(&self) -> Option<&ScalarFnRef> {
        Some(ScalarFnArrayExt::scalar_fn(self))
    }

    fn child(&self, idx: usize) -> ReduceNodeRef {
        Arc::new(self.child_at(idx).clone())
    }

    fn child_count(&self) -> usize {
        ScalarFnArrayExt::nchildren(self)
    }
}

impl ReduceNode for ArrayRef {
    fn as_array(&self) -> ArrayRef {
        self.clone()
    }

    fn as_expression(&self) -> ExpressionReduceNode {
        vortex_panic!("Cannot convert ArrayRef to ExpressionReduceNode")
    }

    fn node_dtype(&self) -> VortexResult<DType> {
        Ok(self.dtype().clone())
    }

    fn scalar_fn(&self) -> Option<&ScalarFnRef> {
        self.as_opt::<ScalarFn>().map(|a| a.data().scalar_fn())
    }

    fn child(&self, idx: usize) -> ReduceNodeRef {
        Arc::new(
            self.nth_child(idx)
                .vortex_expect("child idx out of bounds")
                .clone(),
        )
    }

    fn child_count(&self) -> usize {
        self.nchildren()
    }
}

struct ArrayReduceCtx {
    // The length of the array being reduced
    len: usize,
}
impl ReduceCtx for ArrayReduceCtx {
    fn new_node(
        &self,
        scalar_fn: ScalarFnRef,
        children: &[ReduceNodeRef],
    ) -> VortexResult<ReduceNodeRef> {
        Ok(Arc::new(
            ScalarFnArray::try_new_with_len(
                scalar_fn,
                children.iter().map(|c| c.as_array()).collect(),
                self.len,
            )?
            .into_array(),
        ))
    }
}

#[derive(Debug)]
struct ScalarFnUnaryFilterPushDownRule;

impl ArrayParentReduceRule<ScalarFn> for ScalarFnUnaryFilterPushDownRule {
    type Parent = Filter;

    fn reduce_parent(
        &self,
        child: ArrayView<'_, ScalarFn>,
        parent: ParentView<'_, Filter>,
        _child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        // If we only have one non-constant child, then it is _always_ cheaper to push down the
        // filter over the children of the scalar function array.
        if child
            .iter_children()
            .filter(|c| !c.is::<Constant>())
            .count()
            == 1
        {
            let new_children: Vec<_> = child
                .iter_children()
                .map(|c| match c.as_opt::<Constant>() {
                    Some(array) => {
                        Ok(ConstantArray::new(array.scalar().clone(), parent.len()).into_array())
                    }
                    None => c.filter(parent.filter_mask().clone()),
                })
                .try_collect()?;

            let new_array = ScalarFnArray::try_new_with_len(
                ScalarFnArrayExt::scalar_fn(&child).clone(),
                new_children,
                parent.len(),
            )?
            .into_array();

            return Ok(Some(new_array));
        }

        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexExpect;

    use crate::array::IntoArray;
    use crate::arrays::ChunkedArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::scalar_fn::rules::ConstantArray;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::expr::cast;
    use crate::expr::is_null;
    use crate::expr::root;

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
