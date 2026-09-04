// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Constant;
use crate::arrays::ConstantArray;
use crate::arrays::Filter;
use crate::arrays::FilterArray;
use crate::arrays::ScalarFn;
use crate::arrays::ScalarFnArray;
use crate::arrays::Slice;
use crate::arrays::StructArray;
use crate::arrays::scalar_fn::ScalarFnArrayExt;
use crate::kernel::ExecuteParentKernel;
use crate::optimizer::kernels::execute_parent_key;
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
    ParentRuleSet::lift(&ScalarFnUnaryFilterPushDownRule),
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
struct ScalarFnUnaryFilterPushDownRule;

impl ArrayParentReduceRule<ScalarFn> for ScalarFnUnaryFilterPushDownRule {
    type Parent = Filter;

    fn reduce_parent(
        &self,
        child: ArrayView<'_, ScalarFn>,
        parent: ArrayView<'_, Filter>,
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

            let new_array =
                ScalarFnArray::try_new(child.scalar_fn().clone(), new_children)?.into_array();

            return Ok(Some(new_array));
        }

        Ok(None)
    }
}

#[derive(Debug)]
struct FilterScalarFnUnaryPushDownRule;

impl ExecuteParentKernel<Filter> for FilterScalarFnUnaryPushDownRule {
    type Parent = ScalarFn;

    fn execute_parent(
        &self,
        child: ArrayView<'_, Filter>,
        parent: ArrayView<'_, ScalarFn>,
        _child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        // If we have one non-constant child, and ScalarFn has a registered
        // kernel for given encoding id (e.g. can operate on compressed data),
        // it's faster to pull Filter up so that it doesn't canonicalize its
        // child.
        //
        // Fn(Filter(x), consts) -> Filter(Fn(x, consts))
        //
        // Example: clickbench q1, SELECT * FROM hits WHERE AdvEngineID <> 0;
        // AdvEngineID <> 0 is Binary over Sparse, but FlatReader applies
        // Filter over Sparse, so we get Binary(Filter(Sparse)). Filter(Sparse)
        // canonicalizes.
        let mut non_const_child_id: usize = usize::MAX;
        for (i, child) in parent.iter_children().enumerate() {
            if child.is::<Constant>() {
                continue;
            }
            if non_const_child_id != usize::MAX {
                return Ok(None);
            }
            non_const_child_id = i;
        }

        let new_non_const_grandchild = parent.child_at(non_const_child_id);

        let key = execute_parent_key(
            parent.scalar_fn.id(),
            new_non_const_grandchild.encoding_id(),
        );
        if !ctx.execute_parent_kernels.contains_key(&key) {
            return Ok(None);
        };

        let new_grandchildren: Vec<_> = parent
            .iter_children()
            .map(|child| match child.as_constant() {
                Some(scalar) => ConstantArray::new(scalar, parent.len()).into_array(),
                None => new_non_const_grandchild.clone(),
            })
            .collect();

        let new_child = ScalarFnArray::try_new(parent.scalar_fn().clone(), new_grandchildren)?;
        let mask = child.filter_mask().clone();
        let new_parent = FilterArray::try_new(new_child.into_array(), mask)?;
        Ok(Some(new_parent.into_array()))
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
