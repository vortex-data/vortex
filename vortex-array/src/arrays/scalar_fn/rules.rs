// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::sync::Arc;

use itertools::Itertools;
use vortex_dtype::DType;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::Array;
use crate::ArrayRef;
use crate::ArrayVisitor;
use crate::Canonical;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::ConstantVTable;
use crate::arrays::FilterArray;
use crate::arrays::FilterVTable;
use crate::arrays::ScalarFnArray;
use crate::arrays::ScalarFnVTable;
use crate::arrays::SliceReduceAdaptor;
use crate::arrays::StructArray;
use crate::expr::Pack;
use crate::expr::ReduceCtx;
use crate::expr::ReduceNode;
use crate::expr::ReduceNodeRef;
use crate::expr::ScalarFn;
use crate::optimizer::rules::ArrayParentReduceRule;
use crate::optimizer::rules::ArrayReduceRule;
use crate::optimizer::rules::ParentRuleSet;
use crate::optimizer::rules::ReduceRuleSet;
use crate::validity::Validity;

pub(super) const RULES: ReduceRuleSet<ScalarFnVTable> = ReduceRuleSet::new(&[
    &ScalarFnPackToStructRule,
    &ScalarFnConstantRule,
    &ScalarFnAbstractReduceRule,
]);

pub(super) const PARENT_RULES: ParentRuleSet<ScalarFnVTable> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&ScalarFnUnaryFilterPushDownRule),
    ParentRuleSet::lift(&SliceReduceAdaptor(ScalarFnVTable)),
]);

/// Converts a ScalarFnArray with Pack into a StructArray directly.
#[derive(Debug)]
struct ScalarFnPackToStructRule;
impl ArrayReduceRule<ScalarFnVTable> for ScalarFnPackToStructRule {
    fn reduce(&self, array: &ScalarFnArray) -> VortexResult<Option<ArrayRef>> {
        let Some(pack_options) = array.scalar_fn.as_opt::<Pack>() else {
            return Ok(None);
        };

        let validity = match pack_options.nullability {
            vortex_dtype::Nullability::NonNullable => Validity::NonNullable,
            vortex_dtype::Nullability::Nullable => Validity::AllValid,
        };

        Ok(Some(
            StructArray::try_new(
                pack_options.names.clone(),
                array.children.clone(),
                array.len,
                validity,
            )?
            .into_array(),
        ))
    }
}

#[derive(Debug)]
struct ScalarFnConstantRule;
impl ArrayReduceRule<ScalarFnVTable> for ScalarFnConstantRule {
    fn reduce(&self, array: &ScalarFnArray) -> VortexResult<Option<ArrayRef>> {
        if !array.children.iter().all(|c| c.is::<ConstantVTable>()) {
            return Ok(None);
        }
        if array.is_empty() {
            Ok(Some(Canonical::empty(array.dtype()).into_array()))
        } else {
            let result = array.scalar_at(0)?;
            Ok(Some(ConstantArray::new(result, array.len).into_array()))
        }
    }
}

#[derive(Debug)]
struct ScalarFnAbstractReduceRule;
impl ArrayReduceRule<ScalarFnVTable> for ScalarFnAbstractReduceRule {
    fn reduce(&self, array: &ScalarFnArray) -> VortexResult<Option<ArrayRef>> {
        if let Some(reduced) = array
            .scalar_fn
            .reduce(array, &ArrayReduceCtx { len: array.len })?
        {
            return Ok(Some(
                reduced
                    .as_any()
                    .downcast_ref::<ArrayRef>()
                    .vortex_expect("ReduceNode is not an ArrayRef")
                    .clone(),
            ));
        }
        Ok(None)
    }
}

impl ReduceNode for ScalarFnArray {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn node_dtype(&self) -> VortexResult<DType> {
        Ok(self.dtype().clone())
    }

    #[allow(clippy::same_name_method)]
    fn scalar_fn(&self) -> Option<&ScalarFn> {
        Some(ScalarFnArray::scalar_fn(self))
    }

    fn child(&self, idx: usize) -> ReduceNodeRef {
        Arc::new(self.children()[idx].clone())
    }

    fn child_count(&self) -> usize {
        self.children.len()
    }
}

impl ReduceNode for ArrayRef {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn node_dtype(&self) -> VortexResult<DType> {
        Ok(self.as_ref().dtype().clone())
    }

    fn scalar_fn(&self) -> Option<&ScalarFn> {
        self.as_opt::<ScalarFnVTable>().map(|a| a.scalar_fn())
    }

    fn child(&self, idx: usize) -> ReduceNodeRef {
        Arc::new(
            self.nth_child(idx)
                .vortex_expect("child index out of bounds"),
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
        scalar_fn: ScalarFn,
        children: &[ReduceNodeRef],
    ) -> VortexResult<ReduceNodeRef> {
        Ok(Arc::new(
            ScalarFnArray::try_new(
                scalar_fn,
                children
                    .iter()
                    .map(|c| {
                        c.as_any()
                            .downcast_ref::<ArrayRef>()
                            .vortex_expect("ReduceNode is not an ArrayRef")
                            .clone()
                    })
                    .collect(),
                self.len,
            )?
            .into_array(),
        ))
    }
}

#[derive(Debug)]
struct ScalarFnUnaryFilterPushDownRule;

impl ArrayParentReduceRule<ScalarFnVTable> for ScalarFnUnaryFilterPushDownRule {
    type Parent = FilterVTable;

    fn reduce_parent(
        &self,
        child: &ScalarFnArray,
        parent: &FilterArray,
        _child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        // If we only have one non-constant child, then it is _always_ cheaper to push down the
        // filter over the children of the scalar function array.
        if child
            .children
            .iter()
            .filter(|c| !c.is::<ConstantVTable>())
            .count()
            == 1
        {
            let new_children: Vec<_> = child
                .children
                .iter()
                .map(|c| match c.as_opt::<ConstantVTable>() {
                    Some(array) => {
                        Ok(ConstantArray::new(array.scalar().clone(), parent.len()).into_array())
                    }
                    None => c.filter(parent.filter_mask().clone()),
                })
                .try_collect()?;

            let new_array =
                ScalarFnArray::try_new(child.scalar_fn.clone(), new_children, parent.len())?
                    .into_array();

            return Ok(Some(new_array));
        }

        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use vortex_dtype::DType;
    use vortex_dtype::Nullability;
    use vortex_dtype::PType;
    use vortex_error::VortexExpect;

    use crate::array::IntoArray;
    use crate::arrays::ChunkedArray;
    use crate::arrays::ConstantArray;
    use crate::arrays::PrimitiveArray;
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
        .to_array();

        let expr = is_null(root());
        array.apply(&expr).vortex_expect("expr evaluation");
    }
}
