// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::sync::Arc;

use itertools::Itertools;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::Canonical;
use crate::DynArray;
use crate::IntoArray;
use crate::arrays::Constant;
use crate::arrays::ConstantArray;
use crate::arrays::Filter;
use crate::arrays::FilterArray;
use crate::arrays::ScalarFnArray;
use crate::arrays::ScalarFnVTable;
use crate::arrays::Slice;
use crate::arrays::SliceArray;
use crate::arrays::StructArray;
use crate::dtype::DType;
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

pub(super) const RULES: ReduceRuleSet<ScalarFnVTable> = ReduceRuleSet::new(&[
    &ScalarFnPackToStructRule,
    &ScalarFnConstantRule,
    &ScalarFnAbstractReduceRule,
]);

pub(super) const PARENT_RULES: ParentRuleSet<ScalarFnVTable> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&ScalarFnUnaryFilterPushDownRule),
    ParentRuleSet::lift(&ScalarFnSliceReduceRule),
]);

/// Converts a ScalarFnArray with Pack into a StructArray directly.
#[derive(Debug)]
struct ScalarFnPackToStructRule;
impl ArrayReduceRule<ScalarFnVTable> for ScalarFnPackToStructRule {
    fn reduce(&self, array: &ScalarFnArray) -> VortexResult<Option<ArrayRef>> {
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
        if !array.children.iter().all(|c| c.is::<Constant>()) {
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
struct ScalarFnSliceReduceRule;
impl ArrayParentReduceRule<ScalarFnVTable> for ScalarFnSliceReduceRule {
    type Parent = Slice;

    fn reduce_parent(
        &self,
        array: &ScalarFnArray,
        parent: &SliceArray,
        _child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        let range = parent.slice_range();

        let children: Vec<_> = array
            .children()
            .iter()
            .map(|c| c.slice(range.clone()))
            .collect::<VortexResult<_>>()?;

        Ok(Some(
            ScalarFnArray {
                vtable: array.vtable.clone(),
                dtype: array.dtype.clone(),
                len: range.len(),
                children,
                stats: Default::default(),
            }
            .into_array(),
        ))
    }
}

#[derive(Debug)]
struct ScalarFnAbstractReduceRule;
impl ArrayReduceRule<ScalarFnVTable> for ScalarFnAbstractReduceRule {
    fn reduce(&self, array: &ScalarFnArray) -> VortexResult<Option<ArrayRef>> {
        if let Some(reduced) = array
            .scalar_fn()
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
    fn scalar_fn(&self) -> Option<&ScalarFnRef> {
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
        self.as_ref().node_dtype()
    }

    fn scalar_fn(&self) -> Option<&ScalarFnRef> {
        self.as_ref().scalar_fn()
    }

    fn child(&self, idx: usize) -> ReduceNodeRef {
        self.as_ref().child(idx)
    }

    fn child_count(&self) -> usize {
        self.as_ref().child_count()
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
    type Parent = Filter;

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
            .filter(|c| !c.is::<Constant>())
            .count()
            == 1
        {
            let new_children: Vec<_> = child
                .children
                .iter()
                .map(|c| match c.as_opt::<Constant>() {
                    Some(array) => {
                        Ok(ConstantArray::new(array.scalar().clone(), parent.len()).into_array())
                    }
                    None => c.filter(parent.filter_mask().clone()),
                })
                .try_collect()?;

            let new_array =
                ScalarFnArray::try_new(child.scalar_fn().clone(), new_children, parent.len())?
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
