// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Array tree traversal using the Node trait infrastructure.
//!
//! This module provides implementations of [`Node`] and [`NodeContainer`] for [`ArrayRef`],
//! enabling powerful tree transformations on array structures similar to expression trees.

use itertools::Itertools;
use vortex_error::VortexResult;

use crate::array::ArrayVisitor;
use crate::expr::traversal::{Node, NodeContainer, Transformed, TraversalOrder};
use crate::{Array, ArrayRef};

impl<'a> NodeContainer<'a, Self> for ArrayRef {
    fn apply_elements<F: FnMut(&'a Self) -> VortexResult<TraversalOrder>>(
        &'a self,
        mut f: F,
    ) -> VortexResult<TraversalOrder> {
        f(self)
    }

    fn map_elements<F: FnMut(Self) -> VortexResult<Transformed<Self>>>(
        self,
        mut f: F,
    ) -> VortexResult<Transformed<Self>> {
        f(self)
    }
}

impl Node for ArrayRef {
    type Child = dyn Array; // Back to dyn Array to match children() return type

    fn apply_children<'a, F: FnMut(&'a Self::Child) -> VortexResult<TraversalOrder>>(
        &'a self,
        mut f: F,
    ) -> VortexResult<TraversalOrder> {
        for child in self.children().iter() {
            let order = f(*child)?; // *child is &'a dyn Array
            if !matches!(order, TraversalOrder::Continue | TraversalOrder::Skip) {
                return Ok(order);
            }
        }

        Ok(TraversalOrder::Continue)
    }

    fn map_children<F: FnMut(Self) -> VortexResult<Transformed<Self>>>(
        self,
        f: F,
    ) -> VortexResult<Transformed<Self>> {
        let transformed = self
            .children()
            .iter()
            .map(|c| c.to_array())
            .collect_vec()
            .map_elements(f)?;

        if transformed.changed {
            Ok(Transformed {
                value: self.with_children(transformed.value.as_ref())?,
                order: transformed.order,
                changed: true,
            })
        } else {
            Ok(Transformed::no(self))
        }
    }

    fn iter_children<T>(&self, f: impl FnOnce(&mut dyn Iterator<Item = &Self>) -> T) -> T {
        // Convert Vec<&dyn Array> to Vec<ArrayRef> so we can create Iterator<Item = &ArrayRef>
        let children: Vec<ArrayRef> = self.children().iter().map(|c| c.to_array()).collect();
        f(&mut children.iter())
    }

    fn children_count(&self) -> usize {
        self.nchildren()
    }

    /// For ArrayRef, manually implement the traversal to avoid visitor type mismatch
    fn accept_on_child<'a, V: crate::expr::traversal::NodeVisitor<'a, NodeTy = Self>>(
        child: &'a Self::Child,
        visitor: &mut V,
    ) -> VortexResult<TraversalOrder> {
        use crate::expr::traversal::TraversalOrder;

        // Manually implement the traversal logic here:
        // 1. Visit down on the child
        let down_order = visitor.visit_down(child)?;

        // 2. If we should continue, recurse through the child's children
        let child_order = down_order.visit_children(|| {
            for grandchild in child.children().iter() {
                let order = Self::accept_on_child(*grandchild, visitor)?;
                if !matches!(order, TraversalOrder::Continue | TraversalOrder::Skip) {
                    return Ok(order);
                }
            }
            Ok(TraversalOrder::Continue)
        })?;

        // 3. Visit up on the child
        child_order.visit_parent(|| visitor.visit_up(child))
    }
}
