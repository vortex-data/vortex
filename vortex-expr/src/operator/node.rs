// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use vortex_array::pipeline::{Operator, OperatorRef};
use vortex_error::VortexResult;

use crate::traversal::{Node, NodeContainer, Transformed, TraversalOrder};

impl<'a> NodeContainer<'a, Self> for OperatorRef {
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

impl Node for OperatorRef {
    fn apply_children<'a, F: FnMut(&'a Self) -> VortexResult<TraversalOrder>>(
        &'a self,
        _f: F,
    ) -> VortexResult<TraversalOrder> {
        todo!()
    }

    fn map_children<F: FnMut(Self) -> VortexResult<Transformed<Self>>>(
        self,
        f: F,
    ) -> VortexResult<Transformed<Self>> {
        let transformed = self
            .children()
            .iter()
            .cloned()
            .collect_vec()
            .map_elements(f)?;

        if transformed.changed {
            Ok(Transformed {
                value: self.with_children(transformed.value),
                order: transformed.order,
                changed: true,
            })
        } else {
            Ok(Transformed::no(self))
        }
    }

    fn iter_children<T>(&self, f: impl FnOnce(&mut dyn Iterator<Item = &Self>) -> T) -> T {
        f(&mut self.children().iter())
    }

    fn children_count(&self) -> usize {
        self.children().len()
    }
}
