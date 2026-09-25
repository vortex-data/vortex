// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::marker::PhantomData;

use vortex_error::VortexResult;

use crate::expr::traversal::Node;
use crate::expr::traversal::NodeExt;
use crate::expr::traversal::NodeVisitor;
use crate::expr::traversal::TraversalOrder;

struct FnVisitor<'a, F, T: 'a>
where
    F: FnMut(&'a T) -> VortexResult<TraversalOrder>,
{
    f: F,
    _data: PhantomData<&'a T>,
}

impl<'a, T, F> NodeVisitor<'a> for FnVisitor<'a, F, T>
where
    F: FnMut(&'a T) -> VortexResult<TraversalOrder>,
    T: NodeExt,
{
    type NodeTy = T;

    fn visit_down(&mut self, node: &'a T) -> VortexResult<TraversalOrder> {
        (self.f)(node)
    }
}

/// Traverse a [`Node`]-based tree using a closure. It will do it by walking the tree from the top going down.
pub fn pre_order_visit_down<'a, T: 'a + Node>(
    tree: &'a T,
    f: impl FnMut(&'a T) -> VortexResult<TraversalOrder>,
) -> VortexResult<()> {
    let mut visitor = FnVisitor {
        f,
        _data: PhantomData,
    };

    tree.accept(&mut visitor)?;

    Ok(())
}
