// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::marker::PhantomData;

use vortex_error::VortexResult;

use crate::traversal::{Node, NodeExt, NodeVisitor, TraversalOrder};

struct FnVisitor<'a, F, T: 'a>
where
    F: FnMut(&'a T) -> VortexResult<TraversalOrder>,
{
    f_down: Option<F>,
    f_up: Option<F>,
    _data: PhantomData<&'a T>,
}

impl<'a, T, F> NodeVisitor<'a> for FnVisitor<'a, F, T>
where
    F: FnMut(&'a T) -> VortexResult<TraversalOrder>,
    T: NodeExt,
{
    type NodeTy = T;

    fn visit_down(&mut self, node: &'a T) -> VortexResult<TraversalOrder> {
        if let Some(f) = self.f_down.as_mut() {
            f(node)
        } else {
            Ok(TraversalOrder::Continue)
        }
    }

    fn visit_up(&mut self, node: &'a T) -> VortexResult<TraversalOrder> {
        if let Some(f) = self.f_up.as_mut() {
            f(node)
        } else {
            Ok(TraversalOrder::Continue)
        }
    }
}

/// Traverse a [`Node`]-based tree using a closure. It will do it by walking the tree from the bottom going up.
pub fn pre_order_visit_up<'a, T: 'a + Node>(
    tree: &'a T,
    f: impl FnMut(&'a T) -> VortexResult<TraversalOrder>,
) -> VortexResult<()> {
    let mut visitor = FnVisitor {
        f_down: None,
        f_up: Some(f),
        _data: PhantomData,
    };

    tree.accept(&mut visitor)?;

    Ok(())
}

/// Traverse a [`Node`]-based tree using a closure. It will do it by walking the tree from the top going down.
pub fn pre_order_visit_down<'a, T: 'a + Node>(
    tree: &'a T,
    f: impl FnMut(&'a T) -> VortexResult<TraversalOrder>,
) -> VortexResult<()> {
    let mut visitor = FnVisitor {
        f_down: Some(f),
        f_up: None,
        _data: PhantomData,
    };

    tree.accept(&mut visitor)?;

    Ok(())
}
