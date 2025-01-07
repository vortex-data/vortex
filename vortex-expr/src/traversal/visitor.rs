use std::marker::PhantomData;

use vortex_error::VortexResult;

use crate::traversal::{Node, NodeVisitor, TraversalOrder};

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
    T: Node,
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

pub fn pre_order_visit_up<'a, T: 'a + Node>(
    f: impl FnMut(&'a T) -> VortexResult<TraversalOrder>,
) -> impl NodeVisitor<'a, NodeTy = T> {
    FnVisitor {
        f_down: None,
        f_up: Some(f),
        _data: Default::default(),
    }
}

pub fn pre_order_visit_down<'a, T: 'a + Node>(
    f: impl FnMut(&'a T) -> VortexResult<TraversalOrder>,
) -> impl NodeVisitor<'a, NodeTy = T> {
    FnVisitor {
        f_down: Some(f),
        f_up: None,
        _data: Default::default(),
    }
}
