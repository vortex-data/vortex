// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_array::pipeline::operators::Operator;
use vortex_error::VortexResult;

use crate::traversal::{FoldUp, Node, NodeExt, NodeFolder, NodeRewriter, Transformed};

pub fn reduce_operator(operator: Arc<dyn Operator>) -> VortexResult<Arc<dyn Operator>> {
    let operator = reduce_up(operator.clone())?;
    reduce_down(operator)
}

pub fn reduce_up(operator: Arc<dyn Operator>) -> VortexResult<Arc<dyn Operator>> {
    let mut folder = UpReducer;
    operator.fold(&mut folder).map(|t| t.value())
}

pub fn reduce_down(operator: Arc<dyn Operator>) -> VortexResult<Arc<dyn Operator>> {
    let mut rewriter = DownReducer;
    operator.rewrite(&mut rewriter).map(|t| t.value)
}

struct UpReducer;

impl NodeFolder for UpReducer {
    type NodeTy = Arc<dyn Operator>;
    type Result = Arc<dyn Operator>;

    fn visit_up(
        &mut self,
        node: Self::NodeTy,
        children: Vec<Self::Result>,
    ) -> VortexResult<FoldUp<Self::Result>> {
        Ok(FoldUp::Continue(
            match node.reduce_children(children.as_slice()) {
                None => node.with_children(children),
                Some(r) => r,
            },
        ))
    }
}

struct DownReducer;

impl NodeRewriter for DownReducer {
    type NodeTy = Arc<dyn Operator>;

    fn visit_down(&mut self, node: Self::NodeTy) -> VortexResult<Transformed<Self::NodeTy>> {
        if node.children_count() != 1 {
            return Ok(Transformed::no(node));
        }
        match node.children()[0].reduce_parent(node.clone()) {
            None => Ok(Transformed::no(node)),
            Some(r) => Ok(Transformed::yes(r)),
        }
    }
}
