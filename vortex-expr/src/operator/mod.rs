mod reduce;

use std::sync::Arc;

pub use reduce::*;
use vortex_array::pipeline::operators::Operator;
use vortex_array::{Array, ArrayRef};
use vortex_error::{VortexResult, vortex_err};

use crate::traversal::{FoldUp, NodeFolder};
use crate::{ExprRef, RootVTable};

pub struct ExprOperatorConverter {
    root: ArrayRef,
}

impl ExprOperatorConverter {
    pub fn new(root: ArrayRef) -> Self {
        Self { root }
    }
}

// Needs a mapping from Root array to encoding -> Operator

impl NodeFolder for ExprOperatorConverter {
    type NodeTy = ExprRef;
    type Result = Arc<dyn Operator>;

    fn visit_up(
        &mut self,
        node: ExprRef,
        children: Vec<Arc<dyn Operator>>,
    ) -> VortexResult<FoldUp<Arc<dyn Operator>>> {
        if node.as_opt::<RootVTable>().is_some() {
            let pipeline = self.root.to_pipeline_plan()?;
            return Ok(FoldUp::Continue(pipeline));
        }
        node.operator(children)
            .ok_or_else(|| vortex_err!("Failed to convert operator: {:?}", node))
            .map(FoldUp::Continue)
    }
}
