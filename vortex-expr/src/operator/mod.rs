// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod reduce;

pub use reduce::*;
use vortex_array::pipeline::OperatorRef;
use vortex_error::VortexResult;

use crate::traversal::{FoldUp, NodeFolder};
use crate::{ExprRef, RootVTable};

pub struct ExprOperatorConverter {
    root: OperatorRef,
}

impl ExprOperatorConverter {
    pub fn new(root: OperatorRef) -> Self {
        Self { root }
    }
}

// Needs a mapping from Root array to encoding -> Operator

impl NodeFolder for ExprOperatorConverter {
    type NodeTy = ExprRef;
    type Result = Option<OperatorRef>;

    fn visit_up(
        &mut self,
        node: ExprRef,
        children: Vec<Option<OperatorRef>>,
    ) -> VortexResult<FoldUp<Option<OperatorRef>>> {
        let Some(children) = children.into_iter().collect::<Option<Vec<_>>>() else {
            return Ok(FoldUp::Stop(None));
        };
        if node.as_opt::<RootVTable>().is_some() {
            return Ok(FoldUp::Continue(Some(self.root.clone())));
        }
        let Some(operator) = node.operator(children) else {
            return Ok(FoldUp::Stop(None));
        };
        Ok(FoldUp::Continue(Some(operator)))
    }
}
