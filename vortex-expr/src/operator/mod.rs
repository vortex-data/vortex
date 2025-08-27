// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod reduce;

use std::rc::Rc;

pub use reduce::*;
use vortex_array::Array;
use vortex_error::{VortexResult};
use vortex_array::pipeline::Operator;

use crate::traversal::{FoldUp, NodeFolder};
use crate::{ExprRef, RootVTable};

pub struct ExprOperatorConverter<'a> {
    root: &'a dyn Array,
}

impl<'a> ExprOperatorConverter<'a> {
    pub fn new(root: &'a dyn Array) -> Self {
        Self { root }
    }
}

// Needs a mapping from Root array to encoding -> Operator

impl<'a> NodeFolder for ExprOperatorConverter<'a> {
    type NodeTy = ExprRef;
    type Result = Option<Rc<dyn Operator>>;

    fn visit_up(
        &mut self,
        node: ExprRef,
        children: Vec<Option<Rc<dyn Operator>>>,
    ) -> VortexResult<FoldUp<Option<Rc<dyn Operator>>>> {
        let Some(children) = children.into_iter().collect::<Option<Vec<_>>>() else {
            return Ok(FoldUp::Stop(None));
        };
        if node.as_opt::<RootVTable>().is_some() {
            let Some(operator) = self.root.to_operator()? else {
                return Ok(FoldUp::Stop(None));
            };
            return Ok(FoldUp::Continue(Some(operator)));
        }
        let Some(operator) = node.operator(children) else {
           return  Ok(FoldUp::Stop(None))
        };
        Ok(FoldUp::Continue(Some(operator)))
    }
}
