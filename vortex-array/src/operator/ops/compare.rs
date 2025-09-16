// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::compute::Operator;
use crate::operator::{ArrayOperator, PipelinedOperator};
use crate::pipeline::operators::BindContext;
use crate::pipeline::Kernel;
use futures::StreamExt;
use itertools::Itertools;
use std::any::Any;
use std::sync::Arc;
use vortex_dtype::DType;
use vortex_error::{VortexExpect, VortexResult};

pub struct CompareOperator {
    children: [Arc<dyn ArrayOperator>; 2],
    operator: Operator,
    dtype: DType,
}

impl ArrayOperator for CompareOperator {
    fn id(&self) -> Arc<str> {
        Arc::from("vortex.compare")
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn len(&self) -> usize {
        debug_assert_eq!(self.children[0].len(), self.children[1].len());
        self.children[0].len()
    }

    fn children(&self) -> &[Arc<dyn ArrayOperator>] {
        &self.children
    }

    fn with_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ArrayOperator>>,
    ) -> VortexResult<Arc<dyn ArrayOperator>> {
        let (lhs, rhs) = children
            .into_iter()
            .tuples()
            .next()
            .vortex_expect("missing");
        Ok(Arc::new(CompareOperator {
            children: [lhs, rhs],
            operator: self.operator,
            dtype: self.dtype.clone(),
        }))
    }

    fn as_pipelined(&self) -> Option<&dyn PipelinedOperator> {
        Some(self)
    }
}

impl PipelinedOperator for CompareOperator {
    fn bind(&self, ctx: &dyn BindContext) -> VortexResult<Box<dyn Kernel>> {
        todo!()
    }
}
