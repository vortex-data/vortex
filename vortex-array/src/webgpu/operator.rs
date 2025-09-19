// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::operator::{
    BatchBindCtx, BatchExecution, BatchExecutionRef, BatchOperator, Operator, OperatorId,
    OperatorRef,
};
use crate::Canonical;
use std::any::Any;
use std::sync::Arc;
use vortex_dtype::DType;
use vortex_error::VortexResult;

/// An operator that collapses a subgraph of WebGPU-capable operators into a single WebGPU operator
/// for batch execution.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct WebGpuOperator {
    // root: NodeId,
    // dag: Vec<PipelineNode>,
    // batch_inputs: Vec<OperatorRef>,
}

impl Operator for WebGpuOperator {
    fn id(&self) -> OperatorId {
        OperatorId::from("vortex.webgpu")
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dtype(&self) -> &DType {
        todo!()
    }

    fn len(&self) -> usize {
        todo!()
    }

    fn children(&self) -> &[OperatorRef] {
        todo!()
    }

    fn with_children(self: Arc<Self>, _children: Vec<OperatorRef>) -> VortexResult<OperatorRef> {
        todo!()
    }
}

impl BatchOperator for WebGpuOperator {
    fn bind(&self, ctx: &mut dyn BatchBindCtx) -> VortexResult<BatchExecutionRef> {
        todo!()
    }
}

impl BatchExecution for WebGpuOperator {
    async fn execute(self: Box<Self>) -> VortexResult<Canonical> {
        todo!()
    }
}
