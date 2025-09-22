// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::sync::Arc;

use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_bail};

use crate::operator::{Operator, OperatorId, OperatorRef};
use crate::webgpu::{GpuBindContext, GpuKernel, WebGpuOperator};

/// Placeholder operator that wraps a batch operator and exposes it for GPU execution.
#[derive(Clone, Debug)]
pub(crate) struct WebGpuInputOperator {
    inner: OperatorRef,
}

impl WebGpuInputOperator {
    pub fn new(inner: OperatorRef) -> Self {
        Self { inner }
    }
}

impl std::hash::Hash for WebGpuInputOperator {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.inner.hash(state);
    }
}

impl PartialEq for WebGpuInputOperator {
    fn eq(&self, other: &Self) -> bool {
        self.inner.eq(&other.inner)
    }
}

impl Eq for WebGpuInputOperator {}

impl Operator for WebGpuInputOperator {
    fn id(&self) -> OperatorId {
        OperatorId::from("vortex.canonical_gpu")
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dtype(&self) -> &DType {
        self.inner.dtype()
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn children(&self) -> &[OperatorRef] {
        // This operator becomes a batch input in the GPU context
        &[]
    }

    fn with_children(self: Arc<Self>, _children: Vec<OperatorRef>) -> VortexResult<OperatorRef> {
        Ok(self)
    }

    fn as_webgpu(&self) -> Option<&dyn WebGpuOperator> {
        Some(self)
    }
}

impl WebGpuOperator for WebGpuInputOperator {
    fn bind_gpu(&self, _ctx: &dyn GpuBindContext) -> VortexResult<Box<dyn GpuKernel>> {
        // TODO: Return a kernel that loads the batch input and exposes it as GPU buffer
        vortex_bail!("CanonicalGpuOperator binding not yet implemented")
    }

    fn gpu_children(&self) -> Vec<usize> {
        vec![]
    }

    fn batch_children(&self) -> Vec<usize> {
        vec![0] // The inner operator becomes a batch input
    }
}
