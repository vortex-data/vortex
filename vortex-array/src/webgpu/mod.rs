// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(dead_code)]

mod execution;
pub(crate) mod input;
pub(crate) mod operator;

use vortex_error::VortexResult;

use crate::Canonical;
use crate::operator::Operator;

/// Trait for operators that can be executed on WebGpu.
pub trait WebGpuOperator: Operator {
    /// Bind the operator into a WebGpu kernel for GPU execution.
    fn bind_gpu(&self, ctx: &dyn GpuBindContext) -> VortexResult<Box<dyn GpuKernel>>;

    /// Returns the child indices of this operator that are passed to the kernel as input buffers.
    fn gpu_children(&self) -> Vec<usize>;

    /// Returns the child indices of this operator that are passed to the kernel as batch inputs.
    fn batch_children(&self) -> Vec<usize>;
}

/// The context used when binding an operator for GPU execution.
pub trait GpuBindContext {
    fn children(&self) -> &[GpuBufferId];
    fn batch_inputs(&self) -> &[BatchId];
}

/// The ID of the GPU buffer to use.
pub type GpuBufferId = usize;
/// The ID of the batch input to use.
pub type BatchId = usize;

/// A GPU kernel that can be executed on WebGpu.
pub trait GpuKernel: Send {
    /// Execute the kernel on the GPU and return the result.
    ///
    /// TODO: Add actual WebGpu execution context here
    fn execute(&mut self, ctx: &GpuExecutionContext) -> VortexResult<()>;
}

/// Context passed to GPU kernels during execution.
pub struct GpuExecutionContext {
    /// Placeholder for WebGpu device, queue, etc.
    /// TODO: Add actual WebGpu resources here
    pub(crate) batch_inputs: Vec<Canonical>,
}
