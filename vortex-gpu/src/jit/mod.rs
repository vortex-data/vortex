// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! # JIT Kernel Composition System
//!
//! This module generates CUDA kernels by composing encoding steps into a single fused kernel.
//!
//! ## How Kernels Are Built
//!
//! Each encoding step (BitPack, FoR, ALP) implements `GPUPipelineJIT` and contributes:
//! 1. **Input parameters** - Data passed from host to device (arrays, scalars)
//! 2. **Declarations** - Local variables needed for the step
//! 3. **Kernel body** - The actual computation logic
//! 4. **Output variable** - The result of this step (e.g., `tmp0`, `tmp1`)
//!
//! Steps are composed in a tree structure. For example: `ALP -> FoR -> BitPack`
//!
//! ## Data Flow Between Steps
//!
//! Each step produces an **output variable** that the parent step consumes:
//!
//! ```text
//! BitPack:  unpacks data → produces `tmp0`
//! FoR:      reads `tmp0` → adds reference → produces `tmp1` = tmp0 + ref0
//! ALP:      reads `tmp1` → scales → produces `tmp2` = tmp1 * f2 * e2
//! ```
//!
//! The `output_var()` method returns the variable name (e.g., "tmp2") that subsequent
//! steps or the final output can read from.
//!
//! ## Final Kernel Result
//!
//! The composed kernel computes the final value and writes it to the output array:
//!
//! ```cuda
//! // BitPack unpacks data
//! tmp0 = unpack(...)
//! // FoR adds reference value
//! tmp1 = tmp0 + ref0
//! // ALP scales the result
//! tmp2 = tmp1 * scale
//! // Final write to output
//! output[out_idx] = tmp2
//! ```
//!
//! ## Continuation-Based Composition
//!
//! Each step calls a continuation function after computing its output:
//! - The continuation function receives a `GPUKernelParameter` (the child's output variable)
//! - The step uses this variable to perform its computation
//! - The step then calls its own continuation with its output variable
//! - This creates a chain: innermost → ... → outermost → final write
//!
//! Example flow:
//! ```text
//! BitPack.kernel_body(w, continuation):
//!   // Unpacks data
//!   tmp0 = unpack(...)
//!   continuation(w, GPUKernelParameter{name: "tmp0", type_: "int32_t"})
//!
//! FoR.kernel_body(w, continuation):
//!   child_var = self.child.kernel_body(w, continuation)  // Gets "tmp0"
//!   tmp1 = child_var + ref0
//!   continuation(w, GPUKernelParameter{name: "tmp1", type_: "int32_t"})
//!
//! ALP.kernel_body(w, continuation):
//!   child_var = self.child.kernel_body(w, continuation)  // Gets "tmp1"
//!   tmp2 = child_var * scale
//!   continuation(w, GPUKernelParameter{name: "tmp2", type_: "float"})
//! ```
//!
//! The root continuation writes the final result to the output array.

mod arrays;
mod convert;
mod kernel_fmt;
mod run;
mod type_;

use std::fmt;
use std::fmt::Write;
use std::sync::Arc;

use cudarc::driver::{CudaStream, LaunchArgs};
pub use run::create_run_jit_kernel;
pub use type_::CUDAType;
use vortex_dtype::PType;
use vortex_error::VortexResult;

use crate::indent::IndentedWriter;

/// Type alias for the continuation function passed to `kernel_body`.
///
/// The continuation receives the output parameter from a child step and returns
/// the final output parameter after all parent steps have been applied.
pub type KernelContinuation<'a> = dyn Fn(
        &mut IndentedWriter<&mut dyn Write>,
        GPUKernelParameter,
    ) -> Result<GPUKernelParameter, fmt::Error>
    + 'a;

/// Trait for encoding steps that can be JIT-compiled into a CUDA kernel.
///
/// Each step contributes a piece of the kernel and specifies its output variable
/// that subsequent steps can read from.
pub trait GPUPipelineJIT {
    /// Adds input parameters (e.g., device pointers, scalars) to the kernel signature
    fn in_params(&self, params: &mut Vec<GPUKernelParameter>);

    /// Adds arguments to the kernel launch (actual values passed at runtime)
    fn args<'a>(&'a self, stream: &Arc<CudaStream>, args: &mut LaunchArgs<'a>) -> VortexResult<()>;

    /// Writes variable declarations needed by this step
    fn decls(&self, w: &mut IndentedWriter<&mut dyn Write>) -> fmt::Result;

    /// Writes the kernel body for this step.
    ///
    /// The continuation function `f` should be called after computing this step's output,
    /// allowing parent steps to consume the output variable via `output_var()`.
    fn kernel_body(
        &self,
        w: &mut IndentedWriter<&mut dyn Write>,
        f: &KernelContinuation,
    ) -> Result<GPUKernelParameter, fmt::Error>;

    /// Returns the name+type of the output variable
    fn output_parameter(&self) -> GPUKernelParameter;

    fn output_type(&self) -> PType;

    /// Visits child steps in the pipeline tree
    fn children<'a>(&'a self, visitor: &mut dyn GPUVisitor<'a>) -> VortexResult<()>;

    /// Returns the launch configuration (block size, etc.) for this kernel
    fn launch_config(&self) -> GPULaunchConfig;
}

pub trait ScalarGPUPipelineJIT {
    fn in_params(&self, params: &mut Vec<GPUKernelParameter>);

    fn args<'a>(&'a self, stream: &Arc<CudaStream>, args: &mut LaunchArgs<'a>) -> VortexResult<()>;

    fn decls(&self, w: &mut IndentedWriter<&mut dyn Write>) -> fmt::Result;

    fn kernel_body(
        &self,
        w: &mut IndentedWriter<&mut dyn Write>,
        f: &KernelContinuation,
    ) -> Result<GPUKernelParameter, fmt::Error>;

    /// Returns the name+type of the output variable
    fn output_parameter(&self) -> GPUKernelParameter;

    fn output_type(&self) -> PType;

    fn child(&self) -> &dyn GPUPipelineJIT;
}

#[derive(Default)]
struct StepIdAllocator {
    next_id: usize,
}

impl StepIdAllocator {
    pub fn fresh_id(&mut self) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        id
    }
}

pub trait GPUVisitor<'a> {
    fn accept(&mut self, node: &'a dyn GPUPipelineJIT) -> VortexResult<()>;
}

#[derive(Clone)]
pub struct GPUKernelParameter {
    name: String,
    type_: String,
}

pub struct GPULaunchConfig {
    block_width: u32,
}

struct ScalarGPUPipelineJITNode<T> {
    inner: T,
}

impl<T: ScalarGPUPipelineJIT> GPUPipelineJIT for ScalarGPUPipelineJITNode<T> {
    fn in_params(&self, params: &mut Vec<GPUKernelParameter>) {
        self.inner.in_params(params)
    }

    fn args<'a>(&'a self, stream: &Arc<CudaStream>, args: &mut LaunchArgs<'a>) -> VortexResult<()> {
        self.inner.args(stream, args)
    }

    fn decls(&self, w: &mut IndentedWriter<&mut dyn Write>) -> fmt::Result {
        self.inner.decls(w)
    }

    fn kernel_body(
        &self,
        w: &mut IndentedWriter<&mut dyn Write>,
        f: &KernelContinuation,
    ) -> Result<GPUKernelParameter, fmt::Error> {
        self.inner.kernel_body(w, f)
    }

    fn output_parameter(&self) -> GPUKernelParameter {
        self.inner.output_parameter()
    }

    fn output_type(&self) -> PType {
        self.inner.output_type()
    }

    fn children<'a>(&'a self, visitor: &mut dyn GPUVisitor<'a>) -> VortexResult<()> {
        visitor.accept(self.inner.child())
    }

    fn launch_config(&self) -> GPULaunchConfig {
        self.inner.child().launch_config()
    }
}
