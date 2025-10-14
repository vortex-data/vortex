// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

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

pub trait GPUPipelineJIT {
    fn step_id(&self) -> usize;

    fn in_params(&self, params: &mut Vec<GPUKernelParameter>);

    fn args<'a>(&'a self, stream: &Arc<CudaStream>, args: &mut LaunchArgs<'a>) -> VortexResult<()>;

    fn decls(&self, w: &mut IndentedWriter<&mut dyn Write>) -> fmt::Result;

    fn kernel_body(
        &self,
        w: &mut IndentedWriter<&mut dyn Write>,
        f: &dyn Fn(&mut IndentedWriter<&mut dyn Write>) -> fmt::Result,
    ) -> fmt::Result;

    fn output_var(&self) -> String;

    fn output_type(&self) -> PType;

    // always pass the output iteration aligned child last.
    fn children<'a>(&'a self, visitor: &mut dyn GPUVisitor<'a>) -> VortexResult<()>;

    fn launch_config(&self) -> GPULaunchConfig;
}

pub trait ScalarGPUPipelineJIT {
    fn step_id(&self) -> usize;

    fn in_params(&self, params: &mut Vec<GPUKernelParameter>);

    fn args<'a>(&'a self, stream: &Arc<CudaStream>, args: &mut LaunchArgs<'a>) -> VortexResult<()>;

    fn decls(&self, w: &mut IndentedWriter<&mut dyn Write>) -> fmt::Result;

    fn kernel_body(
        &self,
        w: &mut IndentedWriter<&mut dyn Write>,
        f: &dyn Fn(&mut IndentedWriter<&mut dyn Write>) -> fmt::Result,
    ) -> fmt::Result;

    fn output_var(&self) -> String;

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

trait GPUVisitor<'a> {
    fn accept(&mut self, node: &'a dyn GPUPipelineJIT) -> VortexResult<()>;
}

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
    fn step_id(&self) -> usize {
        self.inner.step_id()
    }

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
        f: &dyn Fn(&mut IndentedWriter<&mut dyn Write>) -> fmt::Result,
    ) -> fmt::Result {
        self.inner.kernel_body(w, f)
    }

    fn output_var(&self) -> String {
        self.inner.output_var()
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
