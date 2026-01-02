// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::fmt::Write;
use std::sync::Arc;

use cudarc::driver::{CudaStream, DeviceRepr, LaunchArgs, PushKernelArg};
use vortex_alp::{ALPArray, ALPFloat, match_each_alp_float_ptype};
use vortex_dtype::PType;
use vortex_error::VortexResult;

use crate::indent::{IndentedWrite, IndentedWriter};
use crate::jit::convert::handle_array;
use crate::jit::{
    CUDAType, GPUKernelParameter, GPUPipelineJIT, ScalarGPUPipelineJIT, ScalarGPUPipelineJITNode,
    StepIdAllocator,
};

struct Alp<A: ALPFloat> {
    step_id: usize,
    float_type: PType,
    child: Box<dyn GPUPipelineJIT>,
    f: A,
    e: A,
}

pub fn new_jit(
    alp: &ALPArray,
    stream: &Arc<CudaStream>,
    allocator: &mut StepIdAllocator,
    output_array: String,
) -> Box<dyn GPUPipelineJIT> {
    match_each_alp_float_ptype!(alp.ptype(), |A| {
        let child = handle_array(alp.encoded(), stream, allocator, output_array);
        let step_id = allocator.fresh_id();
        Box::new(ScalarGPUPipelineJITNode {
            inner: Alp {
                step_id,
                float_type: alp.ptype(),
                child,
                f: A::F10[alp.exponents().f as usize],
                e: A::IF10[alp.exponents().e as usize],
            },
        })
    })
}

impl<A: ALPFloat> Alp<A> {
    fn tmp_var(&self) -> String {
        format!("tmp{}", self.step_id)
    }

    fn e_var(&self) -> String {
        format!("e{}", self.step_id)
    }

    fn f_var(&self) -> String {
        format!("f{}", self.step_id)
    }
}

impl<A: ALPFloat + DeviceRepr> ScalarGPUPipelineJIT for Alp<A> {
    fn in_params(&self, params: &mut Vec<GPUKernelParameter>) {
        params.extend([
            GPUKernelParameter {
                name: self.e_var(),
                type_: CUDAType::from(A::PTYPE).to_string(),
            },
            GPUKernelParameter {
                name: self.f_var(),
                type_: CUDAType::from(A::PTYPE).to_string(),
            },
        ])
    }

    fn args<'a>(
        &'a self,
        _stream: &Arc<CudaStream>,
        args: &mut LaunchArgs<'a>,
    ) -> VortexResult<()> {
        args.arg(&self.e);
        args.arg(&self.f);
        Ok(())
    }

    fn decls(&self, w: &mut IndentedWriter<&mut dyn Write>) -> fmt::Result {
        let output_cuda_type = CUDAType::from(self.float_type);
        writeln!(w, "{} tmp{};", output_cuda_type, self.step_id)?;
        Ok(())
    }

    fn kernel_body(
        &self,
        w: &mut IndentedWriter<&mut dyn Write>,
        f: &dyn Fn(
            &mut IndentedWriter<&mut dyn Write>,
            GPUKernelParameter,
        ) -> Result<GPUKernelParameter, fmt::Error>,
    ) -> Result<GPUKernelParameter, fmt::Error> {
        self.child
            .kernel_body(w, &|w: &mut IndentedWrite, in_: GPUKernelParameter| {
                let in_var = in_.name;
                writeln!(
                    w,
                    "{out} = ((({type_}){in_var}) * {f}) * {e};",
                    out = self.tmp_var(),
                    type_ = CUDAType::from(self.float_type),
                    f = self.f_var(),
                    e = self.e_var(),
                )?;
                f(w, self.output_parameter())
            })
    }

    fn output_parameter(&self) -> GPUKernelParameter {
        GPUKernelParameter {
            name: self.tmp_var(),
            type_: CUDAType::from(self.float_type).to_string(),
        }
    }

    fn output_type(&self) -> PType {
        self.float_type
    }

    fn child(&self) -> &dyn GPUPipelineJIT {
        self.child.as_ref()
    }
}
