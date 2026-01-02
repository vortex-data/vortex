// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::fmt::Write;
use std::sync::Arc;

use cudarc::driver::{CudaStream, DeviceRepr, LaunchArgs, PushKernelArg};
use vortex_dtype::{NativePType, PType, match_each_native_ptype};
use vortex_error::{VortexExpect, VortexResult};
use vortex_fastlanes::FoRArray;

use crate::indent::{IndentedWrite, IndentedWriter};
use crate::jit::convert::handle_array;
use crate::jit::{
    CUDAType, GPUKernelParameter, GPUPipelineJIT, ScalarGPUPipelineJIT, ScalarGPUPipelineJITNode,
    StepIdAllocator,
};

struct FoR<P> {
    step_id: usize,
    reference: P,
    child: Box<dyn GPUPipelineJIT>,
}

pub fn new_jit(
    for_: &FoRArray,
    stream: &Arc<CudaStream>,
    allocator: &mut StepIdAllocator,
    output_array: String,
) -> Box<dyn GPUPipelineJIT> {
    match_each_native_ptype!(for_.reference_scalar().as_primitive().ptype(), |P| {
        let child = handle_array(for_.encoded(), stream, allocator, output_array);
        Box::new(ScalarGPUPipelineJITNode {
            inner: FoR {
                step_id: allocator.fresh_id(),
                reference: for_
                    .reference_scalar()
                    .as_primitive()
                    .as_::<P>()
                    .vortex_expect("cannot have a null reference"),
                child,
            },
        })
    })
}

impl<P> FoR<P> {
    fn tmp_var(&self) -> String {
        format!("tmp{}", self.step_id)
    }

    fn ref_var(&self) -> String {
        format!("ref{}", self.step_id)
    }
}

impl<P: NativePType + DeviceRepr> ScalarGPUPipelineJIT for FoR<P> {
    fn in_params(&self, p: &mut Vec<GPUKernelParameter>) {
        p.push(GPUKernelParameter {
            name: self.ref_var(),
            type_: self.output_parameter().type_,
        })
    }

    fn args<'a>(
        &'a self,
        _stream: &Arc<CudaStream>,
        args: &mut LaunchArgs<'a>,
    ) -> VortexResult<()> {
        args.arg(&self.reference);
        Ok(())
    }

    fn decls(&self, w: &mut IndentedWriter<&mut dyn Write>) -> fmt::Result {
        let output_param = self.output_parameter();
        writeln!(w, "{} {};", output_param.type_, output_param.name)?;
        Ok(())
    }

    fn kernel_body(
        &self,
        w: &mut IndentedWrite,
        f: &dyn Fn(
            &mut IndentedWrite,
            GPUKernelParameter,
        ) -> Result<GPUKernelParameter, fmt::Error>,
    ) -> Result<GPUKernelParameter, fmt::Error> {
        let output_param = self.output_parameter();
        let child_output_type = self.child.output_parameter().type_;
        assert_eq!(output_param.type_, child_output_type);
        let ref_var = self.ref_var();
        self.child
            .kernel_body(w, &move |w: &mut IndentedWrite, in_: GPUKernelParameter| {
                let in_var = in_.name;
                writeln!(
                    w,
                    "{out_var} = {in_var} + {ref_var};",
                    out_var = output_param.name
                )?;
                f(w, output_param.clone())
            })
    }

    fn output_parameter(&self) -> GPUKernelParameter {
        GPUKernelParameter {
            name: self.tmp_var(),
            type_: CUDAType::from(P::PTYPE).to_string(),
        }
    }

    fn output_type(&self) -> PType {
        P::PTYPE
    }

    fn child(&self) -> &dyn GPUPipelineJIT {
        self.child.as_ref()
    }
}
