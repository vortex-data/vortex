// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Write;
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaFunction};
use vortex_error::{VortexExpect, VortexResult, vortex_err};

use crate::indent::{IndentedWrite, IndentedWriter};
use crate::jit::{GPUKernelParameter, GPUPipelineJIT, GPUVisitor};

struct DeclPrinter<'a, 'b: 'a> {
    w: &'a mut IndentedWrite<'b>,
}

fn write_kernel_declarations(w: &mut IndentedWrite, node: &dyn GPUPipelineJIT) {
    let mut decl = DeclPrinter { w };
    decl.accept(node).vortex_expect("write decl cannot fail");
}

impl<'a> GPUVisitor<'a> for DeclPrinter<'a, '_> {
    fn accept(&mut self, node: &'a dyn GPUPipelineJIT) -> VortexResult<()> {
        node.children(self)?;
        node.decls(self.w)
            .map_err(|e| vortex_err!("cannot write {}", e))
    }
}

struct InParamCollector {
    params: Vec<GPUKernelParameter>,
}

impl GPUVisitor<'_> for InParamCollector {
    fn accept(&mut self, node: &dyn GPUPipelineJIT) -> VortexResult<()> {
        node.children(self)?;
        node.in_params(&mut self.params);
        Ok(())
    }
}

fn collect_in_param(node: &dyn GPUPipelineJIT) -> VortexResult<Vec<GPUKernelParameter>> {
    let mut params = InParamCollector { params: Vec::new() };
    params.accept(node)?;
    Ok(params.params)
}

pub fn create_kernel_str(
    w: &mut IndentedWrite,
    output: &dyn GPUPipelineJIT,
    kernel_out_array: &str,
) -> VortexResult<()> {
    let mut params = collect_in_param(output)?;
    params.push(GPUKernelParameter {
        name: "_output".to_string(),
        type_: format!("{} *__restrict__", output.output_parameter().type_),
    });

    (|| {
        // TODO: include when only for fast lanes codecs
        writeln!(w, "__device__ int FL_ORDER[] = {{0, 4, 2, 6, 1, 5, 3, 7}};")?;
        writeln!(
            w,
            "#define INDEX(row, lane) (FL_ORDER[row / 8] * 16 + (row % 8) * 128 + lane)"
        )?;
        writeln!(w, "extern \"C\" __global__ void kernel(")?;
        w.indent(|w| {
            for (idx, p) in params.iter().enumerate() {
                let separator = if idx < params.len() - 1 { "," } else { "" };
                writeln!(w, "{} {}{}", p.type_, p.name, separator)?;
            }
            Ok(())
        })?;
        writeln!(w, ") {{")?;

        w.indent(|w| {
            writeln!(
                w,
                "{output_type} *output = _output + (blockIdx.x * 1024);",
                output_type = output.output_parameter().type_
            )?;

            writeln!(w, "__shared__ float {kernel_out_array}[1024];")?;

            write_kernel_declarations(w, output);
            writeln!(w)?;
            output.kernel_body(w, &|_w: &mut IndentedWrite, in_: GPUKernelParameter| {
                Ok(in_)
            })?;
            writeln!(w)?;

            writeln!(w, "for (int i = 0; i < 32; i++) {{")?;
            w.indent(|w| {
                writeln!(w, "auto idx = i * 32 + threadIdx.x;")?;
                writeln!(w, "output[idx] = {kernel_out_array}[idx];")
            })?;
            writeln!(w, "}}")
        })?;

        writeln!(w, "}}")
    })()
    .map_err(|e| vortex_err!("format err {e}"))
}

pub fn create_kernel(
    ctx: Arc<CudaContext>,
    array: &dyn GPUPipelineJIT,
    kernel_out_array: &str,
) -> VortexResult<CudaFunction> {
    let mut s = String::new();
    let w = &mut s as &mut dyn Write;
    let mut ind = IndentedWriter::new(w);
    let w = &mut ind;

    create_kernel_str(w, array, kernel_out_array)
        .map_err(|e| vortex_err!("jit str cannot fail {e}"))?;

    let module =
        cudarc::nvrtc::compile_ptx(s.clone()).map_err(|e| vortex_err!("compile ptx {e}"))?;

    // Dynamically load it into the device
    let module = ctx
        .load_module(module)
        .map_err(|e| vortex_err!("load module {e}"))?;

    module
        .load_function("kernel")
        .map_err(|e| vortex_err!("load_function {e}"))
}
