// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::fmt::{Display, Write};
use std::sync::Arc;
use std::time::Duration;

use cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, DeviceRepr, LaunchArgs, LaunchConfig,
    PushKernelArg,
};
use vortex_alp::{ALPFloat, ALPVTable, match_each_alp_float_ptype};
use vortex_array::arrays::PrimitiveArray;
use vortex_array::validity::Validity;
use vortex_array::{Array, ArrayRef, Canonical, IntoArray};
use vortex_buffer::{Buffer, BufferMut};
use vortex_dtype::{NativePType, PType, match_each_native_ptype};
use vortex_error::{VortexExpect, VortexResult, VortexUnwrap, vortex_err};
use vortex_fastlanes::{BitPackedVTable, FoRVTable};

use crate::indent::IndentedWriter;

struct GPUKernelParameter {
    name: String,
    type_: String,
}

struct GPULaunchConfig {
    block_width: u32,
}

// bp -> output
// tmp = ...
// out[i] = tmp;

// bp -> for -> output

// tmp = ....
// tmp_for = tmp + ref
// output[i] = tmp_for[i]

// have leaves only bp for now.

// step-type (each one has a unique step_id)
// step_id
// in_params
// decls/setup
// kernel-step body // fn body(var, writer) -> str
// output_var + output_type

trait GPUVisitor<'a> {
    fn accept(&mut self, node: &'a dyn GPUPipelineJIT) -> VortexResult<()>;
}

trait GPUPipelineJIT {
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

trait ScalarGPUPipelineJIT {
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

    fn child<'a>(&'a self) -> &'a dyn GPUPipelineJIT;
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

struct CUDAType(&'static str);

impl Display for CUDAType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.0)
    }
}

impl From<PType> for CUDAType {
    fn from(value: PType) -> Self {
        CUDAType(match value {
            PType::U8 => "unsigned char",
            PType::U16 => "unsigned short",
            PType::U32 => "unsigned int",
            PType::U64 => "unsigned long long",
            PType::I8 => "char",
            PType::I16 => "short",
            PType::I32 => "int",
            PType::I64 => "long long",
            PType::F32 => "float",
            PType::F64 => "double",
            PType::F16 => todo!(),
        })
    }
}

struct BitPack<P> {
    step_id: usize,
    bit_width: u8,
    output_type: PType,
    cuda_slice: CudaSlice<P>,
}

impl<P: NativePType + DeviceRepr> GPUPipelineJIT for BitPack<P> {
    fn step_id(&self) -> usize {
        self.step_id
    }

    fn in_params(&self, p: &mut Vec<GPUKernelParameter>) {
        p.push(GPUKernelParameter {
            name: self.in_var_g(),
            type_: format!(
                "{type_} *__restrict",
                type_ = CUDAType::from(self.output_type.to_unsigned())
            ),
        });
    }

    fn args<'a>(
        &'a self,
        _stream: &Arc<CudaStream>,
        launch_args: &mut LaunchArgs<'a>,
    ) -> VortexResult<()> {
        launch_args.arg(&self.cuda_slice);

        Ok(())
    }

    fn decls(&self, w: &mut IndentedWriter<&mut dyn Write>) -> fmt::Result {
        let output_cuda_type = CUDAType::from(self.output_type);
        let uoutput_cuda_type = CUDAType::from(self.output_type.to_unsigned());
        writeln!(
            w,
            "unsigned int LANE_COUNT = {bits};",
            bits = 1024 / self.output_type.bit_width()
        )?;
        writeln!(w, "{output_cuda_type} {};", self.tmp_var())?;
        writeln!(w, "{uoutput_cuda_type} {};", self.src_var())?;
        writeln!(w, "{uoutput_cuda_type} {};", self.utmp_var())?;
        writeln!(w, "unsigned int out_idx;")?;
        writeln!(w, "unsigned int lane = threadIdx.x;")?;
        writeln!(
            w,
            "{uoutput_cuda_type} *{in_l} = {in_g} + (blockIdx.x * 128 * {bit_width} / {bit_size});",
            in_l = self.in_var_l(),
            in_g = self.in_var_g(),
            bit_width = self.bit_width,
            bit_size = P::PTYPE.byte_width()
        )?;
        Ok(())
    }

    fn kernel_body(
        &self,
        w: &mut IndentedWriter<&mut dyn Write>,
        f: &dyn Fn(&mut IndentedWriter<&mut dyn Write>) -> fmt::Result,
    ) -> fmt::Result {
        let bit_width = self.bit_width as usize;
        let bits = self.output_type.bit_width();
        let in_ = self.in_var_l();
        if bit_width == 0 {
            writeln!(w, "uint{bits}_t zero = 0ULL;")?;
            writeln!(w)?;
            for row in 0..bits {
                writeln!(w, "out[INDEX({row}, lane)] = zero;")?;
            }
        } else if bit_width == bits {
            writeln!(w)?;
            for row in 0..bits {
                writeln!(
                    w,
                    "out[INDEX({row}, lane)] = {in_}[LANE_COUNT * {row} + lane];",
                )?;
            }
        } else {
            let src = self.src_var();
            let utmp = self.utmp_var();
            let tmp = self.tmp_var();

            let mask_fn = |bits: usize| {
                format!(
                    "((({type_})1 << {width}) - 1)",
                    type_ = CUDAType::from(P::PTYPE.to_unsigned()),
                    width = bits
                )
            };

            writeln!(w)?;
            writeln!(w, "{src} = {in}[lane];", in = self.in_var_l())?;
            for row in 0..bits {
                let curr_word = (row * bit_width) / bits;
                let next_word = ((row + 1) * bit_width) / bits;
                let shift = (row * bit_width) % bits;

                if next_word > curr_word {
                    let remaining_bits = ((row + 1) * bit_width) % bits;
                    let current_bits = bit_width - remaining_bits;
                    writeln!(
                        w,
                        "{utmp} = ({src} >> {shift}) & {mask};",
                        mask = mask_fn(current_bits)
                    )?;

                    if next_word < bit_width {
                        writeln!(w, "{src} = {in_}[lane + LANE_COUNT * {next_word}];")?;
                        writeln!(
                            w,
                            "{utmp} |= ({src} & {mask}) << {current_bits};",
                            mask = mask_fn(remaining_bits)
                        )?;
                    }
                } else {
                    writeln!(
                        w,
                        "{utmp} = ({src} >> {shift}) & {mask};",
                        mask = mask_fn(bit_width)
                    )?;
                }
                writeln!(
                    w,
                    "{tmp} = ({type_}){utmp};",
                    type_ = CUDAType::from(self.output_type),
                )?;

                writeln!(w, "out_idx = INDEX({row}, lane);")?;
                f(w)?;
                writeln!(w)?;
            }
        }
        Ok(())
    }

    fn output_var(&self) -> String {
        self.tmp_var()
    }

    fn output_type(&self) -> PType {
        self.output_type
    }

    fn children(&self, _visitor: &mut dyn GPUVisitor) -> VortexResult<()> {
        Ok(())
    }

    fn launch_config(&self) -> GPULaunchConfig {
        GPULaunchConfig {
            block_width: if P::PTYPE == PType::U64 { 16 } else { 32 },
        }
    }
}

impl<P> BitPack<P> {
    fn tmp_var(&self) -> String {
        format!("tmp{}", self.step_id)
    }

    fn src_var(&self) -> String {
        format!("src{}", self.step_id)
    }

    fn utmp_var(&self) -> String {
        format!("utmp{}", self.step_id)
    }

    fn in_var_l(&self) -> String {
        format!("in{}", self.step_id)
    }

    fn in_var_g(&self) -> String {
        format!("_in{}", self.step_id)
    }
}

struct FoR<P> {
    step_id: usize,
    reference: P,
    child: Box<dyn GPUPipelineJIT>,
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
    fn step_id(&self) -> usize {
        self.step_id
    }

    fn in_params(&self, p: &mut Vec<GPUKernelParameter>) {
        p.push(GPUKernelParameter {
            name: self.ref_var(),
            type_: CUDAType::from(self.output_type()).to_string(),
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
        let output_cuda_type = CUDAType::from(self.output_type());
        writeln!(w, "{} tmp{};", output_cuda_type, self.step_id)?;
        Ok(())
    }

    fn kernel_body(
        &self,
        w: &mut IndentedWriter<&mut dyn Write>,
        f: &dyn Fn(&mut IndentedWriter<&mut dyn Write>) -> fmt::Result,
    ) -> fmt::Result {
        assert_eq!(self.output_type(), self.child.output_type());
        let in_var = self.child.output_var();
        let out_var = self.tmp_var();
        let ref_var = self.ref_var();
        self.child
            .kernel_body(w, &|w: &mut IndentedWriter<&mut dyn Write>| {
                writeln!(w, "{out_var} = {in_var} + {ref_var};")?;
                f(w)
            })
    }

    fn output_var(&self) -> String {
        self.tmp_var()
    }

    fn output_type(&self) -> PType {
        P::PTYPE
    }

    fn child<'a>(&'a self) -> &'a dyn GPUPipelineJIT {
        self.child.as_ref()
    }
}

fn handle_array(a: &ArrayRef, stream: &Arc<CudaStream>, step_id: usize) -> Box<dyn GPUPipelineJIT> {
    if let Some(alp) = a.as_opt::<ALPVTable>() {
        match_each_alp_float_ptype!(alp.ptype(), |A| {
            return Box::new(ScalarGPUPipelineJITNode {
                inner: ALP {
                    step_id,
                    float_type: alp.ptype(),
                    child: handle_array(alp.encoded(), stream, step_id + 1),
                    f: A::F10[alp.exponents().f as usize],
                    e: A::IF10[alp.exponents().e as usize],
                },
            });
        })
    }
    if let Some(bp) = a.as_opt::<BitPackedVTable>() {
        assert_eq!(bp.offset(), 0);
        assert!(bp.patches().is_none());
        match_each_native_ptype!(bp.ptype(), |P| {
            let values = Buffer::<P>::from_byte_buffer(bp.packed().clone());
            let cuda_slice = stream
                .memcpy_stod(values.as_slice())
                .map_err(|e| vortex_err!("Failed to copy to device: {e}"))
                .vortex_unwrap();
            return Box::new(BitPack::<P> {
                step_id,
                bit_width: bp.bit_width(),
                output_type: bp.ptype(),
                cuda_slice,
            });
        })
    };

    if let Some(for_) = a.as_opt::<FoRVTable>() {
        match_each_native_ptype!(for_.reference_scalar().as_primitive().ptype(), |P| {
            return Box::new(ScalarGPUPipelineJITNode {
                inner: FoR {
                    step_id,
                    reference: for_
                        .reference_scalar()
                        .as_primitive()
                        .as_::<P>()
                        .vortex_expect("cannot have a null reference"),
                    child: handle_array(for_.encoded(), stream, step_id + 1),
                },
            });
        })
    }

    todo!()
}

struct ALP<A: ALPFloat> {
    step_id: usize,
    float_type: PType,
    child: Box<dyn GPUPipelineJIT>,
    f: A,
    e: A,
}

impl<A: ALPFloat> ALP<A> {
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

impl<A: ALPFloat + DeviceRepr> ScalarGPUPipelineJIT for ALP<A> {
    fn step_id(&self) -> usize {
        self.step_id
    }

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
        f: &dyn Fn(&mut IndentedWriter<&mut dyn Write>) -> fmt::Result,
    ) -> fmt::Result {
        self.child
            .kernel_body(w, &|w: &mut IndentedWriter<&mut dyn Write>| {
                writeln!(
                    w,
                    "{out} = ((({type_}){tmp}) * {f}) * {e};",
                    out = self.tmp_var(),
                    type_ = CUDAType::from(self.float_type),
                    tmp = self.child.output_var(),
                    f = self.f_var(),
                    e = self.e_var(),
                )?;
                f(w)
            })
    }

    fn output_var(&self) -> String {
        self.tmp_var()
    }

    fn output_type(&self) -> PType {
        self.float_type
    }

    fn child<'a>(&'a self) -> &'a dyn GPUPipelineJIT {
        self.child.as_ref()
    }
}

struct DeclPrinter<'a, 'b: 'a> {
    w: &'a mut IndentedWriter<&'b mut dyn Write>,
}

fn write_kernel_declarations(w: &mut IndentedWriter<&mut dyn Write>, node: &dyn GPUPipelineJIT) {
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

struct ArgCollector<'a, 'b> {
    stream: Arc<CudaStream>,
    params: &'b mut LaunchArgs<'a>,
}

impl<'a> GPUVisitor<'a> for ArgCollector<'a, '_> {
    fn accept(&mut self, node: &'a dyn GPUPipelineJIT) -> VortexResult<()> {
        node.children(self)?;
        node.args(&self.stream, self.params)?;
        Ok(())
    }
}

fn collect_args<'a, 'b>(
    node: &'a dyn GPUPipelineJIT,
    stream: Arc<CudaStream>,
    args: &'b mut LaunchArgs<'a>,
) -> VortexResult<()> {
    let mut collector = ArgCollector {
        stream: stream.clone(),

        params: args,
    };
    collector.accept(node)?;

    Ok(())
}

fn create_jit_str(
    w: &mut IndentedWriter<&mut dyn Write>,
    output: &dyn GPUPipelineJIT,
) -> VortexResult<()> {
    let mut params = collect_in_param(output)?;
    params.push(GPUKernelParameter {
        name: "_output".to_string(),
        type_: format!("{} *__restrict__", CUDAType::from(output.output_type())),
    });

    (|| {
        writeln!(w, "__device__ int FL_ORDER[] = {{0, 4, 2, 6, 1, 5, 3, 7}};")?;
        writeln!(
            w,
            "#define INDEX(row, lane) (FL_ORDER[row / 8] * 16 + (row % 8) * 128 + lane)"
        )?;
        writeln!(w, "extern \"C\" __global__ void kernel(")?;
        w.indent(|w| {
            params.iter().enumerate().try_for_each(|(idx, p)| {
                writeln!(
                    w,
                    "{} {}{end}",
                    p.type_,
                    p.name,
                    end = if idx == params.len() - 1 { "" } else { "," }
                )
            })
        })?;
        writeln!(w, ") {{")?;

        w.indent(|w| {
            writeln!(
                w,
                "{output_type} *output = _output + (blockIdx.x * 1024);",
                output_type = CUDAType::from(output.output_type())
            )?;

            writeln!(w, "__shared__ float s_output[1024];")?;

            write_kernel_declarations(w, output);
            writeln!(w)?;
            output.kernel_body(w, &|w: &mut IndentedWriter<&mut dyn Write>| {
                writeln!(w, "s_output[out_idx] = {tmp};", tmp = output.output_var())
            })
        })?;

        writeln!(
            w,
            "    for (int i = 0; i < 32; i++) {{
        auto idx = i * 32 + threadIdx.x;
        output[idx] = s_output[idx];
    }}"
        )?;

        writeln!(w, "}}")
    })()
    .map_err(|e| vortex_err!("format err {e}"))
}

fn create_kernel(ctx: Arc<CudaContext>, array: &dyn GPUPipelineJIT) -> VortexResult<CudaFunction> {
    let mut s = String::new();
    let w = &mut s as &mut dyn Write;
    let mut ind = IndentedWriter::new(w);
    let w = &mut ind;

    let _ = create_jit_str(w, array).map_err(|e| vortex_err!("jit str cannot fail {e}"));

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

pub fn create_run_jit_kernel(
    ctx: Arc<CudaContext>,
    array: &ArrayRef,
) -> VortexResult<(ArrayRef, Duration)> {
    let stream = ctx.default_stream();

    let output = handle_array(array, &stream, 0);
    let kernel = create_kernel(ctx.clone(), output.as_ref())?;

    let num_chunks =
        u32::try_from(array.len().div_ceil(1024)).vortex_expect("Too many grid elements");

    let mut launch_builder = stream.launch_builder(&kernel);

    let launch_config = LaunchConfig {
        grid_dim: (num_chunks, 1, 1),
        block_dim: (output.launch_config().block_width, 1, 1),
        shared_mem_bytes: u32::try_from(output.output_type().byte_width())
            .vortex_expect("oversized output type byte width")
            * 1024,
    };

    collect_args(output.as_ref(), stream.clone(), &mut launch_builder)?;

    match_each_native_ptype!(array.dtype().as_ptype(), |P| {
        // append final argument (output) of the kernel
        let mut out = stream.alloc_zeros::<P>(array.len()).unwrap();
        launch_builder.arg(&mut out);
        stream
            .synchronize()
            .map_err(|e| vortex_err!("failed to sync {e}"))?;
        let start = stream
            .record_event(Some(CU_EVENT_DEFAULT))
            .ok()
            .vortex_expect("Failed to record event");
        let _ = unsafe { launch_builder.launch(launch_config) };
        ctx.synchronize()
            .map_err(|e| vortex_err!("Failed to synchronize: {e}"))?;
        let end = stream
            .record_event(Some(CU_EVENT_DEFAULT))
            .ok()
            .vortex_expect("Failed to record event");

        let duration = start.elapsed_ms(&end).unwrap();

        let mut buffer = BufferMut::<P>::with_capacity(array.len());
        unsafe { buffer.set_len(array.len()) }

        stream
            .memcpy_dtoh(&out, &mut buffer)
            .map_err(|e| vortex_err!("Failed to copy to device: {e}"))
            .vortex_unwrap();
        stream
            .synchronize()
            .map_err(|e| vortex_err!("Failed to synchronize: {e}"))
            .vortex_unwrap();
        let c =
            Canonical::Primitive(PrimitiveArray::new(buffer, Validity::NonNullable)).into_array();

        Ok((c, Duration::from_secs_f32(duration / 1000.0)))
    })
}

#[cfg(test)]
mod tests {
    use cudarc::driver::CudaContext;
    use vortex_alp::{ALPArray, Exponents};
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::{IntoArray, ToCanonical};
    use vortex_error::VortexResult;
    use vortex_fastlanes::{BitPackedArray, FoRArray};

    use crate::jit::create_run_jit_kernel;

    #[test]
    fn jit_arr_f32() -> VortexResult<()> {
        let ctx = CudaContext::new(0).unwrap();
        ctx.set_blocking_synchronize().unwrap();
        let for_ = ALPArray::try_new(
            FoRArray::try_new(
                BitPackedArray::encode((0i32..1024 * 2).collect::<PrimitiveArray>().as_ref(), 12)?
                    .into_array(),
                2i32.into(),
            )?
            .into_array(),
            Exponents { e: 4, f: 5 },
            None,
        )?
        .into_array();

        let (d, _) = create_run_jit_kernel(ctx, &for_)?;
        let prim = d.to_primitive();
        let expect = for_.to_primitive();

        for i in 0..prim.len() {
            assert_eq!(
                prim.as_slice::<f32>()[i],
                expect.as_slice::<f32>()[i],
                "i = {i}"
            );
        }

        Ok(())
    }

    #[test]
    fn jit_arrs() -> VortexResult<()> {
        let ctx = CudaContext::new(0).unwrap();
        ctx.set_blocking_synchronize().unwrap();
        let for_ = BitPackedArray::encode(
            (0i32..1024)
                .map(|_| 1u32)
                .collect::<PrimitiveArray>()
                .as_ref(),
            2,
        )?
        .into_array();

        create_run_jit_kernel(ctx, &for_.into_array())?;

        Ok(())
    }
}
