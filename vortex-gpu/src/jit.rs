// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::fmt::{Debug, Display, Write};
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DeviceRepr, LaunchArgs, PushKernelArg};
use vortex_alp::{ALPFloat, ALPVTable, Exponents, match_each_alp_float_ptype};
use vortex_array::ArrayRef;
use vortex_buffer::ByteBuffer;
use vortex_dtype::half::f16;
use vortex_dtype::{NativePType, PType, match_each_native_ptype};
use vortex_error::{VortexExpect, VortexResult, VortexUnwrap, vortex_err};
use vortex_fastlanes::{BitPackedArray, BitPackedVTable, FoRVTable};

use crate::indent::IndentedWriter;

pub enum IterationOrder {
    InOrder,
    FastLanesTransposed,
}

struct GPUKernelParameter {
    name: String,
    type_: String,
}

struct GPULaunchConfig {
    block_width: usize,
    grid_width: usize,
}

// struct GPUPipelineParameters {
//     inputs: Vec<GPUKernelParameter>,
//     output: GPUKernelParameter,
//     block_width: usize,
//     grid_width: usize,
//     iteration_order: IterationOrder,
// }
//
// struct GPUPipeline {
//     body: String,
//     parameters: GPUPipelineParameters,
// }

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

    fn launch_config(&self) -> Option<GPULaunchConfig> {
        None
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
            PType::U8 => "uint8_t",
            PType::U16 => "uint16_t",
            PType::U32 => "uint32_t",
            PType::U64 => "uint64_t",
            PType::I8 => "int8_t",
            PType::I16 => "int16_t",
            PType::I32 => "int32_t",
            PType::I64 => "int64_t",
            PType::F32 => "float",
            PType::F64 => "double",
            PType::F16 => todo!(),
        })
    }
}

struct BitPack<P> {
    step_id: usize,
    bit_width: u8,
    packed: ByteBuffer,
    output_type: PType,
    cuda_slice: CudaSlice<P>,
}

impl<P: NativePType + DeviceRepr> GPUPipelineJIT for BitPack<P> {
    fn step_id(&self) -> usize {
        self.step_id
    }

    fn in_params(&self, p: &mut Vec<GPUKernelParameter>) {
        p.push(GPUKernelParameter {
            name: self.src_var(),
            type_: format!(
                "{type_} *__restrict",
                type_ = CUDAType::from(self.output_type)
            ),
        });
    }

    fn args<'a>(
        &'a self,
        stream: &Arc<CudaStream>,
        launch_args: &mut LaunchArgs<'a>, // args: &mut Vec<Box<dyn DeviceRepr>>,
    ) -> VortexResult<()> {
        launch_args.arg(&self.cuda_slice);

        Ok(())
    }

    fn decls(&self, w: &mut IndentedWriter<&mut dyn Write>) -> fmt::Result {
        let output_cuda_type = CUDAType::from(self.output_type);
        // TODO: all types
        writeln!(
            w,
            "unsigned int LANE_COUNT = {bits};",
            bits = 1024 / self.output_type.bit_width()
        )?;
        writeln!(w, "{output_cuda_type} src{};", self.step_id)?;
        writeln!(w, "{output_cuda_type} tmp{};", self.step_id)?;
        Ok(())
    }

    fn kernel_body(
        &self,
        w: &mut IndentedWriter<&mut dyn Write>,
        f: &dyn Fn(&mut IndentedWriter<&mut dyn Write>) -> fmt::Result,
    ) -> fmt::Result {
        let output = w;
        let bit_width = self.bit_width as usize;
        let bits = self.output_type.bit_width();
        if bit_width == 0 {
            writeln!(output, "uint{bits}_t zero = 0ULL;")?;
            writeln!(output)?;
            for row in 0..bits {
                writeln!(output, "out[INDEX({row}, lane)] = zero;")?;
            }
        } else if bit_width == bits {
            writeln!(output)?;
            for row in 0..bits {
                writeln!(
                    output,
                    "out[INDEX({row}, lane)] = in[LANE_COUNT * {row} + lane];",
                )?;
            }
        } else {
            let src = self.src_var();
            let tmp = self.tmp_var();

            writeln!(output)?;
            writeln!(output, "{src} = in[lane];")?;
            for row in 0..bits {
                let curr_word = (row * bit_width) / bits;
                let next_word = ((row + 1) * bit_width) / bits;
                let shift = (row * bit_width) % bits;

                if next_word > curr_word {
                    let remaining_bits = ((row + 1) * bit_width) % bits;
                    let current_bits = bit_width - remaining_bits;
                    writeln!(
                        output,
                        "{tmp} = ({src} >> {shift}) & MASK(uint{bits}_t, {current_bits});"
                    )?;

                    if next_word < bit_width {
                        writeln!(output, "{src} = in[lane + LANE_COUNT * {next_word}];")?;
                        writeln!(
                            output,
                            "{tmp} |= ({src} & MASK(uint{bits}_t, {remaining_bits})) << {current_bits};"
                        )?;
                    }
                } else {
                    writeln!(
                        output,
                        "{tmp} = ({src} >> {shift}) & MASK(uint{bits}_t, {bit_width});"
                    )?;
                }

                f(output)?;
                writeln!(output)?;
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

    fn launch_config(&self) -> Option<GPULaunchConfig> {
        Some(GPULaunchConfig {
            block_width: 1024,
            grid_width: 1,
        })
    }
}

impl<P> BitPack<P> {
    fn tmp_var(&self) -> String {
        format!("tmp{}", self.step_id)
    }

    fn src_var(&self) -> String {
        format!("src{}", self.step_id)
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

impl<P: NativePType + DeviceRepr> GPUPipelineJIT for FoR<P> {
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
        // TODO: supprort all types
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

    fn children<'a>(&'a self, visitor: &mut dyn GPUVisitor<'a>) -> VortexResult<()> {
        visitor.accept(self.child.as_ref())
    }
}

fn handle_array(a: &ArrayRef, stream: &Arc<CudaStream>, step_id: usize) -> Box<dyn GPUPipelineJIT> {
    if let Some(alp) = a.as_opt::<ALPVTable>() {
        match_each_alp_float_ptype!(alp.ptype(), |A| {
            return Box::new(ALP {
                step_id,
                float_type: alp.ptype(),
                child: handle_array(alp.encoded(), stream, step_id + 1),
                f: A::F10[alp.exponents().f as usize],
                e: A::IF10[alp.exponents().e as usize],
            });
        })
    }
    if let Some(bp) = a.as_opt::<BitPackedVTable>() {
        assert_eq!(bp.offset(), 0);
        assert!(bp.patches().is_none());
        let cuda_slice = stream
            .memcpy_stod(bp.packed().as_slice())
            .map_err(|e| vortex_err!("Failed to copy to device: {e}"))
            .vortex_unwrap();
        return Box::new(BitPack {
            step_id,
            bit_width: bp.bit_width(),
            packed: bp.packed().clone(),
            output_type: bp.ptype(),
            cuda_slice,
        });
    };

    if let Some(for_) = a.as_opt::<FoRVTable>() {
        match_each_native_ptype!(for_.reference_scalar().as_primitive().ptype(), |P| {
            return Box::new(FoR {
                step_id,
                reference: for_
                    .reference_scalar()
                    .as_primitive()
                    .as_::<P>()
                    .vortex_expect("cannot have a null reference"),
                child: handle_array(for_.encoded(), stream, step_id + 1),
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

impl<A: ALPFloat + DeviceRepr> GPUPipelineJIT for ALP<A> {
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

    fn args<'a>(&'a self, stream: &Arc<CudaStream>, args: &mut LaunchArgs<'a>) -> VortexResult<()> {
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
                    tmp = self.tmp_var(),
                    type_ = CUDAType::from(self.float_type),
                    out = self.child.output_var(),
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

    fn children<'a>(&'a self, visitor: &mut dyn GPUVisitor<'a>) -> VortexResult<()> {
        visitor.accept(self.child.as_ref())
    }
}

struct DeclPrinter<'a, 'b: 'a> {
    w: &'a mut IndentedWriter<&'b mut dyn Write>,
}

impl<'a> GPUVisitor<'a> for DeclPrinter<'a, '_> {
    fn accept(&mut self, node: &'a dyn GPUPipelineJIT) -> VortexResult<()> {
        node.children(self)?;
        node.decls(self.w)
            .map_err(|e| vortex_err!("cannot write {}", e))
    }
}

struct InParamPrinter {
    params: Vec<GPUKernelParameter>,
}

impl GPUVisitor<'_> for InParamPrinter {
    fn accept(&mut self, node: &dyn GPUPipelineJIT) -> VortexResult<()> {
        node.children(self)?;
        node.in_params(&mut self.params);
        Ok(())
    }
}

struct ArgCollector<'a> {
    stream: Arc<CudaStream>,
    params: &'a mut LaunchArgs<'a>,
}

impl<'a> GPUVisitor<'a> for ArgCollector<'a> {
    fn accept(&mut self, node: &'a dyn GPUPipelineJIT) -> VortexResult<()> {
        node.children(self)?;
        node.args(&self.stream, self.params)?;
        Ok(())
    }
}

fn _create_jit(a: &ArrayRef) -> fmt::Result {
    let ctx = CudaContext::new(0).unwrap();
    ctx.set_blocking_synchronize().unwrap();
    let output = handle_array(a, &ctx.default_stream(), 0);

    let mut s = String::new();
    let w = &mut s as &mut dyn Write;
    let mut ind = IndentedWriter::new(w);
    let w = &mut ind;

    let mut params = InParamPrinter { params: Vec::new() };
    params.accept(output.as_ref()).vortex_expect("cannot fail");

    params.params.push(GPUKernelParameter {
        name: "output".to_string(),
        type_: format!("{} *__restrict", CUDAType::from(output.output_type())),
    });

    writeln!(w, "__global__ void kernel(")?;
    w.indent(|w| {
        params
            .params
            .iter()
            .try_for_each(|p| writeln!(w, "{} {},", p.type_, p.name))
    })?;
    writeln!(w, ") {{")?;

    w.indent(|w| {
        let mut decl = DeclPrinter { w };
        decl.accept(output.as_ref()).vortex_expect("cannot fail");
        writeln!(w)?;
        output.kernel_body(w, &|w: &mut IndentedWriter<&mut dyn Write>| {
            writeln!(w, "output[idx] = {}", output.output_var())
        })
    })?;
    writeln!(w, "}}")?;

    println!("{}", s);

    Ok(())
}

fn create_jit(a: &ArrayRef) -> VortexResult<()> {
    _create_jit(a).map_err(|e| vortex_err!("failed to write decls {e}"))
}

#[cfg(test)]
mod tests {
    use vortex_alp::{ALPArray, Exponents};
    use vortex_array::IntoArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_error::VortexResult;
    use vortex_fastlanes::{BitPackedArray, FoRArray};

    use crate::jit::create_jit;

    #[test]
    fn jit_arr() -> VortexResult<()> {
        let for_ = ALPArray::try_new(
            FoRArray::try_new(
                BitPackedArray::encode(
                    (0i32..10)
                        .map(|_| 1i32)
                        .collect::<PrimitiveArray>()
                        .as_ref(),
                    2,
                )?
                .into_array(),
                2i32.into(),
            )?
            .into_array(),
            Exponents { e: 4, f: 5 },
            None,
        )?;

        create_jit(&for_.into_array())?;

        Ok(())
    }
}
