// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::fmt::{Display, Write};

use vortex_array::ArrayRef;
use vortex_dtype::PType;
use vortex_error::{VortexResult, vortex_err};
use vortex_fastlanes::{BitPackedVTable, FoRVTable};

use crate::indent::IndentedWriter;

pub enum IterationOrder {
    InOrder,
    FastLanesTransposed,
}

struct GPUKernelParameter {
    name: String,
    type_: String,
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

trait GPUPipelineJIT {
    fn step_id(&self) -> usize;

    fn in_params(&self, params: &mut Vec<GPUKernelParameter>);

    fn decls(&self, w: &mut IndentedWriter<&mut dyn Write>) -> fmt::Result;

    fn kernel_body(
        &self,
        w: &mut IndentedWriter<&mut dyn Write>,
        f: &dyn Fn(&mut IndentedWriter<&mut dyn Write>) -> fmt::Result,
    ) -> fmt::Result;

    fn output_var(&self) -> String;

    fn output_type(&self) -> PType;
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

struct BitPack {
    step_id: usize,
    output_type: PType,
}

impl GPUPipelineJIT for BitPack {
    fn step_id(&self) -> usize {
        self.step_id
    }

    fn in_params(&self, p: &mut Vec<GPUKernelParameter>) {
        p.push(GPUKernelParameter {
            name: self.src_var(),
            type_: format!("{} *__restrict", CUDAType::from(self.output_type)),
        });
    }

    fn decls(&self, w: &mut IndentedWriter<&mut dyn Write>) -> fmt::Result {
        let output_cuda_type = CUDAType::from(self.output_type);
        // TODO: all types
        writeln!(w, "unsigned int LANE_COUNT = 32;")?;
        writeln!(w, "{output_cuda_type} src{};", self.step_id)?;
        writeln!(w, "{output_cuda_type} tmp{};", self.step_id)?;
        Ok(())
    }

    fn kernel_body(
        &self,
        w: &mut IndentedWriter<&mut dyn Write>,
        f: &dyn Fn(&mut IndentedWriter<&mut dyn Write>) -> fmt::Result,
    ) -> fmt::Result {
        for i in 0..4 {
            let src = self.src_var();
            let tmp = self.tmp_var();
            writeln!(w, "{src} = in[thread_ix + {i}];")?;
            writeln!(w, "{tmp} = ({src} >> 0) & MASK(uint32_t, 1);")?;
            f(w)?;
            writeln!(w)?;
        }
        Ok(())
    }

    fn output_var(&self) -> String {
        self.tmp_var()
    }

    fn output_type(&self) -> PType {
        self.output_type
    }
}

impl BitPack {
    fn tmp_var(&self) -> String {
        format!("tmp{}", self.step_id)
    }

    fn src_var(&self) -> String {
        format!("src{}", self.step_id)
    }
}

struct FoR {
    step_id: usize,
    reference_type: PType,
    child: Box<dyn GPUPipelineJIT>,
}

impl FoR {
    fn tmp_var(&self) -> String {
        format!("tmp{}", self.step_id)
    }

    fn ref_var(&self) -> String {
        format!("ref{}", self.step_id)
    }
}

impl GPUPipelineJIT for FoR {
    fn step_id(&self) -> usize {
        self.step_id
    }

    fn in_params(&self, p: &mut Vec<GPUKernelParameter>) {
        self.child.in_params(p);
        p.push(GPUKernelParameter {
            name: self.ref_var(),
            type_: CUDAType::from(self.output_type()).to_string(),
        })
    }

    fn decls(&self, w: &mut IndentedWriter<&mut dyn Write>) -> fmt::Result {
        self.child.decls(w)?;
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
        self.reference_type
    }
}

fn handle_array(a: &ArrayRef, step_id: usize) -> Box<dyn GPUPipelineJIT> {
    if let Some(bp) = a.as_opt::<BitPackedVTable>() {
        return Box::new(BitPack {
            step_id,
            output_type: bp.ptype(),
        });
    };

    if let Some(for_) = a.as_opt::<FoRVTable>() {
        return Box::new(FoR {
            step_id,
            reference_type: for_.reference_scalar().as_primitive().ptype(),
            child: handle_array(for_.encoded(), step_id + 1),
        });
    }

    todo!()
}

fn _create_jit(a: &ArrayRef) -> fmt::Result {
    let output = handle_array(a, 0);

    let mut s = String::new();
    let w = &mut s as &mut dyn Write;
    let mut ind = IndentedWriter::new(w);
    let w = &mut ind;

    let mut params = Vec::new();
    output.in_params(&mut params);
    params.push(GPUKernelParameter {
        name: "output".to_string(),
        type_: format!("{} *__restrict", CUDAType::from(output.output_type())),
    });

    writeln!(w, "__global__ void kernel(")?;
    w.indent(|w| {
        params
            .iter()
            .try_for_each(|p| writeln!(w, "{} {},", p.type_, p.name))
    })?;
    writeln!(w, ") {{")?;

    w.indent(|w| {
        output.decls(w)?;
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
    use vortex_array::IntoArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_error::VortexResult;
    use vortex_fastlanes::{BitPackedArray, FoRArray};

    use crate::jit::create_jit;

    #[test]
    fn jit_arr() -> VortexResult<()> {
        let for_ = FoRArray::try_new(
            BitPackedArray::encode(
                (0u32..10)
                    .map(|_| 1u32)
                    .collect::<PrimitiveArray>()
                    .as_ref(),
                2,
            )?
            .into_array(),
            2u32.into(),
        )?;

        create_jit(&for_.into_array())?;

        Ok(())
    }
}
