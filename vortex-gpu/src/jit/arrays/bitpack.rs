// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, DeviceRepr, LaunchArgs, PushKernelArg};
use vortex_buffer::Buffer;
use vortex_dtype::{NativePType, PType, match_each_native_ptype};
use vortex_error::{VortexResult, VortexUnwrap, vortex_err};
use vortex_fastlanes::BitPackedArray;

use crate::indent::IndentedWrite;
use crate::jit::{
    CUDAType, GPUKernelParameter, GPULaunchConfig, GPUPipelineJIT, GPUVisitor, StepIdAllocator,
};

struct BitPack<P> {
    step_id: usize,
    bit_width: u8,
    output_type: PType,
    cuda_slice: CudaSlice<P>,
    output_array: String,
}

pub fn new_jit(
    bp: &BitPackedArray,
    stream: &Arc<CudaStream>,
    allocator: &mut StepIdAllocator,
    output_array: String,
) -> Box<dyn GPUPipelineJIT> {
    assert_eq!(bp.offset(), 0);
    assert!(bp.patches().is_none());
    match_each_native_ptype!(bp.ptype(), |P| {
        let values = Buffer::<P>::from_byte_buffer(bp.packed().clone());
        let cuda_slice = stream
            .memcpy_stod(values.as_slice())
            .map_err(|e| vortex_err!("Failed to copy to device: {e}"))
            .vortex_unwrap();
        let step_id = allocator.fresh_id();
        Box::new(BitPack::<P> {
            step_id,
            bit_width: bp.bit_width(),
            output_type: bp.ptype(),
            cuda_slice,
            output_array,
        })
    })
}

impl<P: NativePType + DeviceRepr> GPUPipelineJIT for BitPack<P> {
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

    fn decls(&self, w: &mut IndentedWrite) -> fmt::Result {
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
        w: &mut IndentedWrite,
        f: &dyn Fn(
            &mut IndentedWrite,
            GPUKernelParameter,
        ) -> Result<GPUKernelParameter, fmt::Error>,
    ) -> Result<GPUKernelParameter, fmt::Error> {
        let bit_width = self.bit_width as usize;
        let bits = self.output_type.bit_width();
        let in_ = self.in_var_l();
        if bit_width == 0 {
            writeln!(w, "uint{bits}_t zero = 0ULL;")?;
            writeln!(w)?;
            for row in 0..bits {
                writeln!(w, "out[INDEX({row}, lane)] = zero;")?;
            }
            Ok(GPUKernelParameter {
                name: "none".to_string(),
                type_: "t_none_".to_string(),
            })
        } else if bit_width == bits {
            writeln!(w)?;
            for row in 0..bits {
                writeln!(
                    w,
                    "out[INDEX({row}, lane)] = {in_}[LANE_COUNT * {row} + lane];",
                )?;
            }
            Ok(GPUKernelParameter {
                name: "none".to_string(),
                type_: "t_none_".to_string(),
            })
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

                let out = f(
                    w,
                    GPUKernelParameter {
                        name: tmp.to_string(),
                        type_: "unsigned int".to_string(),
                    },
                )?;
                writeln!(
                    w,
                    "{output_a}[INDEX({row}, lane)] = {in_var};",
                    output_a = self.output_array,
                    in_var = out.name
                )?;
                writeln!(w)?;
            }
            Ok(GPUKernelParameter {
                name: "none___".to_string(),
                type_: "t_none_".to_string(),
            })
        }
    }

    fn output_parameter(&self) -> GPUKernelParameter {
        GPUKernelParameter {
            name: self.tmp_var(),
            type_: CUDAType::from(self.output_type).to_string(),
        }
    }

    fn output_type(&self) -> PType {
        P::PTYPE
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
