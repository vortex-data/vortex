// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::time::Duration;

use cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT;
use cudarc::driver::{CudaContext, CudaStream, LaunchArgs, LaunchConfig, PushKernelArg};
use vortex_array::arrays::PrimitiveArray;
use vortex_array::validity::Validity;
use vortex_array::{ArrayRef, Canonical, IntoArray};
use vortex_buffer::BufferMut;
use vortex_dtype::match_each_native_ptype;
use vortex_error::{VortexExpect, VortexResult, VortexUnwrap, vortex_err};

use crate::jit::convert::new_jit_array;
use crate::jit::kernel_fmt::create_kernel;
use crate::jit::{GPUPipelineJIT, GPUVisitor};

pub fn create_run_jit_kernel(
    ctx: Arc<CudaContext>,
    array: &ArrayRef,
) -> VortexResult<(ArrayRef, Duration)> {
    let stream = ctx.default_stream();

    let kernel_output_arr_name = "s_output";
    let output = new_jit_array(array, &stream, kernel_output_arr_name.to_string());
    let kernel = create_kernel(ctx.clone(), output.as_ref(), kernel_output_arr_name)?;

    let num_chunks =
        u32::try_from(array.len().div_ceil(1024)).vortex_expect("Too many grid elements");

    let mut launch_builder = stream.launch_builder(&kernel);

    let config = output.launch_config();

    let launch_config = LaunchConfig {
        grid_dim: (num_chunks, 1, 1),
        block_dim: (config.block_width, 1, 1),
        shared_mem_bytes: u32::try_from(output.output_type().byte_width())
            .vortex_expect("oversized output type byte width")
            * 1024,
    };

    collect_args(output.as_ref(), stream.clone(), &mut launch_builder)?;

    match_each_native_ptype!(array.dtype().as_ptype(), |P| {
        // append final argument (output) of the kernel
        let mut out = stream
            .alloc_zeros::<P>(array.len())
            .map_err(|e| vortex_err!("failed to alloc zeros {e}"))?;
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

        let duration = start
            .elapsed_ms(&end)
            .map_err(|e| vortex_err!("failed to get elapsed time {e}"))?;

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

fn collect_args<'a>(
    node: &'a dyn GPUPipelineJIT,
    stream: Arc<CudaStream>,
    args: &mut LaunchArgs<'a>,
) -> VortexResult<()> {
    let mut collector = ArgCollector {
        stream,

        params: args,
    };
    collector.accept(node)?;

    Ok(())
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
