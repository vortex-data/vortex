// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::time::Duration;

use cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaStream, CudaViewMut, DeviceRepr, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;
use vortex_array::Canonical;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::validity::Validity;
use vortex_buffer::BufferMut;
use vortex_dtype::{NativePType, PType, match_each_native_ptype};
use vortex_error::{VortexExpect, VortexResult, vortex_err};
use vortex_fastlanes::{BitPackedVTable, FoRArray};

use crate::bit_unpack;
use crate::task::GPUTask;

struct ForTask<P> {
    stream: Arc<CudaStream>,
    func: CudaFunction,
    bp_task: Box<dyn GPUTask>,
    launch_config: LaunchConfig,
    reference: P,
}

pub fn new_task(
    array: &FoRArray,
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
) -> VortexResult<Box<dyn GPUTask>> {
    assert!(!array.is_empty());
    let bp = array.encoded().as_::<BitPackedVTable>();
    let bp_task = bit_unpack::new_task(bp, ctx.clone(), stream.clone())?;

    let num_chunks =
        u32::try_from(array.len().div_ceil(1024)).vortex_expect("Too many grid elements");

    match_each_native_ptype!(array.ptype(), |P| {
        Ok(Box::new(ForTask {
            stream,
            func: cuda_for_kernel(array.ptype(), &ctx)?,
            bp_task,
            launch_config: LaunchConfig {
                grid_dim: (num_chunks, 1, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            },
            reference: array
                .reference_scalar()
                .as_primitive()
                .as_::<P>()
                .vortex_expect("cannot have a null ref"),
        }))
    })
}

fn cuda_for_kernel(ptype: PType, ctx: &Arc<CudaContext>) -> VortexResult<CudaFunction> {
    let module = ctx
        .load_module(Ptx::from_file("kernels/for.ptx"))
        .map_err(|e| vortex_err!("Failed to load kernel module: {e}"))?;

    module
        .load_function(format!("for_v{}", ptype).as_ref())
        .map_err(|e| vortex_err!("Failed to load function: {e}"))
}

impl<P: NativePType + DeviceRepr> GPUTask for ForTask<P> {
    fn launch_task(&mut self) -> VortexResult<()> {
        let len = self.len();
        self.bp_task.launch_task()?;
        let mut launch = self.stream.launch_builder(&self.func);
        let mut view = unsafe {
            self.bp_task
                .output()
                .transmute_mut::<P>(len)
                .vortex_expect("")
        };
        launch.arg(&mut view);
        launch.arg(&self.reference);
        unsafe { launch.launch(self.launch_config) }
            .map_err(|e| vortex_err!("Failed to launch: {e}"))
            .map(|_| ())
    }

    fn export_result(&mut self) -> VortexResult<Canonical> {
        let len = self.len();
        let mut buffer = BufferMut::<P>::with_capacity(len);

        unsafe { buffer.set_len(len) }
        self.stream
            .memcpy_dtoh(
                &unsafe { self.bp_task.output().transmute::<P>(len).vortex_expect("") },
                &mut buffer,
            )
            .map_err(|e| vortex_err!("Failed to copy to device: {e}"))?;
        self.stream
            .synchronize()
            .map_err(|e| vortex_err!("Failed to synchronize: {e}"))?;
        Ok(Canonical::Primitive(PrimitiveArray::new(
            buffer,
            Validity::NonNullable,
        )))
    }

    fn output(&mut self) -> CudaViewMut<'_, u8> {
        self.bp_task.output()
    }

    fn len(&self) -> usize {
        self.bp_task.len()
    }
}

pub fn cuda_for_unpack_timed(
    array: &FoRArray,
    ctx: Arc<CudaContext>,
) -> VortexResult<(PrimitiveArray, Duration)> {
    let stream = ctx.default_stream();
    let mut task = new_task(array, ctx.clone(), stream.clone())?;
    let start = stream
        .record_event(Some(CU_EVENT_DEFAULT))
        .ok()
        .vortex_expect("Failed to record event");
    task.launch_task()?;
    ctx.synchronize()
        .map_err(|e| vortex_err!("Failed to synchronize: {e}"))?;
    let end = stream
        .record_event(Some(CU_EVENT_DEFAULT))
        .ok()
        .vortex_expect("Failed to record event");
    let time = Duration::from_secs_f32(
        start
            .elapsed_ms(&end)
            .ok()
            .vortex_expect("Failed to get elapsed time")
            / 1000.0,
    );
    task.export_result()
        .map(|c| c.into_primitive())
        .map(|x| (x, time))
}

pub fn cuda_for_unpack(array: &FoRArray, ctx: Arc<CudaContext>) -> VortexResult<PrimitiveArray> {
    let stream = ctx.default_stream();
    let mut task = new_task(array, ctx, stream)?;
    task.launch_task()?;
    task.export_result().map(|c| c.into_primitive())
}

#[cfg(all(target_os = "linux", feature = "cuda"))]
#[cfg(test)]
mod tests {
    use cudarc::driver::CudaContext;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::validity::Validity;
    use vortex_array::{IntoArray, ToCanonical};
    use vortex_buffer::Buffer;
    use vortex_error::VortexUnwrap;
    use vortex_fastlanes::{BitPackedArray, FoRArray};

    use super::*;

    #[test]
    fn test_cuda_for_bp() {
        let primitive_array = PrimitiveArray::new(
            (0u32..4096).map(|i| i % 63).collect::<Buffer<_>>(),
            Validity::NonNullable,
        );
        let array = BitPackedArray::encode(primitive_array.as_ref(), 6).vortex_unwrap();
        let array = FoRArray::try_new(array.into_array(), 1u32.into()).vortex_unwrap();
        let ctx = CudaContext::new(0).unwrap();
        ctx.set_blocking_synchronize().unwrap();
        let unpacked = cuda_for_unpack(&array, ctx).unwrap();
        let primitive_array = array.into_array().to_primitive();
        assert_eq!(
            primitive_array.as_slice::<u32>(),
            unpacked.as_slice::<u32>()
        );
    }
}
