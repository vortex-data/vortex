// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::time::Duration;

use cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, CudaViewMut, DeviceRepr, LaunchConfig,
    PushKernelArg,
};
use cudarc::nvrtc::Ptx;
use vortex_array::Canonical;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::validity::Validity;
use vortex_buffer::{Buffer, BufferMut};
use vortex_dtype::{NativePType, PType};
use vortex_error::{VortexExpect, VortexResult, vortex_err};
use vortex_fastlanes::{BitPackedVTable, FoRArray};

use crate::task::GPUTask;

struct FoRBPTask<P> {
    stream: Arc<CudaStream>,
    func: CudaFunction,
    launch_config: LaunchConfig,

    packed: CudaSlice<P>,
    unpacked: CudaSlice<P>,
    reference: P,

    len: usize,
}

pub fn new_task(
    array: &FoRArray,
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
) -> VortexResult<Box<dyn GPUTask>> {
    assert!(!array.is_empty());
    assert_eq!(array.ptype(), PType::U32);
    let bp = array.encoded().as_::<BitPackedVTable>();
    assert_eq!(bp.offset(), 0);
    assert_eq!(bp.bit_width(), 6);

    let num_chunks =
        u32::try_from(array.len().div_ceil(1024)).vortex_expect("Too many grid elements");

    let values = Buffer::<u32>::from_byte_buffer(bp.packed().clone());

    let cu_slice = stream
        .memcpy_stod(values.as_slice())
        .map_err(|e| vortex_err!("Failed to copy to device: {e}"))?;
    let cu_out = unsafe {
        stream
            .alloc::<u32>(array.len().next_multiple_of(1024))
            .map_err(|e| vortex_err!("Failed to allocate stream: {e}"))?
    };

    Ok(Box::new(FoRBPTask {
        stream,
        func: cuda_for_bp_kernel(array.ptype(), &ctx)?,
        launch_config: LaunchConfig {
            grid_dim: (num_chunks, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        },
        packed: cu_slice,
        unpacked: cu_out,
        reference: array
            .reference_scalar()
            .as_primitive()
            .as_::<u32>()
            .vortex_expect("cannot have a null ref"),
        len: array.len(),
    }))
}

fn cuda_for_bp_kernel(_ptype: PType, ctx: &Arc<CudaContext>) -> VortexResult<CudaFunction> {
    let module = ctx
        .load_module(Ptx::from_file("kernels/fused_bitpack_for.ptx"))
        .map_err(|e| vortex_err!("Failed to load kernel module: {e}"))?;

    let kernel_func = module
        .load_function("fused_bitpack6_for_u32")
        .map_err(|e| vortex_err!("Failed to load function: {e}"))?;
    Ok(kernel_func)
}

impl<P: NativePType + DeviceRepr> GPUTask for FoRBPTask<P> {
    fn launch_task(&mut self) -> VortexResult<()> {
        let mut launch = self.stream.launch_builder(&self.func);
        launch.arg(&self.packed);
        launch.arg(&self.unpacked);
        launch.arg(&self.reference);
        launch.record_kernel_launch(CU_EVENT_DEFAULT);
        unsafe { launch.launch(self.launch_config) }
            .map_err(|e| vortex_err!("Failed to launch: {e}"))?;
        Ok(())
    }

    fn export_result(&mut self) -> VortexResult<Canonical> {
        let len = self.len();
        let mut buffer = BufferMut::<P>::with_capacity(len);

        unsafe { buffer.set_len(len) }
        self.stream
            .memcpy_dtoh(&self.unpacked, &mut buffer)
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
        todo!()
    }

    fn len(&self) -> usize {
        self.len
    }
}

pub fn cuda_for_bp_unpack_timed(
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

pub fn cuda_for_bp_unpack(array: &FoRArray, ctx: Arc<CudaContext>) -> VortexResult<PrimitiveArray> {
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
            (0u32..4096).map(|i| i % 8).collect::<Buffer<_>>(),
            Validity::NonNullable,
        );
        let array = BitPackedArray::encode(primitive_array.as_ref(), 6).vortex_unwrap();
        let array = FoRArray::try_new(array.into_array(), 8u32.into()).vortex_unwrap();
        let ctx = CudaContext::new(0).unwrap();
        ctx.set_blocking_synchronize().unwrap();
        let unpacked = cuda_for_bp_unpack(&array, ctx).unwrap();
        let primitive_array = array.into_array().to_primitive();
        assert_eq!(
            primitive_array.as_slice::<u32>(),
            unpacked.as_slice::<u32>()
        );
        for i in 0..primitive_array.len() {
            assert_eq!(
                primitive_array.as_slice::<u32>()[i],
                unpacked.as_slice::<u32>()[i],
                "i {i}"
            );
        }
    }
}
