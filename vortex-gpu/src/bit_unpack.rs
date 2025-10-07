// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// This code is only exercised on CI with cuda and linux
#![allow(dead_code)]

use std::sync::{Arc, LazyLock};
use std::time::Duration;

use cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, CudaViewMut, DeviceRepr, LaunchConfig,
    PushKernelArg,
};
use cudarc::nvrtc::Ptx;
use parking_lot::RwLock;
use vortex_array::Canonical;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::validity::Validity;
use vortex_buffer::{Buffer, BufferMut};
use vortex_dtype::{NativePType, PType, match_each_unsigned_integer_ptype};
use vortex_error::{VortexExpect, VortexResult, vortex_err};
use vortex_fastlanes::BitPackedArray;
use vortex_utils::aliases::hash_map::HashMap;

use crate::task::GPUTask;

#[derive(Hash, PartialEq, Eq, Debug)]
struct UnpackKernelId {
    bit_width: u8,
    output_bit_width: u8,
}

impl UnpackKernelId {
    fn new(bit_width: u8, output_bit_width: u8) -> Self {
        Self {
            bit_width,
            output_bit_width,
        }
    }
}

static CUDA_KERNELS: LazyLock<RwLock<HashMap<UnpackKernelId, CudaFunction>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));

fn cuda_bit_unpack_kernel(
    kernel_id: UnpackKernelId,
    ctx: Arc<CudaContext>,
) -> VortexResult<CudaFunction> {
    if let Some(kernel) = CUDA_KERNELS.read().get(&kernel_id) {
        return Ok(kernel.clone());
    }
    let module = ctx
        .load_module(Ptx::from_file(format!(
            "kernels/fls_{}_bit_unpack.ptx",
            kernel_id.output_bit_width
        )))
        .map_err(|e| vortex_err!("Failed to load kernel module: {e}"))?;

    let kernel_func = module
        .load_function(
            format!(
                "fls_unpack_{}bw_{}ow_{}t",
                kernel_id.bit_width,
                kernel_id.output_bit_width,
                if kernel_id.output_bit_width == 64 {
                    "16"
                } else {
                    "32"
                }
            )
            .as_ref(),
        )
        .map_err(|e| vortex_err!("Failed to load function: {e}"))?;
    CUDA_KERNELS.write().insert(kernel_id, kernel_func.clone());
    Ok(kernel_func)
}

pub fn cuda_bit_unpack(
    array: &BitPackedArray,
    ctx: Arc<CudaContext>,
) -> VortexResult<PrimitiveArray> {
    let stream = ctx.default_stream();
    let mut task = new_task(array, ctx, stream)?;
    task.launch_task()?;
    task.export_result().map(|c| c.into_primitive())
}

/// Returns the time (in nanoseconds) to execute just the GPU kernel, excluding memory transfers.
/// The input array must already be allocated on the GPU.
pub fn cuda_bit_unpack_timed(
    array: &BitPackedArray,
    ctx: Arc<CudaContext>,
) -> VortexResult<Duration> {
    let stream = ctx.default_stream();
    let mut task = new_task(array, ctx.clone(), stream.clone())?;

    let start = stream
        .record_event(Some(CU_EVENT_DEFAULT))
        .ok()
        .vortex_expect("Failed to record event");

    task.launch_task()?;

    // Synchronize to ensure kernel completes
    ctx.synchronize()
        .map_err(|e| vortex_err!("Failed to synchronize: {e}"))?;

    let end = stream
        .record_event(Some(CU_EVENT_DEFAULT))
        .ok()
        .vortex_expect("Failed to record event");

    // Get elapsed time in milliseconds
    let elapsed_ms = Duration::from_secs_f32(
        start
            .elapsed_ms(&end)
            .map_err(|e| vortex_err!("Failed to get elapsed time: {e}"))?
            / 1000.0,
    );

    // Convert to nanoseconds
    Ok(elapsed_ms)
}

struct BitPackingTask<P> {
    packed: CudaSlice<P>,
    unpacked: CudaSlice<P>,
    func: CudaFunction,
    launch_config: LaunchConfig,
    stream: Arc<CudaStream>,
    len: usize,
    ptype: PType,
}

pub fn new_task(
    array: &BitPackedArray,
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
) -> VortexResult<Box<dyn GPUTask>> {
    assert!(!array.is_empty());

    assert!(array.patches().is_none(), "Patches not supported");
    assert_eq!(array.offset(), 0, "Offset must be 0");
    assert_eq!(
        array.len() % 1024,
        0,
        "Array can't have incomplete end chunk"
    );

    let kernel_func = cuda_bit_unpack_kernel(
        UnpackKernelId::new(
            array.bit_width(),
            u8::try_from(array.dtype().as_ptype().bit_width())
                .vortex_expect("bit width must fit in u8"),
        ),
        ctx,
    )?;
    let num_chunks =
        u32::try_from(array.len().div_ceil(1024)).vortex_expect("Too many grid elements");

    match_each_unsigned_integer_ptype!(array.dtype().as_ptype().to_unsigned(), |P| {
        let values = Buffer::<P>::from_byte_buffer(array.packed().clone());
        // TODO(robert): You likely want to register (cuMemHostRegister) and unregister here
        let cu_slice = stream
            .memcpy_stod(values.as_slice())
            .map_err(|e| vortex_err!("Failed to copy to device: {e}"))?;
        let cu_out = unsafe {
            stream
                .alloc::<P>(array.len().next_multiple_of(1024))
                .map_err(|e| vortex_err!("Failed to allocate stream: {e}"))?
        };

        let launch_config = LaunchConfig {
            grid_dim: (num_chunks, 1, 1),
            block_dim: (if size_of::<P>() == 8 { 16 } else { 32 }, 1, 1),
            shared_mem_bytes: 0,
        };

        Ok(Box::new(BitPackingTask {
            packed: cu_slice,
            unpacked: cu_out,
            func: kernel_func,
            launch_config,
            stream,
            len: array.len(),
            ptype: array.ptype(),
        }))
    })
}

impl<P: NativePType + DeviceRepr> GPUTask for BitPackingTask<P> {
    fn launch_task(&mut self) -> VortexResult<()> {
        let mut launch = self.stream.launch_builder(&self.func);
        launch.arg(&self.packed);
        launch.arg(&self.unpacked);
        unsafe { launch.launch(self.launch_config) }
            .map_err(|e| vortex_err!("Failed to launch: {e}"))
            .map(|_| ())
    }

    fn export_result(&mut self) -> VortexResult<Canonical> {
        let mut buffer = BufferMut::<P>::with_capacity(self.len());

        unsafe { buffer.set_len(self.len()) }
        self.stream
            .memcpy_dtoh(&self.unpacked, &mut buffer)
            .map_err(|e| vortex_err!("Failed to copy to device: {e}"))?;
        self.stream
            .synchronize()
            .map_err(|e| vortex_err!("Failed to synchronize: {e}"))?;
        Ok(Canonical::Primitive(
            PrimitiveArray::new(buffer, Validity::NonNullable).reinterpret_cast(self.ptype),
        ))
    }

    fn output(&mut self) -> CudaViewMut<'_, u8> {
        unsafe {
            self.unpacked
                .transmute_mut(self.len() * size_of::<P>())
                .vortex_expect("Failed to transmute")
        }
    }

    fn len(&self) -> usize {
        self.len
    }
}

#[cfg(all(target_os = "linux", feature = "cuda"))]
#[cfg(test)]
mod tests {
    use cudarc::driver::CudaContext;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_error::VortexUnwrap;
    use vortex_fastlanes::BitPackedArray;

    use super::*;

    #[test]
    fn test_cuda_bitunpack() {
        let primitive_array = PrimitiveArray::new(
            (0u32..4096).map(|i| i % 63).collect::<Buffer<_>>(),
            Validity::NonNullable,
        );
        let array = BitPackedArray::encode(primitive_array.as_ref(), 6).vortex_unwrap();
        let ctx = CudaContext::new(0).unwrap();
        ctx.set_blocking_synchronize().unwrap();
        let unpacked = cuda_bit_unpack(&array, ctx).unwrap();
        assert_eq!(
            primitive_array.as_slice::<u32>(),
            unpacked.as_slice::<u32>()
        );
    }
}
