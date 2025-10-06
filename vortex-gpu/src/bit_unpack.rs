// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// This code is only exercised on CI with cuda and linux
#![allow(dead_code)]

use std::sync::{Arc, LazyLock};
use std::time::{Duration, Instant};

use cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, CudaViewMut, DeviceRepr, LaunchArgs,
    LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;
use parking_lot::RwLock;
use vortex_array::Canonical;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::compute::Cost::Canonicalize;
use vortex_array::validity::Validity;
use vortex_array::vtable::ValidityHelper;
use vortex_buffer::{Buffer, BufferMut};
use vortex_dtype::{
    NativePType, PType, match_each_native_ptype, match_each_unsigned_integer_ptype,
};
use vortex_error::{VortexExpect, VortexResult, vortex_err};
use vortex_fastlanes::{BitPackedArray, BitPackedVTable, FoRArray};
use vortex_utils::aliases::hash_map::HashMap;

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
    let mut task = new_bit_packing_task(array, ctx, stream.clone())?;
    task.launch_task()?;
    task.export_result().map(|c| c.into_primitive())
}

pub fn cuda_for_unpack_timed(
    array: &FoRArray,
    ctx: Arc<CudaContext>,
) -> VortexResult<PrimitiveArray> {
    let stream = ctx.default_stream();
    let mut task = new_for_task(array, ctx, stream.clone())?;
    let time = Instant::now();
    let start = stream.record_event(Some(CU_EVENT_DEFAULT));
    task.launch_task()?;
    let end = stream.record_event(Some(CU_EVENT_DEFAULT));
    let time = Duration::from_secs_f32(start.unwrap().elapsed_ms(&end.unwrap()).unwrap() / 1000.0);
    println!("time {:?}", time);
    task.export_result().map(|c| c.into_primitive())
}

pub fn cuda_for_unpack(array: &FoRArray, ctx: Arc<CudaContext>) -> VortexResult<PrimitiveArray> {
    let stream = ctx.default_stream();
    let mut task = new_for_task(array, ctx, stream.clone())?;
    let time = Instant::now();
    task.launch_task()?;
    task.export_result().map(|c| c.into_primitive())
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

trait GPUTask {
    // Must call `launch_task` or `launch_task_timed` once
    fn launch_task(&mut self) -> VortexResult<()>;

    // Must call this after launch_task
    fn export_result(&mut self) -> VortexResult<Canonical>;

    // Re can transmute as runtime
    fn output(&mut self) -> CudaViewMut<u8>;

    fn len(&self) -> usize;
}

fn new_bit_packing_task(
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
        ctx.clone(),
    )?;
    let num_chunks =
        u32::try_from(array.len().div_ceil(1024)).vortex_expect("Too many grid elements");
    let stream = ctx.default_stream();

    match_each_unsigned_integer_ptype!(array.dtype().as_ptype().to_unsigned(), |P| {
        let values = Buffer::<P>::from_byte_buffer(array.packed().clone());
        // TODO(robert): You likely want to register (cuMemHostRegister) and unregister here
        let cu_slice = stream
            .memcpy_stod(values.as_slice())
            .map_err(|e| vortex_err!("Failed to copy to device: {e}"))?;
        let mut cu_out = unsafe {
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
            ptype: P::PTYPE,
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

    fn output(&mut self) -> CudaViewMut<u8> {
        unsafe {
            self.unpacked
                .transmute_mut(self.len() * size_of::<P>())
                .unwrap()
        }
    }

    fn len(&self) -> usize {
        self.len
    }
}

struct ForTask<P> {
    stream: Arc<CudaStream>,
    func: CudaFunction,
    bp_task: Box<dyn GPUTask>,
    launch_config: LaunchConfig,
    reference: P,
}

fn new_for_task(
    array: &FoRArray,
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
) -> VortexResult<Box<dyn GPUTask>> {
    assert!(!array.is_empty());
    let bp = array.encoded().as_::<BitPackedVTable>();
    let bp_task = new_bit_packing_task(bp, ctx.clone(), stream.clone())?;

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

    let kernel_func = module
        .load_function(format!("for_v{}", ptype).as_ref())
        .map_err(|e| vortex_err!("Failed to load function: {e}"))?;
    Ok(kernel_func)
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

    fn output(&mut self) -> CudaViewMut<u8> {
        self.bp_task.output()
    }

    fn len(&self) -> usize {
        self.bp_task.len()
    }
}

#[cfg(all(target_os = "linux", feature = "cuda"))]
#[cfg(test)]
mod tests {
    use cudarc::driver::CudaContext;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::validity::Validity;
    use vortex_array::{Array, IntoArray, ToCanonical};
    use vortex_buffer::Buffer;
    use vortex_error::VortexUnwrap;
    use vortex_fastlanes::{BitPackedArray, FoRArray};

    use super::*;
    use crate::bit_unpack::{cuda_bit_unpack, cuda_for_unpack};

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
        let unpacked = cuda_for_unpack_timed(&array, ctx).unwrap();
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

    // #[test]
    // fn test_cuda_bitunpack_u64() {
    //     let primitive_array = PrimitiveArray::new(
    //         (0u64..4096).map(|i| i % 63).collect::<Buffer<_>>(),
    //         Validity::NonNullable,
    //     );
    //     let array = BitPackedArray::encode(primitive_array.as_ref(), 6).vortex_unwrap();
    //     let ctx = CudaContext::new(0).unwrap();
    //     ctx.set_blocking_synchronize().unwrap();
    //     let unpacked = cuda_bit_unpack(&array, ctx).unwrap();
    //     assert_eq!(
    //         primitive_array.as_slice::<u64>(),
    //         unpacked.as_slice::<u64>()
    //     );
    // }
}
