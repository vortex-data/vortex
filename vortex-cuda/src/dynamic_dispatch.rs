// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Host interface for dynamic CUDA kernel dispatch.
//!
//! Provides an API to dynamically dispatch a sequence of decoding operations
//! defined in a plan, to execute them in a single GPU kernel.
//!
//! Each plan has a single source operation to unpack the compressed data into
//! shared memory (e.g., bitunpack). After that scalar operations transform
//! values in registers (e.g., FoR, zigzag, ALP).
//!
//! The public type aliases in this module are to make the bindgen-generated
//! names from `dynamic_dispatch.h` more ergonomic. These types are shared
//! across Rust and CUDA.
//!
//! # Example
//!
//! ```text
//! let plan = DynamicDispatchPlan::new(
//!     SourceOp::bitunpack(6),
//!     &[ScalarOp::frame_of_ref(100), ScalarOp::alp(10.0, 1.0)],
//! );
//! ```

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use vortex_cuda_macros::cuda_tests;

include!(concat!(env!("OUT_DIR"), "/dynamic_dispatch.rs"));

// SAFETY: DynamicDispatchPlan is a C ABI struct with contiguous memory.
unsafe impl cudarc::driver::DeviceRepr for DynamicDispatchPlan {}

/// Enumeration of source operation types.
pub type SourceOpCode = SourceOp_SourceOpCode;

/// Enumeration of scalar operation types.
pub type ScalarOpCode = ScalarOp_ScalarOpCode;

pub const SourceOpCode_BITUNPACK: SourceOpCode = SourceOp_SourceOpCode_BITUNPACK;
pub const ScalarOpCode_FOR: ScalarOpCode = ScalarOp_ScalarOpCode_FOR;
pub const ScalarOpCode_ZIGZAG: ScalarOpCode = ScalarOp_ScalarOpCode_ZIGZAG;
pub const ScalarOpCode_ALP: ScalarOpCode = ScalarOp_ScalarOpCode_ALP;

pub type BitunpackParams = SourceParams_BitunpackParams;
pub type FoRParams = ScalarParams_FoRParams;
pub type AlpParams = ScalarParams_AlpParams;

impl SourceOp {
    /// Create a bitunpack source op with the given bit width.
    pub fn bitunpack(bit_width: u8) -> Self {
        Self {
            op_code: SourceOpCode_BITUNPACK,
            params: SourceParams {
                bitunpack: BitunpackParams { bit_width },
            },
        }
    }
}

impl ScalarOp {
    /// Create a frame-of-reference scalar op that adds the given reference value.
    pub fn frame_of_ref(reference: u64) -> Self {
        Self {
            op_code: ScalarOpCode_FOR,
            params: ScalarParams {
                frame_of_ref: FoRParams { reference },
            },
        }
    }

    /// Create a zigzag decode scalar op.
    pub fn zigzag() -> Self {
        // SAFETY: Zigzag has no parameters; zeroed union is valid.
        Self {
            op_code: ScalarOpCode_ZIGZAG,
            params: unsafe { std::mem::zeroed() },
        }
    }

    /// Create an ALP decode scalar op with the given factors.
    pub fn alp(f: f32, e: f32) -> Self {
        Self {
            op_code: ScalarOpCode_ALP,
            params: ScalarParams {
                alp: AlpParams { f, e },
            },
        }
    }
}

impl DynamicDispatchPlan {
    /// Create a new dispatch plan from a source op and a slice of scalar ops.
    ///
    /// # Panics
    ///
    /// Panics if `scalar_ops.len() > MAX_SCALAR_OPS`.
    #[allow(clippy::cast_possible_truncation)]
    pub fn new(source: SourceOp, scalar_ops: &[ScalarOp]) -> Self {
        assert!(scalar_ops.len() <= MAX_SCALAR_OPS as usize);
        // SAFETY: ScalarOp is a repr(C) union type; zeroed memory is valid for unused slots.
        let mut plan_ops: [ScalarOp; MAX_SCALAR_OPS as usize] = unsafe { std::mem::zeroed() };
        plan_ops[..scalar_ops.len()].copy_from_slice(scalar_ops);
        Self {
            source,
            num_scalar_ops: scalar_ops.len() as u8,
            scalar_ops: plan_ops,
        }
    }
}

#[cuda_tests]
#[allow(clippy::cast_possible_truncation)]
mod tests {
    use std::sync::Arc;

    use cudarc::driver::DevicePtr;
    use cudarc::driver::LaunchConfig;
    use cudarc::driver::PushKernelArg;
    use vortex_alp::ALPFloat;
    use vortex_alp::Exponents;
    use vortex_alp::alp_encode;
    use vortex_array::ToCanonical;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::buffer::BufferHandle;
    use vortex_array::validity::Validity::NonNullable;
    use vortex_buffer::Buffer;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;
    use vortex_fastlanes::BitPackedArray;
    use vortex_fastlanes::FoRArray;
    use vortex_session::VortexSession;

    use super::DynamicDispatchPlan;
    use super::ScalarOp;
    use super::ScalarOpCode_ZIGZAG;
    use super::SourceOp;
    use crate::CudaBufferExt;
    use crate::CudaDeviceBuffer;
    use crate::CudaExecutionCtx;
    use crate::session::CudaSession;

    fn make_bitpacked_array_u32(bit_width: u8, len: usize) -> BitPackedArray {
        let max_val = (1u64 << bit_width).saturating_sub(1);
        let values: Vec<u32> = (0..len)
            .map(|i| ((i as u64) % (max_val + 1)) as u32)
            .collect();
        let primitive = PrimitiveArray::new(Buffer::from(values), NonNullable);
        BitPackedArray::encode(primitive.as_ref(), bit_width)
            .vortex_expect("failed to create BitPacked array")
    }

    fn run_dynamic_dispatch_u32(
        cuda_ctx: &CudaExecutionCtx,
        input_ptr: u64,
        output_len: usize,
        plan: &DynamicDispatchPlan,
    ) -> VortexResult<Vec<u32>> {
        let output_slice = cuda_ctx
            .device_alloc::<u32>(output_len)
            .vortex_expect("alloc output");
        let output_buf = CudaDeviceBuffer::new(output_slice);
        let output_ptr = output_buf.as_view::<u32>().device_ptr(cuda_ctx.stream()).0;

        let device_plan = Arc::new(
            cuda_ctx
                .stream()
                .clone_htod(std::slice::from_ref(plan))
                .expect("copy plan to device"),
        );
        let plan_ptr = device_plan.device_ptr(cuda_ctx.stream()).0;
        let array_len_u64 = output_len as u64;

        cuda_ctx.stream().synchronize().expect("sync");

        let cuda_function = cuda_ctx
            .load_function("dynamic_dispatch", &["u32"])
            .vortex_expect("load kernel");
        let mut launch_builder = cuda_ctx.launch_builder(&cuda_function);
        launch_builder.arg(&input_ptr);
        launch_builder.arg(&output_ptr);
        launch_builder.arg(&array_len_u64);
        launch_builder.arg(&plan_ptr);

        let num_blocks = u32::try_from(output_len.div_ceil(2048))?;
        let config = LaunchConfig {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (64, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            launch_builder.launch(config).expect("kernel launch");
        }

        let host_output: Vec<u32> = cuda_ctx
            .stream()
            .clone_dtoh(&output_buf.as_view::<u32>())
            .expect("copy back");

        Ok(host_output)
    }

    fn run_dynamic_dispatch_f32(
        cuda_ctx: &CudaExecutionCtx,
        input_ptr: u64,
        output_len: usize,
        plan: &DynamicDispatchPlan,
    ) -> VortexResult<Vec<f32>> {
        let result = run_dynamic_dispatch_u32(cuda_ctx, input_ptr, output_len, plan)?;
        // SAFETY: f32 and u32 have identical size and alignment.
        Ok(unsafe { std::mem::transmute::<Vec<u32>, Vec<f32>>(result) })
    }

    fn copy_to_device(
        cuda_ctx: &CudaExecutionCtx,
        bitpacked: &BitPackedArray,
    ) -> VortexResult<(u64, BufferHandle)> {
        let packed = bitpacked.packed().clone();
        let device_input = futures::executor::block_on(cuda_ctx.move_to_device(packed)?)
            .vortex_expect("move to device");
        let ptr = device_input
            .cuda_view::<u32>()
            .vortex_expect("input view")
            .device_ptr(cuda_ctx.stream())
            .0;
        Ok((ptr, device_input))
    }

    #[test]
    fn test_bitunpack() -> VortexResult<()> {
        let bit_width: u8 = 10;
        let len = 3000;

        let max_val = (1u64 << bit_width).saturating_sub(1);
        let expected: Vec<u32> = (0..len)
            .map(|i| ((i as u64) % (max_val + 1)) as u32)
            .collect();

        let bitpacked = make_bitpacked_array_u32(bit_width, len);
        let cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())?;
        let (input_ptr, _device_input) = copy_to_device(&cuda_ctx, &bitpacked)?;

        let plan = DynamicDispatchPlan::new(SourceOp::bitunpack(bit_width), &[]);

        let result = run_dynamic_dispatch_u32(&cuda_ctx, input_ptr, len, &plan)?;
        assert_eq!(result, expected);

        Ok(())
    }

    #[test]
    fn test_bitunpack_for() -> VortexResult<()> {
        let bit_width: u8 = 10;
        let len = 3000;
        let reference: u32 = 42;

        let max_val = (1u64 << bit_width).saturating_sub(1);
        let expected: Vec<u32> = (0..len)
            .map(|i| ((i as u64) % (max_val + 1)) as u32 + reference)
            .collect();

        let bitpacked = make_bitpacked_array_u32(bit_width, len);
        let cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())?;
        let (input_ptr, _device_input) = copy_to_device(&cuda_ctx, &bitpacked)?;

        let plan = DynamicDispatchPlan::new(
            SourceOp::bitunpack(bit_width),
            &[ScalarOp::frame_of_ref(reference as u64)],
        );

        let result = run_dynamic_dispatch_u32(&cuda_ctx, input_ptr, len, &plan)?;
        assert_eq!(result, expected);

        Ok(())
    }

    #[test]
    fn test_bitunpack_for_alp() -> VortexResult<()> {
        let len = 2050;

        let exponents = Exponents { e: 2, f: 0 };
        let floats: Vec<f32> = (0..len)
            .map(|i| <f32 as ALPFloat>::decode_single(10 + (i as i32 % 64), exponents))
            .collect();
        let float_prim = PrimitiveArray::new(Buffer::from(floats.clone()), NonNullable);

        // ALP encode f32 → i32 encoded integers + exponents.
        let alp_array = alp_encode(&float_prim, Some(exponents))?;
        assert!(alp_array.patches().is_none());

        // FOR encode the ALP-encoded i32 integers.
        let for_array = FoRArray::encode(alp_array.encoded().to_primitive())?;
        let reference = i32::try_from(for_array.reference_scalar())? as u32;

        // BitPack the FOR-encoded values.
        let bit_width: u8 = 6;
        let bitpacked = BitPackedArray::encode(for_array.encoded(), bit_width)?;

        let alp_f = <f32 as ALPFloat>::F10[alp_array.exponents().f as usize];
        let alp_e = <f32 as ALPFloat>::IF10[alp_array.exponents().e as usize];

        let cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())?;
        let (input_ptr, _device_input) = copy_to_device(&cuda_ctx, &bitpacked)?;

        let plan = DynamicDispatchPlan::new(
            SourceOp::bitunpack(bit_width),
            &[
                ScalarOp::frame_of_ref(reference as u64),
                ScalarOp::alp(alp_f, alp_e),
            ],
        );

        let result = run_dynamic_dispatch_f32(&cuda_ctx, input_ptr, len, &plan)?;
        assert_eq!(result, floats);

        Ok(())
    }

    #[test]
    fn test_max_scalar_ops() -> VortexResult<()> {
        let bit_width: u8 = 6;
        let len = 2050;
        let references: [u32; 8] = [1, 2, 4, 8, 16, 32, 64, 128];
        let total_reference: u32 = references.iter().sum();

        let max_val = (1u64 << bit_width).saturating_sub(1);
        let expected: Vec<u32> = (0..len)
            .map(|i| ((i as u64) % (max_val + 1)) as u32 + total_reference)
            .collect();

        let bitpacked = make_bitpacked_array_u32(bit_width, len);
        let cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())?;
        let (input_ptr, _device_input) = copy_to_device(&cuda_ctx, &bitpacked)?;

        let scalar_ops: Vec<ScalarOp> = references
            .iter()
            .map(|&r| ScalarOp::frame_of_ref(r as u64))
            .collect();

        let plan = DynamicDispatchPlan::new(SourceOp::bitunpack(bit_width), &scalar_ops);
        assert_eq!(plan.num_scalar_ops, 8);

        let result = run_dynamic_dispatch_u32(&cuda_ctx, input_ptr, len, &plan)?;
        assert_eq!(result, expected);

        Ok(())
    }

    #[test]
    fn test_dynamic_dispatch_plan() {
        let plan = DynamicDispatchPlan::new(
            SourceOp::bitunpack(10),
            &[
                ScalarOp::frame_of_ref(42),
                ScalarOp::zigzag(),
                ScalarOp::alp(10.0, 0.01),
            ],
        );

        assert_eq!(unsafe { plan.source.params.bitunpack.bit_width }, 10);
        assert_eq!(plan.num_scalar_ops, 3);
        assert_eq!(
            unsafe { plan.scalar_ops[0].params.frame_of_ref.reference },
            42
        );
        assert_eq!(plan.scalar_ops[1].op_code, ScalarOpCode_ZIGZAG);
        assert_eq!(unsafe { plan.scalar_ops[2].params.alp.f }, 10.0);
        assert_eq!(unsafe { plan.scalar_ops[2].params.alp.e }, 0.01);
    }
}
