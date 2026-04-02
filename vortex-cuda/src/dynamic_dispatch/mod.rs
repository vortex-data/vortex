// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Host interface for dynamic CUDA kernel dispatch.
//!
//! [`DispatchPlan::new`] walks an encoding tree (e.g., `ALP(FoR(BitPacked))`)
//! in a single pass and returns one of three variants:
//!
//! - [`Fused`](DispatchPlan::Fused) — call [`FusedPlan::materialize`].
//! - [`PartiallyFused`](DispatchPlan::PartiallyFused) — execute pending
//!   subtrees, then call [`FusedPlan::materialize_with_subtrees`].
//! - [`Unfused`](DispatchPlan::Unfused) — fall back to single-kernel dispatch.

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(clippy::cast_possible_truncation)]

use std::borrow::Borrow;
use std::mem::size_of;
use std::slice::from_raw_parts;
use std::sync::Arc;

use cudarc::driver::DevicePtr;
use cudarc::driver::LaunchConfig;
use cudarc::driver::PushKernelArg;
use vortex::array::Canonical;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::buffer::BufferHandle;
use vortex::array::buffer::DeviceBufferExt;
use vortex::array::match_each_unsigned_integer_ptype;
use vortex::array::validity::Validity;
use vortex::buffer::Alignment;
use vortex::buffer::ByteBuffer;
use vortex::buffer::ByteBufferMut;
use vortex::dtype::Nullability;
use vortex::dtype::PType;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_err;

use crate::CudaDeviceBuffer;
use crate::executor::CudaExecutionCtx;

pub(crate) mod plan_builder;
pub use plan_builder::DispatchPlan;
pub use plan_builder::FusedPlan;
pub use plan_builder::MaterializedPlan;

include!(concat!(env!("OUT_DIR"), "/dynamic_dispatch.rs"));

/// Reinterpret a `&T` as a byte slice for serialization into the packed plan.
///
/// # Safety
///
/// The caller must ensure `T` is a `#[repr(C)]` type whose layout is
/// compatible with the C ABI.  All the types we serialise (`PlanHeader`,
/// `PackedStage`, `ScalarOp`) are bindgen-generated `#[repr(C)]` structs.
/// Padding bytes may be uninitialised on the Rust side, but the C reader
/// never inspects them, so the values are irrelevant.
fn as_bytes<T: Sized>(val: &T) -> &[u8] {
    unsafe { from_raw_parts(std::ptr::addr_of!(*val).cast(), size_of::<T>()) }
}

/// A stage used to build a [`CudaDispatchPlan`] on the host side.
///
/// This is NOT a C ABI struct — it exists purely on the Rust side and is
/// serialized into the packed plan byte buffer by [`CudaDispatchPlan::new`].
#[derive(Clone)]
pub struct MaterializedStage {
    /// Device pointer to the input buffer for this stage.
    pub input_ptr: u64,
    /// Byte offset into shared memory where this stage's data is stored.
    pub smem_offset: u32,
    /// Number of elements in this stage.
    pub len: u32,
    /// The source operation that produces the initial values (e.g. load, bitunpack, sequence).
    pub source: SourceOp,
    /// Chain of element-wise scalar operations applied after the source (e.g. frame-of-reference, zigzag, ALP).
    pub scalar_ops: Vec<ScalarOp>,
}

impl MaterializedStage {
    pub fn new(
        input_ptr: u64,
        smem_offset: u32,
        len: u32,
        source: SourceOp,
        scalar_ops: &[ScalarOp],
    ) -> Self {
        Self {
            input_ptr,
            smem_offset,
            len,
            source,
            scalar_ops: scalar_ops.to_vec(),
        }
    }
}

/// Read-only view of a parsed stage from a [`CudaDispatchPlan`].
///
/// Returned by [`CudaDispatchPlan::stage`] for test inspection.
#[derive(Clone)]
pub struct ParsedStage {
    pub input_ptr: u64,
    pub smem_offset: u32,
    pub len: u32,
    pub source: SourceOp,
    pub num_scalar_ops: u8,
    pub scalar_ops: Vec<ScalarOp>,
}

/// A dispatch plan serialized as a packed byte buffer.
///
/// Matching the C-side `PlanHeader` + `PackedStage` ABI in `dynamic_dispatch.h`:
///
/// ```text
/// [PlanHeader]                            — sizeof(PlanHeader) bytes
/// [PackedStage 0][ScalarOp × N0]          — variable
/// [PackedStage 1][ScalarOp × N1]          — variable
/// ...
/// ```
#[derive(Clone)]
pub struct CudaDispatchPlan {
    buffer: ByteBuffer,
}

impl CudaDispatchPlan {
    /// Build a packed plan from a sequence of stages.
    ///
    /// The last stage is the output pipeline; earlier stages are input stages.
    ///
    /// # Panics
    ///
    /// Panics if `stages` is empty or the serialized plan exceeds 65535 bytes.
    pub fn new<I>(stages: I) -> Self
    where
        I: IntoIterator,
        I::Item: Borrow<MaterializedStage>,
    {
        let stages: Vec<MaterializedStage> =
            stages.into_iter().map(|s| s.borrow().clone()).collect();
        assert!(!stages.is_empty());

        let header_size = size_of::<PlanHeader>();
        let stage_header_size = size_of::<PackedStage>();
        let scalar_op_size = size_of::<ScalarOp>();

        // Calculate total size and validate.
        let mut total_size = header_size;
        for stage in &stages {
            total_size += stage_header_size;
            total_size += stage.scalar_ops.len() * scalar_op_size;
        }
        assert!(
            total_size <= u16::MAX as usize,
            "packed plan size {total_size} exceeds u16::MAX"
        );

        let mut buffer = ByteBufferMut::with_capacity_aligned(total_size, Alignment::of::<u32>());

        // Write header.
        let header = PlanHeader {
            num_stages: stages.len() as u8,
            plan_size_bytes: total_size as u16,
        };
        buffer.extend_from_slice(as_bytes(&header));

        // Write each stage header followed by its scalar ops.
        for stage in &stages {
            let packed_stage = PackedStage {
                input_ptr: stage.input_ptr,
                smem_offset: stage.smem_offset,
                len: stage.len,
                source: stage.source,
                num_scalar_ops: stage.scalar_ops.len() as u8,
            };
            buffer.extend_from_slice(as_bytes(&packed_stage));
            for op in &stage.scalar_ops {
                buffer.extend_from_slice(as_bytes(op));
            }
        }

        assert_eq!(buffer.len(), total_size);
        Self {
            buffer: buffer.freeze(),
        }
    }

    /// The raw packed plan bytes, ready for upload to the device.
    pub fn as_bytes(&self) -> &[u8] {
        self.buffer.as_ref()
    }

    /// Number of stages in the plan.
    pub fn num_stages(&self) -> u8 {
        let header: PlanHeader = unsafe { *self.buffer.as_ptr().cast() };
        header.num_stages
    }

    /// Parse and return a read-only view of the stage at `index`.
    ///
    /// # Panics
    ///
    /// Panics if `index >= num_stages()`.
    pub fn stage(&self, index: usize) -> ParsedStage {
        let header_size = size_of::<PlanHeader>();
        let stage_header_size = size_of::<PackedStage>();
        let scalar_op_size = size_of::<ScalarOp>();

        let mut offset = header_size;

        // Skip past stages before `index`.
        for _ in 0..index {
            assert!(offset + stage_header_size <= self.buffer.len());
            let ps: PackedStage = unsafe { *self.buffer.as_ptr().add(offset).cast() };
            offset += stage_header_size + ps.num_scalar_ops as usize * scalar_op_size;
        }

        assert!(offset + stage_header_size <= self.buffer.len());
        let ps: PackedStage = unsafe { *self.buffer.as_ptr().add(offset).cast() };
        offset += stage_header_size;

        let mut scalar_ops = Vec::with_capacity(ps.num_scalar_ops as usize);
        for _ in 0..ps.num_scalar_ops {
            assert!(offset + scalar_op_size <= self.buffer.len());
            let op: ScalarOp = unsafe { *self.buffer.as_ptr().add(offset).cast() };
            scalar_ops.push(op);
            offset += scalar_op_size;
        }

        ParsedStage {
            input_ptr: ps.input_ptr,
            smem_offset: ps.smem_offset,
            len: ps.len,
            source: ps.source,
            num_scalar_ops: ps.num_scalar_ops,
            scalar_ops,
        }
    }
}

impl SourceOp {
    /// Unpack bit-packed data using FastLanes layout.
    ///
    /// `element_offset` (0..1023) is the sub-block position within the first
    /// FastLanes block. The device pointer already accounts for buffer slicing,
    /// but sub-block alignment cannot be expressed as pointer arithmetic on
    /// bit-packed data, so it is passed as a kernel parameter.
    pub fn bitunpack(bit_width: u8, element_offset: u16) -> Self {
        Self {
            op_code: SourceOp_SourceOpCode_BITUNPACK,
            params: SourceParams {
                bitunpack: SourceParams_BitunpackParams {
                    bit_width,
                    element_offset: u32::from(element_offset),
                },
            },
        }
    }

    /// Copy elements verbatim from global memory to shared memory.
    pub fn load() -> Self {
        Self {
            op_code: SourceOp_SourceOpCode_LOAD,
            params: unsafe { std::mem::zeroed() },
        }
    }

    /// Decode run-end encoding.
    ///
    /// # Arguments
    ///
    /// * `ends_smem_offset` - smem region holding run-end endpoints
    /// * `values_smem_offset` - smem region holding per-run values
    /// * `num_runs` - number of runs (length of ends/values)
    /// * `offset` - logical offset for sliced arrays
    pub fn runend(
        ends_smem_offset: u32,
        values_smem_offset: u32,
        num_runs: u64,
        offset: u64,
    ) -> Self {
        Self {
            op_code: SourceOp_SourceOpCode_RUNEND,
            params: SourceParams {
                runend: SourceParams_RunEndParams {
                    ends_smem_offset,
                    values_smem_offset,
                    num_runs,
                    offset,
                },
            },
        }
    }

    /// Generate a linear sequence: `value[i] = base + i * multiplier`.
    /// Used for SequenceArray (e.g. monotonic run-end endpoints).
    pub fn sequence(base: i64, multiplier: i64) -> Self {
        Self {
            op_code: SourceOp_SourceOpCode_SEQUENCE,
            params: SourceParams {
                sequence: SourceParams_SequenceParams { base, multiplier },
            },
        }
    }
}

impl ScalarOp {
    /// Frame-of-reference: add a constant.
    pub fn frame_of_ref(reference: u64) -> Self {
        Self {
            op_code: ScalarOp_ScalarOpCode_FOR,
            params: ScalarParams {
                frame_of_ref: ScalarParams_FoRParams { reference },
            },
        }
    }

    /// Zigzag decode.
    pub fn zigzag() -> Self {
        // SAFETY: Zigzag has no parameters; zeroed union is valid.
        Self {
            op_code: ScalarOp_ScalarOpCode_ZIGZAG,
            params: unsafe { std::mem::zeroed() },
        }
    }

    /// ALP floating-point decode.
    pub fn alp(f: f32, e: f32) -> Self {
        Self {
            op_code: ScalarOp_ScalarOpCode_ALP,
            params: ScalarParams {
                alp: ScalarParams_AlpParams { f, e },
            },
        }
    }

    /// Dictionary gather: use current value as index into decoded values
    /// in shared memory (populated by an earlier input stage).
    pub fn dict(values_smem_offset: u32) -> Self {
        Self {
            op_code: ScalarOp_ScalarOpCode_DICT,
            params: ScalarParams {
                dict: ScalarParams_DictParams { values_smem_offset },
            },
        }
    }
}

impl MaterializedPlan {
    pub fn execute(
        self,
        output_ptype: PType,
        len: usize,
        ctx: &mut CudaExecutionCtx,
    ) -> VortexResult<Canonical> {
        let unsigned_ptype = match output_ptype {
            PType::U8 | PType::I8 => PType::U8,
            PType::U16 | PType::I16 => PType::U16,
            PType::U32 | PType::I32 | PType::F32 => PType::U32,
            PType::U64 | PType::I64 => PType::U64,
            other => vortex_bail!("dynamic dispatch does not support PType {:?}", other),
        };
        match_each_unsigned_integer_ptype!(unsigned_ptype, |T| {
            self.execute_typed::<T>(output_ptype, len, ctx)
        })
    }

    fn execute_typed<T>(
        self,
        output_ptype: PType,
        len: usize,
        ctx: &mut CudaExecutionCtx,
    ) -> VortexResult<Canonical>
    where
        T: cudarc::driver::DeviceRepr + vortex::dtype::NativePType,
    {
        if len == 0 {
            return Ok(Canonical::Primitive(PrimitiveArray::empty::<T>(
                Nullability::NonNullable,
            )));
        }

        let output_buf = CudaDeviceBuffer::new(ctx.device_alloc::<T>(len.next_multiple_of(1024))?);

        // Upload the packed plan bytes to the device.
        let device_plan = Arc::new(
            ctx.stream()
                .clone_htod(self.dispatch_plan.as_bytes())
                .map_err(|e| vortex_err!("copy plan to device: {e}"))?,
        );

        let cuda_function = ctx.load_function("dynamic_dispatch", &[T::PTYPE])?;
        let num_blocks = u32::try_from(len.div_ceil(2048))?;
        let config = LaunchConfig {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (64, 1, 1),
            shared_mem_bytes: self.shared_mem_bytes,
        };

        let output_ptr = output_buf.offset_ptr();
        let plan_ptr = device_plan.device_ptr(ctx.stream()).0;
        let array_len_u64 = len as u64;

        ctx.launch_kernel_config(&cuda_function, config, len, |args| {
            args.arg(&output_ptr);
            args.arg(&array_len_u64);
            args.arg(&plan_ptr);
        })?;

        Ok(Canonical::Primitive(PrimitiveArray::from_buffer_handle(
            BufferHandle::new_device(output_buf.slice_typed::<T>(0..len)),
            output_ptype,
            Validity::NonNullable,
        )))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use cudarc::driver::DevicePtr;
    use cudarc::driver::LaunchConfig;
    use cudarc::driver::PushKernelArg;
    use rstest::rstest;
    use vortex::array::IntoArray;
    use vortex::array::ToCanonical;
    use vortex::array::arrays::DictArray;
    use vortex::array::arrays::PrimitiveArray;
    use vortex::array::scalar::Scalar;
    use vortex::array::validity::Validity::NonNullable;
    use vortex::buffer::Buffer;
    use vortex::dtype::PType;
    use vortex::encodings::alp::ALP;
    use vortex::encodings::alp::ALPFloat;
    use vortex::encodings::alp::Exponents;
    use vortex::encodings::alp::alp_encode;
    use vortex::encodings::fastlanes::BitPacked;
    use vortex::encodings::fastlanes::BitPackedArray;
    use vortex::encodings::fastlanes::FoR;
    use vortex::encodings::runend::RunEnd;
    use vortex::encodings::zigzag::ZigZag;
    use vortex::error::VortexExpect;
    use vortex::error::VortexResult;
    use vortex::session::VortexSession;

    use super::CudaDispatchPlan;
    use super::DispatchPlan;
    use super::MaterializedStage;
    use super::SMEM_TILE_SIZE;
    use super::ScalarOp;
    use super::SourceOp;
    use super::*;
    use crate::CudaBufferExt;
    use crate::CudaDeviceBuffer;
    use crate::CudaExecutionCtx;
    use crate::session::CudaSession;

    fn bitpacked_array_u32(bit_width: u8, len: usize) -> BitPackedArray {
        let max_val = (1u64 << bit_width).saturating_sub(1);
        let values: Vec<u32> = (0..len)
            .map(|i| ((i as u64) % (max_val + 1)) as u32)
            .collect();
        let primitive = PrimitiveArray::new(Buffer::from(values), NonNullable);
        BitPacked::encode(&primitive.into_array(), bit_width)
            .vortex_expect("failed to create BitPacked array")
    }

    fn dispatch_plan(
        array: &vortex::array::ArrayRef,
        ctx: &CudaExecutionCtx,
    ) -> VortexResult<MaterializedPlan> {
        match DispatchPlan::new(array)? {
            DispatchPlan::Fused(plan) => plan.materialize(ctx),
            _ => vortex_bail!("array encoding not fusable"),
        }
    }

    #[crate::test]
    fn test_max_scalar_ops() -> VortexResult<()> {
        let bit_width: u8 = 6;
        let len = 2050;
        let references: [u32; 4] = [1, 2, 4, 8];
        let total_reference: u32 = references.iter().sum();

        let max_val = (1u64 << bit_width).saturating_sub(1);
        let expected: Vec<u32> = (0..len)
            .map(|i| ((i as u64) % (max_val + 1)) as u32 + total_reference)
            .collect();

        let bitpacked = bitpacked_array_u32(bit_width, len);
        let cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())?;
        let packed = bitpacked.packed().clone();
        let device_input = futures::executor::block_on(cuda_ctx.ensure_on_device(packed))?;
        let input_ptr = device_input.cuda_device_ptr()?;

        let scalar_ops: Vec<ScalarOp> = references
            .iter()
            .map(|&r| ScalarOp::frame_of_ref(r as u64))
            .collect();

        let plan = CudaDispatchPlan::new([MaterializedStage::new(
            input_ptr,
            0,
            len as u32,
            SourceOp::bitunpack(bit_width, 0),
            &scalar_ops,
        )]);
        assert_eq!(plan.stage(0).num_scalar_ops, 4);

        let actual = run_dynamic_dispatch_plan(&cuda_ctx, len, &plan, SMEM_TILE_SIZE * 4)?;
        assert_eq!(actual, expected);

        Ok(())
    }

    #[crate::test]
    fn test_plan_structure() {
        // Stage 0: input dict values (BP→FoR) into smem[0..256)
        // Stage 1: output codes (BP→FoR→DICT) into smem[256..1280), gather from smem[0]
        let plan = CudaDispatchPlan::new([
            MaterializedStage::new(
                0xAAAA,
                0,
                256,
                SourceOp::bitunpack(4, 0),
                &[ScalarOp::frame_of_ref(10)],
            ),
            MaterializedStage::new(
                0xBBBB,
                256,
                1024,
                SourceOp::bitunpack(6, 0),
                &[ScalarOp::frame_of_ref(42), ScalarOp::dict(0)],
            ),
        ]);

        assert_eq!(plan.num_stages(), 2);

        // Input stage
        let s0 = plan.stage(0);
        assert_eq!(s0.smem_offset, 0);
        assert_eq!(s0.len, 256);
        assert_eq!(s0.input_ptr, 0xAAAA);

        // Output stage
        let s1 = plan.stage(1);
        assert_eq!(s1.smem_offset, 256);
        assert_eq!(s1.len, SMEM_TILE_SIZE);
        assert_eq!(s1.input_ptr, 0xBBBB);
        assert_eq!(s1.num_scalar_ops, 2);
        assert_eq!(
            unsafe { s1.scalar_ops[1].params.dict.values_smem_offset },
            0
        );
    }

    /// Copy a raw u32 slice to device memory and return (device_ptr, handle).
    fn copy_raw_to_device(
        cuda_ctx: &CudaExecutionCtx,
        data: &[u32],
    ) -> VortexResult<(u64, Arc<cudarc::driver::CudaSlice<u32>>)> {
        let device_buf = Arc::new(cuda_ctx.stream().clone_htod(data).expect("htod"));
        let (ptr, _) = device_buf.device_ptr(cuda_ctx.stream());
        Ok((ptr, device_buf))
    }

    #[crate::test]
    fn test_load_for_zigzag_alp() -> VortexResult<()> {
        // Max scalar ops depth with LOAD source: LOAD → FoR → ZigZag → ALP
        // (Exercises all four scalar op types without DICT)
        let len = 2048;
        let reference = 5u32;
        let alp_f = 10.0f32;
        let alp_e = 0.1f32;

        let data: Vec<u32> = (0..len).map(|i| (i as u32) % 64).collect();
        let expected: Vec<u32> = data
            .iter()
            .map(|&v| {
                let after_for = v + reference;
                let after_zz = (after_for >> 1) ^ (0u32.wrapping_sub(after_for & 1));
                let float_val = (after_zz as i32) as f32 * alp_f * alp_e;
                float_val.to_bits()
            })
            .collect();

        let cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())?;
        let (input_ptr, _di) = copy_raw_to_device(&cuda_ctx, &data)?;

        let plan = CudaDispatchPlan::new([MaterializedStage::new(
            input_ptr,
            0,
            len as u32,
            SourceOp::load(),
            &[
                ScalarOp::frame_of_ref(reference as u64),
                ScalarOp::zigzag(),
                ScalarOp::alp(alp_f, alp_e),
            ],
        )]);

        let actual = run_dynamic_dispatch_plan(&cuda_ctx, len, &plan, SMEM_TILE_SIZE * 4)?;
        assert_eq!(actual, expected);

        Ok(())
    }

    /// Runs a dynamic dispatch plan on the GPU.
    fn run_dynamic_dispatch_plan(
        cuda_ctx: &CudaExecutionCtx,
        output_len: usize,
        plan: &CudaDispatchPlan,
        shared_mem_bytes: u32,
    ) -> VortexResult<Vec<u32>> {
        let output_slice = cuda_ctx
            .device_alloc::<u32>(output_len)
            .vortex_expect("alloc output");
        let output_buf = CudaDeviceBuffer::new(output_slice);
        let output_view = output_buf.as_view::<u32>();
        let (output_ptr, record_output) = output_view.device_ptr(cuda_ctx.stream());

        let device_plan = Arc::new(
            cuda_ctx
                .stream()
                .clone_htod(plan.as_bytes())
                .expect("copy plan to device"),
        );
        let (plan_ptr, record_plan) = device_plan.device_ptr(cuda_ctx.stream());
        let array_len_u64 = output_len as u64;

        cuda_ctx.stream().synchronize().expect("sync");

        let cuda_function = cuda_ctx
            .load_function("dynamic_dispatch", &[PType::U32])
            .vortex_expect("load kernel");
        let mut launch_builder = cuda_ctx.launch_builder(&cuda_function);
        launch_builder.arg(&output_ptr);
        launch_builder.arg(&array_len_u64);
        launch_builder.arg(&plan_ptr);

        let num_blocks = u32::try_from(output_len.div_ceil(2048))?;
        let config = LaunchConfig {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (64, 1, 1),
            shared_mem_bytes,
        };
        unsafe {
            launch_builder.launch(config).expect("kernel launch");
        }
        drop((record_output, record_plan));

        Ok(cuda_ctx
            .stream()
            .clone_dtoh(&output_buf.as_view::<u32>())
            .expect("copy back"))
    }

    fn run_dispatch_plan_f32(
        cuda_ctx: &CudaExecutionCtx,
        output_len: usize,
        plan: &CudaDispatchPlan,
        shared_mem_bytes: u32,
    ) -> VortexResult<Vec<f32>> {
        let actual = run_dynamic_dispatch_plan(cuda_ctx, output_len, plan, shared_mem_bytes)?;
        // SAFETY: f32 and u32 have identical size and alignment.
        Ok(unsafe { std::mem::transmute::<Vec<u32>, Vec<f32>>(actual) })
    }

    #[crate::test]
    fn test_bitpacked() -> VortexResult<()> {
        let bit_width: u8 = 10;
        let len = 3000;
        let max_val = (1u64 << bit_width).saturating_sub(1);
        let expected: Vec<u32> = (0..len)
            .map(|i| ((i as u64) % (max_val + 1)) as u32)
            .collect();

        let bp = bitpacked_array_u32(bit_width, len);
        let cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())?;
        let plan = dispatch_plan(&bp.into_array(), &cuda_ctx)?;

        let actual =
            run_dynamic_dispatch_plan(&cuda_ctx, len, &plan.dispatch_plan, plan.shared_mem_bytes)?;
        assert_eq!(actual, expected);

        Ok(())
    }

    #[crate::test]
    fn test_for_bitpacked() -> VortexResult<()> {
        let bit_width: u8 = 6;
        let len = 3000;
        let reference = 42u32;
        let max_val = (1u64 << bit_width).saturating_sub(1);

        let raw: Vec<u32> = (0..len)
            .map(|i| ((i as u64) % (max_val + 1)) as u32)
            .collect();
        let expected: Vec<u32> = raw.iter().map(|&v| v + reference).collect();

        let bp = bitpacked_array_u32(bit_width, len);
        let for_arr = FoR::try_new(bp.into_array(), Scalar::from(reference))?;

        let cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())?;
        let plan = dispatch_plan(&for_arr.into_array(), &cuda_ctx)?;

        let actual =
            run_dynamic_dispatch_plan(&cuda_ctx, len, &plan.dispatch_plan, plan.shared_mem_bytes)?;
        assert_eq!(actual, expected);

        Ok(())
    }

    #[crate::test]
    fn test_runend() -> VortexResult<()> {
        let ends: Vec<u32> = vec![1000, 2000, 3000];
        let values: Vec<u32> = vec![10, 20, 30];
        let len = 3000;

        let mut expected = Vec::with_capacity(len);
        for i in 0..len {
            let run = ends.iter().position(|&e| (i as u32) < e).unwrap();
            expected.push(values[run]);
        }

        let ends_arr = PrimitiveArray::new(Buffer::from(ends), NonNullable).into_array();
        let values_arr = PrimitiveArray::new(Buffer::from(values), NonNullable).into_array();
        let re = RunEnd::new(ends_arr, values_arr);

        let cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())?;
        let plan = dispatch_plan(&re.into_array(), &cuda_ctx)?;

        let actual =
            run_dynamic_dispatch_plan(&cuda_ctx, len, &plan.dispatch_plan, plan.shared_mem_bytes)?;
        assert_eq!(actual, expected);

        Ok(())
    }

    #[crate::test]
    fn test_dict_for_bp_values_bp_codes() -> VortexResult<()> {
        // Dict where both codes and values are BitPacked+FoR.
        let dict_reference = 1_000_000u32;
        let dict_residuals: Vec<u32> = (0..64).collect();
        let dict_expected: Vec<u32> = dict_residuals.iter().map(|&r| r + dict_reference).collect();
        let dict_size = dict_residuals.len();

        let len = 3000;
        let codes: Vec<u32> = (0..len).map(|i| (i % dict_size) as u32).collect();
        let expected: Vec<u32> = codes.iter().map(|&c| dict_expected[c as usize]).collect();

        // BitPack+FoR the dict values
        let dict_prim = PrimitiveArray::new(Buffer::from(dict_residuals), NonNullable);
        let dict_bp = BitPacked::encode(&dict_prim.into_array(), 6)?;
        let dict_for = FoR::try_new(dict_bp.into_array(), Scalar::from(dict_reference))?;

        // BitPack the codes
        let codes_prim = PrimitiveArray::new(Buffer::from(codes), NonNullable);
        let codes_bp = BitPacked::encode(&codes_prim.into_array(), 6)?;

        let dict = DictArray::try_new(codes_bp.into_array(), dict_for.into_array())?;

        let cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())?;
        let plan = dispatch_plan(&dict.into_array(), &cuda_ctx)?;

        let actual =
            run_dynamic_dispatch_plan(&cuda_ctx, len, &plan.dispatch_plan, plan.shared_mem_bytes)?;
        assert_eq!(actual, expected);

        Ok(())
    }

    #[crate::test]
    fn test_alp_for_bitpacked() -> VortexResult<()> {
        // ALP(FoR(BitPacked)): encode each layer, then reassemble the tree
        // bottom-up because encode() methods produce flat outputs.
        let len = 3000;
        let exponents = Exponents { e: 2, f: 0 };
        let floats: Vec<f32> = (0..len)
            .map(|i| <f32 as ALPFloat>::decode_single(10 + (i as i32 % 64), exponents))
            .collect();
        let float_prim = PrimitiveArray::new(Buffer::from(floats.clone()), NonNullable);

        let alp = alp_encode(&float_prim, Some(exponents))?;
        assert!(alp.patches().is_none());
        let for_arr = FoR::encode(alp.encoded().to_primitive())?;
        let bp = BitPacked::encode(for_arr.encoded(), 6)?;

        let tree = ALP::new(
            FoR::try_new(bp.into_array(), for_arr.reference_scalar().clone())?.into_array(),
            exponents,
            None,
        );

        let cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())?;
        let plan = dispatch_plan(&tree.into_array(), &cuda_ctx)?;

        let actual =
            run_dispatch_plan_f32(&cuda_ctx, len, &plan.dispatch_plan, plan.shared_mem_bytes)?;
        assert_eq!(actual, floats);

        Ok(())
    }

    #[crate::test]
    fn test_zigzag_bitpacked() -> VortexResult<()> {
        // ZigZag(BitPacked): unpack then zigzag-decode.
        let bit_width: u8 = 4;
        let len = 3000;
        let max_val = (1u64 << bit_width).saturating_sub(1);

        let raw: Vec<u32> = (0..len)
            .map(|i| ((i as u64) % (max_val + 1)) as u32)
            .collect();
        let expected: Vec<u32> = raw
            .iter()
            .map(|&v| (v >> 1) ^ (0u32.wrapping_sub(v & 1)))
            .collect();

        let prim = PrimitiveArray::new(Buffer::from(raw), NonNullable);
        let bp = BitPacked::encode(&prim.into_array(), bit_width)?;
        let zz = ZigZag::try_new(bp.into_array())?;

        let cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())?;
        let plan = dispatch_plan(&zz.into_array(), &cuda_ctx)?;

        let actual =
            run_dynamic_dispatch_plan(&cuda_ctx, len, &plan.dispatch_plan, plan.shared_mem_bytes)?;
        assert_eq!(actual, expected);

        Ok(())
    }

    #[crate::test]
    fn test_for_runend() -> VortexResult<()> {
        // FoR(RunEnd): expand runs then add constant.
        let ends: Vec<u32> = vec![500, 1000, 1500, 2000, 2500, 3000];
        let values: Vec<u32> = vec![1, 2, 3, 4, 5, 6];
        let len = 3000;
        let reference = 1000u32;

        let mut expected = Vec::with_capacity(len);
        for i in 0..len {
            let run = ends.iter().position(|&e| (i as u32) < e).unwrap();
            expected.push(values[run] + reference);
        }

        let ends_arr = PrimitiveArray::new(Buffer::from(ends), NonNullable).into_array();
        let values_arr = PrimitiveArray::new(Buffer::from(values), NonNullable).into_array();
        let re = RunEnd::new(ends_arr, values_arr);
        let for_arr = FoR::try_new(re.into_array(), Scalar::from(reference))?;

        let cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())?;
        let plan = dispatch_plan(&for_arr.into_array(), &cuda_ctx)?;

        let actual =
            run_dynamic_dispatch_plan(&cuda_ctx, len, &plan.dispatch_plan, plan.shared_mem_bytes)?;
        assert_eq!(actual, expected);

        Ok(())
    }

    #[crate::test]
    fn test_for_dict() -> VortexResult<()> {
        // FoR(Dict(codes=Primitive, values=Primitive)): gather then add constant.
        let dict_values: Vec<u32> = vec![100, 200, 300, 400];
        let dict_size = dict_values.len();
        let reference = 5000u32;
        let len = 3000;

        let codes: Vec<u32> = (0..len).map(|i| (i % dict_size) as u32).collect();
        let expected: Vec<u32> = codes
            .iter()
            .map(|&c| dict_values[c as usize] + reference)
            .collect();

        let codes_prim = PrimitiveArray::new(Buffer::from(codes), NonNullable);
        let values_prim = PrimitiveArray::new(Buffer::from(dict_values), NonNullable);
        let dict = DictArray::try_new(codes_prim.into_array(), values_prim.into_array())?;
        let for_arr = FoR::try_new(dict.into_array(), Scalar::from(reference))?;

        let cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())?;
        let plan = dispatch_plan(&for_arr.into_array(), &cuda_ctx)?;

        let actual =
            run_dynamic_dispatch_plan(&cuda_ctx, len, &plan.dispatch_plan, plan.shared_mem_bytes)?;
        assert_eq!(actual, expected);

        Ok(())
    }

    #[crate::test]
    fn test_dict_for_bp_codes() -> VortexResult<()> {
        // Dict(codes=FoR(BitPacked), values=primitive)
        let dict_values: Vec<u32> = (0..8).map(|i| i * 1000 + 7).collect();
        let dict_size = dict_values.len();
        let len = 3000;
        let codes: Vec<u32> = (0..len).map(|i| (i % dict_size) as u32).collect();
        let expected: Vec<u32> = codes.iter().map(|&c| dict_values[c as usize]).collect();

        // BitPack codes, then wrap in FoR (reference=0 so values unchanged)
        let bit_width: u8 = 3;
        let codes_prim = PrimitiveArray::new(Buffer::from(codes), NonNullable);
        let codes_bp = BitPacked::encode(&codes_prim.into_array(), bit_width)?;
        let codes_for = FoR::try_new(codes_bp.into_array(), Scalar::from(0u32))?;

        let values_prim = PrimitiveArray::new(Buffer::from(dict_values), NonNullable);
        let dict = DictArray::try_new(codes_for.into_array(), values_prim.into_array())?;

        let cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())?;
        let plan = dispatch_plan(&dict.into_array(), &cuda_ctx)?;

        let actual =
            run_dynamic_dispatch_plan(&cuda_ctx, len, &plan.dispatch_plan, plan.shared_mem_bytes)?;
        assert_eq!(actual, expected);

        Ok(())
    }

    #[crate::test]
    fn test_dict_primitive_values_bp_codes() -> VortexResult<()> {
        let dict_values: Vec<u32> = vec![100, 200, 300, 400];
        let dict_size = dict_values.len();
        let len = 3000;
        let codes: Vec<u32> = (0..len).map(|i| (i % dict_size) as u32).collect();
        let expected: Vec<u32> = codes.iter().map(|&c| dict_values[c as usize]).collect();

        let bit_width: u8 = 2;
        let codes_prim = PrimitiveArray::new(Buffer::from(codes), NonNullable);
        let codes_bp = BitPacked::encode(&codes_prim.into_array(), bit_width)?;
        let values_prim = PrimitiveArray::new(Buffer::from(dict_values), NonNullable);

        let dict = DictArray::try_new(codes_bp.into_array(), values_prim.into_array())?;

        let cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())?;
        let plan = dispatch_plan(&dict.into_array(), &cuda_ctx)?;

        let actual =
            run_dynamic_dispatch_plan(&cuda_ctx, len, &plan.dispatch_plan, plan.shared_mem_bytes)?;
        assert_eq!(actual, expected);

        Ok(())
    }

    #[crate::test]
    fn test_dict_mismatched_ptypes_rejected() -> VortexResult<()> {
        let dict_values: Vec<u32> = vec![100, 200, 300, 400];
        let len = 3000;
        let codes: Vec<u8> = (0..len).map(|i| (i % dict_values.len()) as u8).collect();

        let codes_prim = PrimitiveArray::new(Buffer::from(codes), NonNullable);
        let values_prim = PrimitiveArray::new(Buffer::from(dict_values), NonNullable);
        let dict = DictArray::try_new(codes_prim.into_array(), values_prim.into_array())?;

        // DispatchPlan::new should return Unfused because u8 codes != u32 values in byte width.
        assert!(matches!(
            DispatchPlan::new(&dict.into_array())?,
            DispatchPlan::Unfused
        ));

        Ok(())
    }

    #[crate::test]
    fn test_runend_mismatched_ptypes_rejected() -> VortexResult<()> {
        let ends: Vec<u64> = vec![1000, 2000, 3000];
        let values: Vec<i32> = vec![10, 20, 30];

        let ends_arr = PrimitiveArray::new(Buffer::from(ends), NonNullable).into_array();
        let values_arr = PrimitiveArray::new(Buffer::from(values), NonNullable).into_array();
        let re = RunEnd::new(ends_arr, values_arr);

        // DispatchPlan::new should return Unfused because u64 ends != i32 values in byte width.
        assert!(matches!(
            DispatchPlan::new(&re.into_array())?,
            DispatchPlan::Unfused
        ));

        Ok(())
    }

    #[rstest]
    #[case(0, 1024)]
    #[case(0, 3000)]
    #[case(0, 4096)]
    #[case(500, 600)]
    #[case(500, 1024)]
    #[case(500, 2048)]
    #[case(500, 4500)]
    #[case(777, 3333)]
    #[case(1024, 2048)]
    #[case(1024, 4096)]
    #[case(1500, 3500)]
    #[case(2048, 4096)]
    #[case(2500, 4500)]
    #[case(3333, 4444)]
    #[crate::test]
    fn test_sliced_primitive(
        #[case] slice_start: usize,
        #[case] slice_end: usize,
    ) -> VortexResult<()> {
        let len = 5000;
        let data: Vec<u32> = (0..len).map(|i| (i * 7) % 1000).collect();

        let prim = PrimitiveArray::new(Buffer::from(data.clone()), NonNullable);

        let sliced = prim.into_array().slice(slice_start..slice_end)?;

        let expected: Vec<u32> = data[slice_start..slice_end].to_vec();

        let cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())?;
        let plan = dispatch_plan(&sliced, &cuda_ctx)?;

        let actual = run_dynamic_dispatch_plan(
            &cuda_ctx,
            expected.len(),
            &plan.dispatch_plan,
            plan.shared_mem_bytes,
        )?;
        assert_eq!(actual, expected);

        Ok(())
    }

    #[rstest]
    #[case(0, 1024)]
    #[case(0, 3000)]
    #[case(0, 4096)]
    #[case(500, 600)]
    #[case(500, 1024)]
    #[case(500, 2048)]
    #[case(500, 4500)]
    #[case(777, 3333)]
    #[case(1024, 2048)]
    #[case(1024, 4096)]
    #[case(1500, 3500)]
    #[case(2048, 4096)]
    #[case(2500, 4500)]
    #[case(3333, 4444)]
    #[crate::test]
    fn test_sliced_zigzag_bitpacked(
        #[case] slice_start: usize,
        #[case] slice_end: usize,
    ) -> VortexResult<()> {
        let bit_width = 10u8;
        let max_val = (1u32 << bit_width) - 1;
        let len = 5000;

        let raw: Vec<u32> = (0..len).map(|i| (i as u32) % max_val).collect();
        let all_decoded: Vec<u32> = raw
            .iter()
            .map(|&v| (v >> 1) ^ (0u32.wrapping_sub(v & 1)))
            .collect();

        let prim = PrimitiveArray::new(Buffer::from(raw), NonNullable);
        let bp = BitPacked::encode(&prim.into_array(), bit_width)?;
        let zz = ZigZag::try_new(bp.into_array())?;

        let sliced = zz.into_array().slice(slice_start..slice_end)?;
        let expected: Vec<u32> = all_decoded[slice_start..slice_end].to_vec();

        let cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())?;
        let plan = dispatch_plan(&sliced, &cuda_ctx)?;

        let actual = run_dynamic_dispatch_plan(
            &cuda_ctx,
            expected.len(),
            &plan.dispatch_plan,
            plan.shared_mem_bytes,
        )?;
        assert_eq!(actual, expected);

        Ok(())
    }

    #[rstest]
    #[case(0, 1024)]
    #[case(0, 3000)]
    #[case(0, 4096)]
    #[case(500, 600)]
    #[case(500, 1024)]
    #[case(500, 2048)]
    #[case(500, 4500)]
    #[case(777, 3333)]
    #[case(1024, 2048)]
    #[case(1024, 4096)]
    #[case(1500, 3500)]
    #[case(2048, 4096)]
    #[case(2500, 4500)]
    #[case(3333, 4444)]
    #[crate::test]
    fn test_sliced_dict_with_primitive_codes(
        #[case] slice_start: usize,
        #[case] slice_end: usize,
    ) -> VortexResult<()> {
        let dict_values: Vec<u32> = vec![100, 200, 300, 400, 500];
        let dict_size = dict_values.len();
        let len = 5000;
        let codes: Vec<u32> = (0..len).map(|i| (i % dict_size) as u32).collect();

        let codes_prim = PrimitiveArray::new(Buffer::from(codes.clone()), NonNullable);
        let values_prim = PrimitiveArray::new(Buffer::from(dict_values.clone()), NonNullable);
        let dict = DictArray::try_new(codes_prim.into_array(), values_prim.into_array())?;

        let sliced = dict.into_array().slice(slice_start..slice_end)?;

        let expected: Vec<u32> = codes[slice_start..slice_end]
            .iter()
            .map(|&c| dict_values[c as usize])
            .collect();

        let cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())?;
        let plan = dispatch_plan(&sliced, &cuda_ctx)?;

        let actual = run_dynamic_dispatch_plan(
            &cuda_ctx,
            expected.len(),
            &plan.dispatch_plan,
            plan.shared_mem_bytes,
        )?;
        assert_eq!(actual, expected);

        Ok(())
    }

    #[rstest]
    #[case(0, 1024)]
    #[case(0, 3000)]
    #[case(0, 4096)]
    #[case(500, 600)]
    #[case(500, 1024)]
    #[case(500, 2048)]
    #[case(500, 4500)]
    #[case(777, 3333)]
    #[case(1024, 2048)]
    #[case(1024, 4096)]
    #[case(1500, 3500)]
    #[case(2048, 4096)]
    #[case(2500, 4500)]
    #[case(3333, 4444)]
    #[crate::test]
    fn test_sliced_bitpacked(
        #[case] slice_start: usize,
        #[case] slice_end: usize,
    ) -> VortexResult<()> {
        let bit_width = 10u8;
        let max_val = (1u32 << bit_width) - 1;
        let len = 5000;

        let data: Vec<u32> = (0..len).map(|i| (i as u32) % max_val).collect();
        let prim = PrimitiveArray::new(Buffer::from(data.clone()), NonNullable);
        let bp = BitPacked::encode(&prim.into_array(), bit_width)?;

        let sliced = bp.into_array().slice(slice_start..slice_end)?;
        let expected: Vec<u32> = data[slice_start..slice_end].to_vec();

        let cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())?;
        let plan = dispatch_plan(&sliced, &cuda_ctx)?;

        let actual = run_dynamic_dispatch_plan(
            &cuda_ctx,
            expected.len(),
            &plan.dispatch_plan,
            plan.shared_mem_bytes,
        )?;
        assert_eq!(actual, expected);

        Ok(())
    }

    #[rstest]
    #[case(0, 1024)]
    #[case(0, 3000)]
    #[case(0, 4096)]
    #[case(500, 600)]
    #[case(500, 1024)]
    #[case(500, 2048)]
    #[case(500, 4500)]
    #[case(777, 3333)]
    #[case(1024, 2048)]
    #[case(1024, 4096)]
    #[case(1500, 3500)]
    #[case(2048, 4096)]
    #[case(2500, 4500)]
    #[case(3333, 4444)]
    #[crate::test]
    fn test_sliced_for_bitpacked(
        #[case] slice_start: usize,
        #[case] slice_end: usize,
    ) -> VortexResult<()> {
        let reference = 100u32;
        let bit_width = 10u8;
        let max_val = (1u32 << bit_width) - 1;
        let len = 5000;

        let encoded_data: Vec<u32> = (0..len).map(|i| (i as u32) % max_val).collect();
        let prim = PrimitiveArray::new(Buffer::from(encoded_data.clone()), NonNullable);
        let bp = BitPacked::encode(&prim.into_array(), bit_width)?;
        let for_arr = FoR::try_new(bp.into_array(), Scalar::from(reference))?;

        let all_decoded: Vec<u32> = encoded_data.iter().map(|&v| v + reference).collect();

        let sliced = for_arr.into_array().slice(slice_start..slice_end)?;
        let expected: Vec<u32> = all_decoded[slice_start..slice_end].to_vec();

        let cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())?;
        let plan = dispatch_plan(&sliced, &cuda_ctx)?;

        let actual = run_dynamic_dispatch_plan(
            &cuda_ctx,
            expected.len(),
            &plan.dispatch_plan,
            plan.shared_mem_bytes,
        )?;
        assert_eq!(actual, expected);

        Ok(())
    }

    #[rstest]
    #[case(0, 1024)]
    #[case(0, 3000)]
    #[case(0, 4096)]
    #[case(500, 600)]
    #[case(500, 1024)]
    #[case(500, 2048)]
    #[case(500, 4500)]
    #[case(777, 3333)]
    #[case(1024, 2048)]
    #[case(1024, 4096)]
    #[case(1500, 3500)]
    #[case(2048, 4096)]
    #[case(2500, 4500)]
    #[case(3333, 4444)]
    #[crate::test]
    fn test_sliced_dict_for_bp_values_bp_codes(
        #[case] slice_start: usize,
        #[case] slice_end: usize,
    ) -> VortexResult<()> {
        let dict_reference = 1_000_000u32;
        let dict_residuals: Vec<u32> = (0..64).collect();
        let dict_expected: Vec<u32> = dict_residuals.iter().map(|&r| r + dict_reference).collect();
        let dict_size = dict_residuals.len();

        let len = 5000;
        let codes: Vec<u32> = (0..len).map(|i| (i % dict_size) as u32).collect();
        let all_decoded: Vec<u32> = codes.iter().map(|&c| dict_expected[c as usize]).collect();

        // BitPack+FoR the dict values
        let dict_prim = PrimitiveArray::new(Buffer::from(dict_residuals), NonNullable);
        let dict_bp = BitPacked::encode(&dict_prim.into_array(), 6)?;
        let dict_for = FoR::try_new(dict_bp.into_array(), Scalar::from(dict_reference))?;

        // BitPack the codes
        let codes_prim = PrimitiveArray::new(Buffer::from(codes), NonNullable);
        let codes_bp = BitPacked::encode(&codes_prim.into_array(), 6)?;

        let dict = DictArray::try_new(codes_bp.into_array(), dict_for.into_array())?;

        let sliced = dict.into_array().slice(slice_start..slice_end)?;
        let expected: Vec<u32> = all_decoded[slice_start..slice_end].to_vec();

        let cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())?;
        let plan = dispatch_plan(&sliced, &cuda_ctx)?;

        let actual = run_dynamic_dispatch_plan(
            &cuda_ctx,
            expected.len(),
            &plan.dispatch_plan,
            plan.shared_mem_bytes,
        )?;
        assert_eq!(actual, expected);

        Ok(())
    }

    #[rstest]
    #[case(0u32, 1u32, 100)]
    #[case(5u32, 3u32, 2048)]
    #[case(0u32, 1u32, 4096)]
    #[case(100u32, 7u32, 5000)]
    #[crate::test]
    fn test_sequence_unsigned(
        #[case] base: u32,
        #[case] multiplier: u32,
        #[case] len: usize,
    ) -> VortexResult<()> {
        use vortex::dtype::Nullability;
        use vortex::encodings::sequence::Sequence;

        let expected: Vec<u32> = (0..len).map(|i| base + (i as u32) * multiplier).collect();

        let seq = Sequence::try_new_typed(base, multiplier, Nullability::NonNullable, len)?;

        let cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())?;
        let plan = dispatch_plan(&seq.into_array(), &cuda_ctx)?;

        let actual = run_dynamic_dispatch_plan(
            &cuda_ctx,
            expected.len(),
            &plan.dispatch_plan,
            plan.shared_mem_bytes,
        )?;
        assert_eq!(actual, expected);

        Ok(())
    }

    #[rstest]
    #[case(0i32, 1i32, 100)]
    #[case(-10i32, 3i32, 2048)]
    #[case(100i32, -1i32, 100)]
    #[case(-500i32, -7i32, 50)]
    #[case(0i32, 1i32, 5000)]
    #[crate::test]
    fn test_sequence_signed(
        #[case] base: i32,
        #[case] multiplier: i32,
        #[case] len: usize,
    ) -> VortexResult<()> {
        use vortex::dtype::Nullability;
        use vortex::encodings::sequence::Sequence;

        let expected: Vec<i32> = (0..len).map(|i| base + (i as i32) * multiplier).collect();

        let seq = Sequence::try_new_typed(base, multiplier, Nullability::NonNullable, len)?;

        let cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())?;
        let plan = dispatch_plan(&seq.into_array(), &cuda_ctx)?;

        let actual_u32 = run_dynamic_dispatch_plan(
            &cuda_ctx,
            expected.len(),
            &plan.dispatch_plan,
            plan.shared_mem_bytes,
        )?;
        let actual: Vec<i32> = actual_u32.into_iter().map(|v| v as i32).collect();
        assert_eq!(actual, expected);

        Ok(())
    }
}
