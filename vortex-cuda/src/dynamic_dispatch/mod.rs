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
use std::sync::Arc;

use cudarc::driver::DevicePtr;
use cudarc::driver::LaunchConfig;
use cudarc::driver::PushKernelArg;
use vortex::array::Canonical;
use vortex::array::IntoArray;
use vortex::array::arrays::ConstantArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::buffer::BufferHandle;
use vortex::array::buffer::DeviceBufferExt;
use vortex::array::match_each_unsigned_integer_ptype;
use vortex::array::scalar::Scalar;
use vortex::array::validity::Validity;
use vortex::buffer::Alignment;
use vortex::buffer::ByteBuffer;
use vortex::buffer::ByteBufferMut;
use vortex::dtype::DType;
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

/// Convert a Rust `PType` to the C `PTypeTag` constant.
pub fn ptype_to_tag(ptype: PType) -> PTypeTag {
    match ptype {
        PType::U8 => PTypeTag_PTYPE_U8,
        PType::U16 => PTypeTag_PTYPE_U16,
        PType::U32 => PTypeTag_PTYPE_U32,
        PType::U64 => PTypeTag_PTYPE_U64,
        PType::I8 => PTypeTag_PTYPE_I8,
        PType::I16 => PTypeTag_PTYPE_I16,
        PType::I32 => PTypeTag_PTYPE_I32,
        PType::I64 => PTypeTag_PTYPE_I64,
        PType::F16 => unreachable!("F16 is not supported by CUDA dynamic dispatch"),
        PType::F32 => PTypeTag_PTYPE_F32,
        PType::F64 => PTypeTag_PTYPE_F64,
    }
}

/// Convert a C `PTypeTag` back to a Rust `PType`.
pub fn tag_to_ptype(tag: PTypeTag) -> PType {
    match tag {
        PTypeTag_PTYPE_U8 => PType::U8,
        PTypeTag_PTYPE_U16 => PType::U16,
        PTypeTag_PTYPE_U32 => PType::U32,
        PTypeTag_PTYPE_U64 => PType::U64,
        PTypeTag_PTYPE_I8 => PType::I8,
        PTypeTag_PTYPE_I16 => PType::I16,
        PTypeTag_PTYPE_I32 => PType::I32,
        PTypeTag_PTYPE_I64 => PType::I64,
        PTypeTag_PTYPE_F32 => PType::F32,
        PTypeTag_PTYPE_F64 => PType::F64,
        _ => unreachable!("unknown PTypeTag {tag}"),
    }
}

/// Serialize a `#[repr(C)]` struct to a byte vector for the packed plan.
///
/// Copies field data into a pre-zeroed buffer so padding holes are
/// deterministically zero, avoiding UB from reading uninitialised bytes.
fn as_bytes<T: Sized>(val: &T) -> Vec<u8> {
    let n = size_of::<T>();
    let mut buf = vec![0u8; n];
    // SAFETY: T is a bindgen-generated #[repr(C)] struct with only
    // integer/float/enum fields. We overwrite the zeroed buffer with
    // the struct's bytes; padding holes keep their zero value.
    unsafe {
        std::ptr::copy_nonoverlapping(std::ptr::addr_of!(*val).cast::<u8>(), buf.as_mut_ptr(), n);
    }
    buf
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
    pub smem_byte_offset: u32,
    /// Number of elements in this stage.
    pub len: u32,
    /// PType tag for the source op's output type.
    pub source_ptype: PTypeTag,
    /// The source operation that produces the initial values (e.g. load, bitunpack, sequence).
    pub source: SourceOp,
    /// Chain of element-wise scalar operations applied after the source (e.g. frame-of-reference, zigzag, ALP).
    pub scalar_ops: Vec<ScalarOp>,
}

impl MaterializedStage {
    pub fn new(
        input_ptr: u64,
        smem_byte_offset: u32,
        len: u32,
        source_ptype: PTypeTag,
        source: SourceOp,
        scalar_ops: &[ScalarOp],
    ) -> Self {
        Self {
            input_ptr,
            smem_byte_offset,
            len,
            source_ptype,
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
    pub smem_byte_offset: u32,
    pub len: u32,
    pub source_ptype: PTypeTag,
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
    pub fn new<I>(stages: I, output_ptype: PTypeTag) -> Self
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

        let header = PlanHeader {
            num_stages: stages.len() as u8,
            output_ptype,
            plan_size_bytes: total_size as u16,
        };
        buffer.extend_from_slice(&as_bytes(&header));

        for stage in &stages {
            let packed_stage = PackedStage {
                input_ptr: stage.input_ptr,
                smem_byte_offset: stage.smem_byte_offset,
                len: stage.len,
                source: stage.source,
                num_scalar_ops: stage.scalar_ops.len() as u8,
                source_ptype: stage.source_ptype,
            };
            buffer.extend_from_slice(&as_bytes(&packed_stage));
            for op in &stage.scalar_ops {
                buffer.extend_from_slice(&as_bytes(op));
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
        self.header().num_stages
    }

    /// PType of the final output array.
    pub fn output_ptype(&self) -> PType {
        tag_to_ptype(self.header().output_ptype)
    }

    fn header(&self) -> PlanHeader {
        unsafe { *self.buffer.as_ptr().cast() }
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
            smem_byte_offset: ps.smem_byte_offset,
            len: ps.len,
            source_ptype: ps.source_ptype,
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
    /// * `ends_smem_byte_offset` - byte offset to decoded ends in smem
    /// * `values_smem_byte_offset` - byte offset to decoded values in smem
    /// * `num_runs` - number of runs (length of ends/values)
    /// * `offset` - logical offset for sliced arrays
    pub fn runend(
        ends_smem_byte_offset: u32,
        values_smem_byte_offset: u32,
        num_runs: u64,
        offset: u64,
    ) -> Self {
        Self {
            op_code: SourceOp_SourceOpCode_RUNEND,
            params: SourceParams {
                runend: SourceParams_RunEndParams {
                    ends_smem_byte_offset,
                    values_smem_byte_offset,
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
    pub fn frame_of_ref(reference: u64, output_ptype: PTypeTag) -> Self {
        Self {
            op_code: ScalarOp_ScalarOpCode_FOR,
            output_ptype,
            params: ScalarParams {
                frame_of_ref: ScalarParams_FoRParams { reference },
            },
        }
    }

    /// Zigzag decode.
    pub fn zigzag(output_ptype: PTypeTag) -> Self {
        // SAFETY: Zigzag has no parameters; zeroed union is valid.
        Self {
            op_code: ScalarOp_ScalarOpCode_ZIGZAG,
            output_ptype,
            params: unsafe { std::mem::zeroed() },
        }
    }

    /// ALP floating-point decode.
    pub fn alp(f: f32, e: f32) -> Self {
        Self {
            op_code: ScalarOp_ScalarOpCode_ALP,
            output_ptype: PTypeTag_PTYPE_F32,
            params: ScalarParams {
                alp: ScalarParams_AlpParams { f, e },
            },
        }
    }

    /// Dictionary gather: use current value as index into decoded values
    /// in shared memory (populated by an earlier input stage).
    pub fn dict(values_smem_byte_offset: u32, output_ptype: PTypeTag) -> Self {
        Self {
            op_code: ScalarOp_ScalarOpCode_DICT,
            output_ptype,
            params: ScalarParams {
                dict: ScalarParams_DictParams {
                    values_smem_byte_offset,
                },
            },
        }
    }
}

impl MaterializedPlan {
    pub fn execute(self, len: usize, ctx: &mut CudaExecutionCtx) -> VortexResult<Canonical> {
        let output_ptype = self.dispatch_plan.output_ptype();

        // All values are null — no need to touch the GPU.
        if matches!(self.validity, Validity::AllInvalid) {
            let dtype = DType::Primitive(output_ptype, Nullability::Nullable);
            return ConstantArray::new(Scalar::null(dtype), len)
                .into_array()
                .execute::<Canonical>(ctx.execution_ctx());
        }

        // The CUDA kernels are instantiated for unsigned integer types only;
        // map signed/float ptypes to their same-width unsigned counterpart.
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
        let nullability = self.validity.nullability();

        if len == 0 {
            return Ok(Canonical::Primitive(PrimitiveArray::empty::<T>(
                nullability,
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
        let num_blocks = u32::try_from(len.div_ceil(ELEMENTS_PER_BLOCK as usize))?;
        let config = LaunchConfig {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
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
            self.validity,
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
    use vortex::array::validity::Validity;
    use vortex::array::validity::Validity::NonNullable;
    use vortex::buffer::Buffer;
    use vortex::dtype::PType;
    use vortex::encodings::alp::ALP;
    use vortex::encodings::alp::ALPArrayExt;
    use vortex::encodings::alp::ALPArraySlotsExt;
    use vortex::encodings::alp::ALPFloat;
    use vortex::encodings::alp::Exponents;
    use vortex::encodings::alp::alp_encode;
    use vortex::encodings::fastlanes::BitPacked;
    use vortex::encodings::fastlanes::BitPackedArray;
    use vortex::encodings::fastlanes::FoR;
    use vortex::encodings::fastlanes::FoRArrayExt;
    use vortex::encodings::runend::RunEnd;
    use vortex::encodings::zigzag::ZigZag;
    use vortex::error::VortexExpect;
    use vortex::error::VortexResult;
    use vortex::session::VortexSession;
    use vortex_array::LEGACY_SESSION;
    use vortex_array::VortexSessionExecute;

    use super::*;
    use crate::CanonicalCudaExt;
    use crate::CudaBufferExt;
    use crate::CudaDeviceBuffer;
    use crate::CudaExecutionCtx;
    use crate::executor::CudaArrayExt;
    use crate::hybrid_dispatch::try_gpu_dispatch;
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
            .map(|&r| ScalarOp::frame_of_ref(r as u64, PTypeTag_PTYPE_U32))
            .collect();

        let plan = CudaDispatchPlan::new(
            [MaterializedStage::new(
                input_ptr,
                0,
                len as u32,
                PTypeTag_PTYPE_U32,
                SourceOp::bitunpack(bit_width, 0),
                &scalar_ops,
            )],
            PTypeTag_PTYPE_U32,
        );
        assert_eq!(plan.stage(0).num_scalar_ops, 4);

        let actual = run_dynamic_dispatch_plan(&cuda_ctx, len, &plan, SMEM_TILE_SIZE * 4)?;
        assert_eq!(actual, expected);

        Ok(())
    }

    #[crate::test]
    fn test_plan_structure() {
        // Stage 0: input dict values (BP→FoR), 256 u32 elements → smem bytes [0..1024)
        // Stage 1: output codes (BP→FoR→DICT), 1024 elements, gather from smem byte 0
        let values_smem_bytes: u32 = 256 * 4; // 256 u32 elements × 4 bytes
        let plan = CudaDispatchPlan::new(
            [
                MaterializedStage::new(
                    0xAAAA,
                    0,
                    256,
                    PTypeTag_PTYPE_U32,
                    SourceOp::bitunpack(4, 0),
                    &[ScalarOp::frame_of_ref(10, PTypeTag_PTYPE_U32)],
                ),
                MaterializedStage::new(
                    0xBBBB,
                    values_smem_bytes,
                    1024,
                    PTypeTag_PTYPE_U32,
                    SourceOp::bitunpack(6, 0),
                    &[
                        ScalarOp::frame_of_ref(42, PTypeTag_PTYPE_U32),
                        ScalarOp::dict(0, PTypeTag_PTYPE_U32),
                    ],
                ),
            ],
            PTypeTag_PTYPE_U32,
        );

        assert_eq!(plan.num_stages(), 2);

        // Input stage
        let s0 = plan.stage(0);
        assert_eq!(s0.smem_byte_offset, 0);
        assert_eq!(s0.len, 256);
        assert_eq!(s0.source_ptype, PTypeTag_PTYPE_U32);
        assert_eq!(s0.input_ptr, 0xAAAA);

        // Output stage
        let s1 = plan.stage(1);
        assert_eq!(s1.smem_byte_offset, values_smem_bytes);
        assert_eq!(s1.len, SMEM_TILE_SIZE);
        assert_eq!(s1.source_ptype, PTypeTag_PTYPE_U32);
        assert_eq!(s1.input_ptr, 0xBBBB);
        assert_eq!(s1.num_scalar_ops, 2);
        assert_eq!(
            unsafe { s1.scalar_ops[1].params.dict.values_smem_byte_offset },
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

        let plan = CudaDispatchPlan::new(
            [MaterializedStage::new(
                input_ptr,
                0,
                len as u32,
                PTypeTag_PTYPE_U32,
                SourceOp::load(),
                &[
                    ScalarOp::frame_of_ref(reference as u64, PTypeTag_PTYPE_U32),
                    ScalarOp::zigzag(PTypeTag_PTYPE_U32),
                    ScalarOp::alp(alp_f, alp_e),
                ],
            )],
            PTypeTag_PTYPE_U32,
        );

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

        let num_blocks = u32::try_from(output_len.div_ceil(ELEMENTS_PER_BLOCK as usize))?;
        let config = LaunchConfig {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
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

        let alp = alp_encode(
            float_prim.as_view(),
            Some(exponents),
            &mut LEGACY_SESSION.create_execution_ctx(),
        )?;
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
    async fn test_dict_mixed_width_u8_codes_u32_values() -> VortexResult<()> {
        let dict_values: Vec<u32> = vec![100, 200, 300, 400];
        let len = 3000;
        let codes: Vec<u8> = (0..len).map(|i| (i % dict_values.len()) as u8).collect();

        let codes_prim = PrimitiveArray::new(Buffer::from(codes.clone()), NonNullable);
        let values_prim = PrimitiveArray::new(Buffer::from(dict_values.clone()), NonNullable);
        let dict = DictArray::try_new(codes_prim.into_array(), values_prim.into_array())?;
        let array = dict.into_array();

        // Mixed-width Dict (u8 codes, u32 values): both are Primitive, so
        // walk_mixed_width_child grabs the codes buffer directly as a LOAD
        // source. No pending subtrees → Fused.
        let plan = DispatchPlan::new(&array)?;
        assert!(
            matches!(plan, DispatchPlan::Fused(..)),
            "expected Fused for mixed-width Dict with primitive codes"
        );

        // Execute through the hybrid dispatch path (handles widening).
        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())?;
        let canonical = try_gpu_dispatch(&array, &mut cuda_ctx).await?;
        let result = CanonicalCudaExt::into_host(canonical).await?.into_array();

        let expected: Vec<u32> = codes.iter().map(|&c| dict_values[c as usize]).collect();
        let expected_arr = PrimitiveArray::new(Buffer::from(expected), NonNullable).into_array();
        vortex::array::assert_arrays_eq!(expected_arr, result);

        Ok(())
    }

    #[crate::test]
    async fn test_dict_mixed_width_u16_codes_u32_values() -> VortexResult<()> {
        let dict_values: Vec<u32> = vec![1000, 2000, 3000, 4000, 5000];
        let len = 2048;
        let codes: Vec<u16> = (0..len).map(|i| (i % dict_values.len()) as u16).collect();

        let codes_prim = PrimitiveArray::new(Buffer::from(codes.clone()), NonNullable);
        let values_prim = PrimitiveArray::new(Buffer::from(dict_values.clone()), NonNullable);
        let dict = DictArray::try_new(codes_prim.into_array(), values_prim.into_array())?;
        let array = dict.into_array();

        // Mixed-width Dict (u16 codes, u32 values): both are Primitive, so
        // walk_mixed_width_child grabs the codes buffer directly as a LOAD
        // source. No pending subtrees → Fused.
        let plan = DispatchPlan::new(&array)?;
        assert!(
            matches!(plan, DispatchPlan::Fused(..)),
            "expected Fused for mixed-width Dict with primitive codes"
        );

        // Execute through the hybrid dispatch path (handles widening).
        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())?;
        let canonical = try_gpu_dispatch(&array, &mut cuda_ctx).await?;
        let result = CanonicalCudaExt::into_host(canonical).await?.into_array();

        let expected: Vec<u32> = codes.iter().map(|&c| dict_values[c as usize]).collect();
        let expected_arr = PrimitiveArray::new(Buffer::from(expected), NonNullable).into_array();
        vortex::array::assert_arrays_eq!(expected_arr, result);

        Ok(())
    }

    #[crate::test]
    async fn test_runend_mixed_width_u64_ends_u32_values() -> VortexResult<()> {
        let ends: Vec<u64> = vec![1000, 2000, 3000];
        let values: Vec<u32> = vec![10, 20, 30];
        let len = 3000;

        let ends_arr = PrimitiveArray::new(Buffer::from(ends), NonNullable).into_array();
        let values_arr = PrimitiveArray::new(Buffer::from(values), NonNullable).into_array();
        let re = RunEnd::new(ends_arr, values_arr);
        let array = re.into_array();

        // Ends (u64) are wider than values (u32), so the kernel would truncate
        // ends via load_element<T>. The plan builder rejects this as Unfused.
        let plan = DispatchPlan::new(&array)?;
        assert!(
            matches!(plan, DispatchPlan::Unfused),
            "expected Unfused for RunEnd with wider ends"
        );

        // Execute through the non-fused dispatch path.
        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())?;
        let canonical = try_gpu_dispatch(&array, &mut cuda_ctx).await?;
        let result = CanonicalCudaExt::into_host(canonical).await?.into_array();

        let expected: Vec<u32> = (0..len as u64)
            .map(|i| {
                if i < 1000 {
                    10
                } else if i < 2000 {
                    20
                } else {
                    30
                }
            })
            .collect();
        let expected_arr = PrimitiveArray::new(Buffer::from(expected), NonNullable).into_array();
        vortex::array::assert_arrays_eq!(expected_arr, result);

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

    #[crate::test]
    async fn test_for_bitpacked_u8() -> VortexResult<()> {
        let bit_width: u8 = 4;
        let len = 3000;
        let reference = 100u8;
        let max_val = (1u64 << bit_width).saturating_sub(1);
        let residuals: Vec<u8> = (0..len).map(|i| (i as u64 % (max_val + 1)) as u8).collect();
        let expected: Vec<u8> = residuals
            .iter()
            .map(|&r| r.wrapping_add(reference))
            .collect();

        let primitive = PrimitiveArray::new(Buffer::from(residuals), NonNullable);
        let bp = BitPacked::encode(&primitive.into_array(), bit_width).vortex_expect("bitpack u8");
        let for_arr = FoR::try_new(
            bp.into_array(),
            Scalar::primitive(reference, Nullability::NonNullable),
        )?;
        let array = for_arr.into_array();

        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())?;
        let canonical = try_gpu_dispatch(&array, &mut cuda_ctx).await?;
        let result = CanonicalCudaExt::into_host(canonical).await?.into_array();

        let expected_arr = PrimitiveArray::new(Buffer::from(expected), NonNullable).into_array();
        vortex::array::assert_arrays_eq!(expected_arr, result);
        Ok(())
    }

    #[crate::test]
    async fn test_for_bitpacked_u16() -> VortexResult<()> {
        let bit_width: u8 = 10;
        let len = 3000;
        let reference = 1000u16;
        let max_val = (1u64 << bit_width).saturating_sub(1);
        let residuals: Vec<u16> = (0..len)
            .map(|i| (i as u64 % (max_val + 1)) as u16)
            .collect();
        let expected: Vec<u16> = residuals
            .iter()
            .map(|&r| r.wrapping_add(reference))
            .collect();

        let primitive = PrimitiveArray::new(Buffer::from(residuals), NonNullable);
        let bp = BitPacked::encode(&primitive.into_array(), bit_width).vortex_expect("bitpack u16");
        let for_arr = FoR::try_new(
            bp.into_array(),
            Scalar::primitive(reference, Nullability::NonNullable),
        )?;
        let array = for_arr.into_array();

        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())?;
        let canonical = try_gpu_dispatch(&array, &mut cuda_ctx).await?;
        let result = CanonicalCudaExt::into_host(canonical).await?.into_array();

        let expected_arr = PrimitiveArray::new(Buffer::from(expected), NonNullable).into_array();
        vortex::array::assert_arrays_eq!(expected_arr, result);
        Ok(())
    }

    #[crate::test]
    async fn test_for_bitpacked_u64() -> VortexResult<()> {
        let bit_width: u8 = 20;
        let len = 3000;
        let reference = 100_000u64;
        let max_val = (1u64 << bit_width).saturating_sub(1);
        let residuals: Vec<u64> = (0..len).map(|i| i as u64 % (max_val + 1)).collect();
        let expected: Vec<u64> = residuals
            .iter()
            .map(|&r| r.wrapping_add(reference))
            .collect();

        let primitive = PrimitiveArray::new(Buffer::from(residuals), NonNullable);
        let bp = BitPacked::encode(&primitive.into_array(), bit_width).vortex_expect("bitpack u64");
        let for_arr = FoR::try_new(
            bp.into_array(),
            Scalar::primitive(reference, Nullability::NonNullable),
        )?;
        let array = for_arr.into_array();

        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())?;
        let canonical = try_gpu_dispatch(&array, &mut cuda_ctx).await?;
        let result = CanonicalCudaExt::into_host(canonical).await?.into_array();

        let expected_arr = PrimitiveArray::new(Buffer::from(expected), NonNullable).into_array();
        vortex::array::assert_arrays_eq!(expected_arr, result);
        Ok(())
    }

    #[crate::test]
    async fn test_empty_array() -> VortexResult<()> {
        let values: Vec<u32> = vec![];
        let primitive = PrimitiveArray::new(Buffer::from(values), NonNullable);
        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())?;
        let canonical = try_gpu_dispatch(&primitive.into_array(), &mut cuda_ctx).await?;
        let result = CanonicalCudaExt::into_host(canonical).await?.into_array();
        assert_eq!(result.len(), 0);
        Ok(())
    }

    #[crate::test]
    async fn test_single_element() -> VortexResult<()> {
        let values: Vec<u32> = vec![42];
        let primitive = PrimitiveArray::new(Buffer::from(values.clone()), NonNullable);
        let bp = BitPacked::encode(&primitive.into_array(), 6).vortex_expect("bitpack");
        let for_arr = FoR::try_new(
            bp.into_array(),
            Scalar::primitive(0u32, Nullability::NonNullable),
        )?;
        let array = for_arr.into_array();

        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())?;
        let canonical = try_gpu_dispatch(&array, &mut cuda_ctx).await?;
        let result = CanonicalCudaExt::into_host(canonical).await?.into_array();

        let expected = PrimitiveArray::new(Buffer::from(values), NonNullable).into_array();
        vortex::array::assert_arrays_eq!(expected, result);
        Ok(())
    }

    #[crate::test]
    async fn test_exactly_elements_per_block() -> VortexResult<()> {
        // Exactly 2048 elements — one full block, no remainder
        let bit_width: u8 = 6;
        let len = 2048;
        let reference = 1000u32;
        let max_val = (1u64 << bit_width).saturating_sub(1);
        let residuals: Vec<u32> = (0..len)
            .map(|i| (i as u64 % (max_val + 1)) as u32)
            .collect();
        let expected: Vec<u32> = residuals.iter().map(|&r| r + reference).collect();

        let primitive = PrimitiveArray::new(Buffer::from(residuals), NonNullable);
        let bp = BitPacked::encode(&primitive.into_array(), bit_width).vortex_expect("bitpack");
        let for_arr = FoR::try_new(
            bp.into_array(),
            Scalar::primitive(reference, Nullability::NonNullable),
        )?;
        let array = for_arr.into_array();

        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())?;
        let canonical = try_gpu_dispatch(&array, &mut cuda_ctx).await?;
        let result = CanonicalCudaExt::into_host(canonical).await?.into_array();

        let expected_arr = PrimitiveArray::new(Buffer::from(expected), NonNullable).into_array();
        vortex::array::assert_arrays_eq!(expected_arr, result);
        Ok(())
    }

    #[crate::test]
    fn test_f64_rejected() {
        // F64 arrays should be rejected by the plan builder, not silently accepted.
        let values: Vec<f64> = vec![1.0, 2.0, 3.0];
        let primitive = PrimitiveArray::new(Buffer::from(values), NonNullable);
        let plan = DispatchPlan::new(&primitive.into_array())
            .expect("DispatchPlan::new should not fail for f64");
        assert!(
            matches!(plan, DispatchPlan::Unfused),
            "expected F64 to be classified as Unfused"
        );
    }

    #[crate::test]
    async fn test_runend_u32_ends_u16_values() -> VortexResult<()> {
        // RunEnd with u32 ends, u16 values. Output type = u16.
        // Ends (u32) differ from output (u16) → pending subtree.
        let ends: Vec<u32> = vec![500, 1000, 1500, 2000];
        let values: Vec<u16> = vec![100, 200, 300, 400];
        let len = 2000;

        let ends_arr = PrimitiveArray::new(Buffer::from(ends), NonNullable).into_array();
        let values_arr = PrimitiveArray::new(Buffer::from(values), NonNullable).into_array();
        let re = RunEnd::new(ends_arr, values_arr);
        let array = re.into_array();

        // Ends (u32) are wider than values (u16), so the kernel would truncate
        // ends via load_element<T>. The plan builder rejects this as Unfused.
        let plan = DispatchPlan::new(&array)?;
        assert!(
            matches!(plan, DispatchPlan::Unfused),
            "expected Unfused for RunEnd with wider ends"
        );

        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())?;
        let canonical = try_gpu_dispatch(&array, &mut cuda_ctx).await?;
        let result = CanonicalCudaExt::into_host(canonical).await?.into_array();

        let expected: Vec<u16> = (0..len as u64)
            .map(|i| {
                if i < 500 {
                    100u16
                } else if i < 1000 {
                    200
                } else if i < 1500 {
                    300
                } else {
                    400
                }
            })
            .collect();
        let expected_arr = PrimitiveArray::new(Buffer::from(expected), NonNullable).into_array();
        vortex::array::assert_arrays_eq!(expected_arr, result);

        Ok(())
    }

    #[crate::test]
    async fn test_dict_bitpacked_u8_codes_u32_values() -> VortexResult<()> {
        // Dict with BitPacked u8 codes (narrower than u32 output) and u32 values.
        // Codes become a pending subtree, values fuse.
        let dict_values: Vec<u32> = vec![100, 200, 300, 400];
        let len = 2048;
        let codes: Vec<u8> = (0..len).map(|i| (i % dict_values.len()) as u8).collect();

        let codes_prim = PrimitiveArray::new(Buffer::from(codes.clone()), NonNullable);
        // BitPack the u8 codes at 2 bits (4 values need 2 bits)
        let codes_bp =
            BitPacked::encode(&codes_prim.into_array(), 2).vortex_expect("bitpack codes");
        let values_prim = PrimitiveArray::new(Buffer::from(dict_values.clone()), NonNullable);
        let dict = DictArray::try_new(codes_bp.into_array(), values_prim.into_array())?;
        let array = dict.into_array();

        let plan = DispatchPlan::new(&array)?;
        assert!(
            matches!(plan, DispatchPlan::PartiallyFused { .. }),
            "expected PartiallyFused for mixed-width Dict with BitPacked codes"
        );

        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())?;
        let canonical = try_gpu_dispatch(&array, &mut cuda_ctx).await?;
        let result = CanonicalCudaExt::into_host(canonical).await?.into_array();

        let expected: Vec<u32> = codes.iter().map(|&c| dict_values[c as usize]).collect();
        let expected_arr = PrimitiveArray::new(Buffer::from(expected), NonNullable).into_array();
        vortex::array::assert_arrays_eq!(expected_arr, result);

        Ok(())
    }

    #[crate::test]
    async fn test_sliced_dict_mixed_width() -> VortexResult<()> {
        // Sliced Dict with u8 codes and u32 values — combines PartiallyFused + slice handling.
        let dict_values: Vec<u32> = vec![100, 200, 300, 400];
        let full_len = 4096;
        let codes: Vec<u8> = (0..full_len)
            .map(|i| (i % dict_values.len()) as u8)
            .collect();

        let codes_prim = PrimitiveArray::new(Buffer::from(codes.clone()), NonNullable);
        let values_prim = PrimitiveArray::new(Buffer::from(dict_values.clone()), NonNullable);
        let dict = DictArray::try_new(codes_prim.into_array(), values_prim.into_array())?;

        // Slice from 1000..3000
        let sliced = dict.into_array().slice(1000..3000)?;

        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())?;
        let canonical = try_gpu_dispatch(&sliced, &mut cuda_ctx).await?;
        let result = CanonicalCudaExt::into_host(canonical).await?.into_array();

        let expected: Vec<u32> = codes[1000..3000]
            .iter()
            .map(|&c| dict_values[c as usize])
            .collect();
        let expected_arr = PrimitiveArray::new(Buffer::from(expected), NonNullable).into_array();
        vortex::array::assert_arrays_eq!(expected_arr, result);

        Ok(())
    }

    /// Verify that `load_element<T>` sign-extends signed narrow types when
    /// widening to a wider T. E.g. i8(-1) = 0xFF must become u32(0xFFFFFFFF)
    /// (the bit-pattern for i32(-1)), not u32(0x000000FF) = 255.
    #[crate::test]
    fn test_load_element_sign_extends_i8_to_u32() -> VortexResult<()> {
        let cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())?;

        let i8_values: Vec<i8> = vec![-1, -2, -3, 127, -128, 0, 1, 42];
        let len = i8_values.len();
        let device_buf = Arc::new(cuda_ctx.stream().clone_htod(&i8_values).expect("htod"));
        let (input_ptr, _) = device_buf.device_ptr(cuda_ctx.stream());

        // Build a single-stage LOAD plan: source ptype = I8, output ptype = U32.
        // The kernel (instantiated as u32) must sign-extend each i8 element.
        let plan = CudaDispatchPlan::new(
            [MaterializedStage::new(
                input_ptr,
                0,
                len as u32,
                PTypeTag_PTYPE_I8,
                SourceOp::load(),
                &[],
            )],
            PTypeTag_PTYPE_U32,
        );

        let actual = run_dynamic_dispatch_plan(&cuda_ctx, len, &plan, SMEM_TILE_SIZE * 4)?;

        // Expected: each i8 sign-extended to i32, then viewed as u32.
        let expected: Vec<u32> = i8_values.iter().map(|&v| (v as i32) as u32).collect();
        assert_eq!(actual, expected);

        Ok(())
    }

    /// Same as above but for i16 → u32 widening.
    #[crate::test]
    fn test_load_element_sign_extends_i16_to_u32() -> VortexResult<()> {
        let cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())?;

        let i16_values: Vec<i16> = vec![-1, -256, -32768, 32767, 0, 1, -100, 12345];
        let len = i16_values.len();
        let device_buf = Arc::new(cuda_ctx.stream().clone_htod(&i16_values).expect("htod"));
        let (input_ptr, _) = device_buf.device_ptr(cuda_ctx.stream());

        let plan = CudaDispatchPlan::new(
            [MaterializedStage::new(
                input_ptr,
                0,
                len as u32,
                PTypeTag_PTYPE_I16,
                SourceOp::load(),
                &[],
            )],
            PTypeTag_PTYPE_U32,
        );

        let actual = run_dynamic_dispatch_plan(&cuda_ctx, len, &plan, SMEM_TILE_SIZE * 4)?;

        let expected: Vec<u32> = i16_values.iter().map(|&v| (v as i32) as u32).collect();
        assert_eq!(actual, expected);

        Ok(())
    }

    // ═══════════════════════════════════════════════════════════════════
    // Validity propagation tests
    // ═══════════════════════════════════════════════════════════════════

    /// Nullable Primitive array — LOAD source with validity propagated.
    #[crate::test]
    async fn test_nullable_primitive() -> VortexResult<()> {
        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())?;

        let array = PrimitiveArray::from_option_iter(
            (0..2048u32).map(|i| if i % 3 == 0 { None } else { Some(i) }),
        );
        let cpu = crate::canonicalize_cpu(array.clone())?.into_array();

        let gpu = try_gpu_dispatch(&array.into_array(), &mut cuda_ctx)
            .await?
            .into_host()
            .await?
            .into_array();

        vortex::array::assert_arrays_eq!(cpu, gpu);
        Ok(())
    }

    /// Nullable FoR(BitPacked) — validity from the root propagated through
    /// the fused plan. The standard encoding flow is: subtract FoR reference
    /// to get residuals, then bitpack. BitPacked::encode preserves input
    /// validity, so this produces a real nullable FoR(BitPacked) tree.
    #[crate::test]
    async fn test_nullable_for_bitpacked() -> VortexResult<()> {
        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())?;

        let len = 2048;
        let reference = 1000u32;

        // Original values in [reference, reference+63], every 5th null.
        let values: Vec<Option<u32>> = (0..len)
            .map(|i| {
                if i % 5 == 0 {
                    None
                } else {
                    Some((i as u32 % 64) + reference)
                }
            })
            .collect();
        let prim = PrimitiveArray::from_option_iter(values.iter().copied());
        let cpu = crate::canonicalize_cpu(prim.clone())?.into_array();

        // FoR encoding: subtract reference to get residuals [0..63].
        // Null positions get 0 (from from_option_iter), which is fine —
        // after subtracting reference it wraps, but validity masks it.
        let residuals =
            PrimitiveArray::from_option_iter(values.iter().map(|v| v.map(|x| x - reference)));

        // BitPacked::encode preserves nullable validity from the input.
        let bp = BitPacked::encode(&residuals.into_array(), 6)?;
        let for_arr = FoR::try_new(bp.into_array(), reference.into())?;

        // Verify the plan actually fuses (not just a LOAD).
        assert!(
            matches!(
                DispatchPlan::new(&for_arr.clone().into_array())?,
                DispatchPlan::Fused(_)
            ),
            "FoR(BitPacked) with nullable validity should produce a Fused plan"
        );

        let gpu = try_gpu_dispatch(&for_arr.into_array(), &mut cuda_ctx)
            .await?
            .into_host()
            .await?
            .into_array();

        vortex::array::assert_arrays_eq!(cpu, gpu);
        Ok(())
    }

    /// AllInvalid array — kernel should be skipped entirely.
    #[crate::test]
    async fn test_all_invalid_skips_kernel() -> VortexResult<()> {
        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())?;

        let array = PrimitiveArray::new(Buffer::from(vec![0u32; 2048]), Validity::AllInvalid);

        let result = try_gpu_dispatch(&array.into_array(), &mut cuda_ctx)
            .await?
            .into_host()
            .await?;

        let prim = result.into_primitive();
        assert_eq!(prim.len(), 2048);
        assert!(matches!(prim.validity()?, Validity::AllInvalid));
        Ok(())
    }

    /// AllValid nullable array — should fuse and produce AllValid output.
    #[crate::test]
    async fn test_all_valid_nullable() -> VortexResult<()> {
        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())?;

        let values: Vec<u32> = (0..2048).collect();
        let array = PrimitiveArray::new(Buffer::from(values.clone()), Validity::AllValid);

        let cpu = crate::canonicalize_cpu(array.clone())?.into_array();
        let gpu = try_gpu_dispatch(&array.into_array(), &mut cuda_ctx)
            .await?
            .into_host()
            .await?
            .into_array();

        vortex::array::assert_arrays_eq!(cpu, gpu);
        Ok(())
    }

    /// Dict with nullable codes must fall back to Unfused (not fused).
    #[crate::test]
    fn test_dict_nullable_codes_rejected() -> VortexResult<()> {
        use vortex::buffer::buffer;

        let codes = PrimitiveArray::from_option_iter([Some(0u32), None, Some(1), None, Some(2)]);
        let values = PrimitiveArray::new(buffer![10u32, 20, 30], NonNullable);
        let dict = DictArray::try_new(codes.into_array(), values.into_array())?;

        let plan = DispatchPlan::new(&dict.into_array())?;
        assert!(
            matches!(plan, DispatchPlan::Unfused),
            "Dict with nullable codes should fall back to Unfused"
        );
        Ok(())
    }

    /// Dict with non-nullable codes but nullable values should still fuse.
    #[crate::test]
    async fn test_dict_nullable_values_fuses() -> VortexResult<()> {
        use vortex::buffer::buffer;

        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())?;

        let codes = PrimitiveArray::new(buffer![0u32, 1, 2, 2, 1, 0], NonNullable);
        let values = PrimitiveArray::from_option_iter([Some(10u32), None, Some(30)]);
        let dict = DictArray::try_new(codes.into_array(), values.into_array())?;

        let cpu = crate::canonicalize_cpu(dict.clone())?.into_array();
        let gpu = dict
            .into_array()
            .execute_cuda(&mut cuda_ctx)
            .await?
            .into_host()
            .await?
            .into_array();

        vortex::array::assert_arrays_eq!(cpu, gpu);
        Ok(())
    }

    /// Nullable FoR(BitPacked) through CUB filter — the original bug scenario.
    /// Validity must survive through fused dispatch and into the filter.
    #[crate::test]
    async fn test_nullable_fused_then_filter() -> VortexResult<()> {
        use vortex::array::arrays::FilterArray;
        use vortex::mask::Mask;

        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())?;

        let len = 2048usize;
        let values: Vec<Option<u32>> = (0..len)
            .map(|i| {
                if i % 7 == 0 {
                    None
                } else {
                    Some((i % 64) as u32)
                }
            })
            .collect();
        let prim = PrimitiveArray::from_option_iter(values.iter().copied());

        // Keep every other element.
        let mask = Mask::from_iter((0..len).map(|i| i % 2 == 0));
        let filter_array = FilterArray::try_new(prim.into_array(), mask)?;

        let cpu = crate::canonicalize_cpu(filter_array.clone())?.into_array();
        let gpu = filter_array
            .into_array()
            .execute_cuda(&mut cuda_ctx)
            .await?
            .into_host()
            .await?
            .into_array();

        vortex::array::assert_arrays_eq!(cpu, gpu);
        Ok(())
    }

    /// Empty nullable array should preserve nullability.
    #[crate::test]
    async fn test_empty_nullable_array() -> VortexResult<()> {
        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())?;

        let array = PrimitiveArray::new(Buffer::<u32>::empty(), Validity::AllValid);
        let result = try_gpu_dispatch(&array.into_array(), &mut cuda_ctx).await?;
        let prim = result.into_primitive();
        assert_eq!(prim.len(), 0);
        assert_eq!(prim.validity()?.nullability(), Nullability::Nullable);
        Ok(())
    }
}
