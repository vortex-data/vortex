// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Host interface for dynamic CUDA kernel dispatch.
//!
//! A [`DynamicDispatchPlan`] is a linear sequence of stages produced by
//! [`build_plan`], which walks an encoding tree (e.g., `ALP(FoR(BitPacked))`)
//! and flattens it into input stages followed by a single output stage.
//!
//! Input stages write intermediate results (dictionary values, run-end
//! endpoints) into bump-allocated shared memory regions. The output stage
//! references those regions and writes final results to global memory.
//!
//! Shared memory is dynamically sized at launch time via
//! [`DynamicDispatchPlan::shared_mem_bytes`].

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(clippy::cast_possible_truncation)]

mod plan_builder;
pub use plan_builder::build_plan;

include!(concat!(env!("OUT_DIR"), "/dynamic_dispatch.rs"));

// SAFETY: C ABI structs with contiguous memory.
unsafe impl cudarc::driver::DeviceRepr for DynamicDispatchPlan {}
unsafe impl cudarc::driver::DeviceRepr for Stage {}

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

    /// Decode run-end encoding. Offsets reference shared memory regions
    /// populated by earlier input stages.
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

impl Stage {
    /// Create an input stage that decodes an input into a shared memory region.
    /// The output persists at `smem_offset` for the output stage to reference.
    pub fn input(
        input_ptr: u64,
        smem_offset: u32,
        len: u32,
        source: SourceOp,
        scalar_ops: &[ScalarOp],
    ) -> Self {
        assert!(scalar_ops.len() <= MAX_SCALAR_OPS as usize);
        let mut ops: [ScalarOp; MAX_SCALAR_OPS as usize] = unsafe { std::mem::zeroed() };
        ops[..scalar_ops.len()].copy_from_slice(scalar_ops);
        Self {
            input_ptr,
            smem_offset,
            len,
            source,
            num_scalar_ops: scalar_ops.len() as u8,
            scalar_ops: ops,
        }
    }

    /// Create the output stage. The kernel tiles `ELEMENTS_PER_BLOCK` elements
    /// through a [`SMEM_TILE_SIZE`] shared-memory region to reduce usage.
    pub fn output(
        input_ptr: u64,
        smem_offset: u32,
        source: SourceOp,
        scalar_ops: &[ScalarOp],
    ) -> Self {
        Self::input(input_ptr, smem_offset, SMEM_TILE_SIZE, source, scalar_ops)
    }
}

impl DynamicDispatchPlan {
    /// Create a dispatch plan from a sequence of stages.
    /// The last stage is the output pipeline; earlier stages are input stages.
    ///
    /// # Panics
    ///
    /// Panics if `stages` is empty or exceeds `MAX_STAGES`.
    pub fn new(stages: impl AsRef<[Stage]>) -> Self {
        let stages_slice = stages.as_ref();
        assert!(!stages_slice.is_empty());
        assert!(stages_slice.len() <= MAX_STAGES as usize);
        let mut buf: [Stage; MAX_STAGES as usize] = unsafe { std::mem::zeroed() };
        buf[..stages_slice.len()].copy_from_slice(stages_slice);
        Self {
            num_stages: stages_slice.len() as u8,
            stages: buf,
        }
    }

    /// Compute the dynamic shared memory bytes needed for this plan.
    ///
    /// All input stage outputs must remain in shared memory simultaneously
    /// so the output stage can reference them (e.g., dictionary lookups,
    /// run-end resolution). The total is `max(smem_offset + len)` across
    /// all stages, multiplied by the element size.
    pub fn shared_mem_bytes<T>(&self) -> u32 {
        let elem_size = size_of::<T>() as u32;
        let mut max_end: u32 = 0;
        for i in 0..self.num_stages as usize {
            let end = self.stages[i].smem_offset + self.stages[i].len;
            if end > max_end {
                max_end = end;
            }
        }
        max_end * elem_size
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
    use vortex::encodings::alp::ALPArray;
    use vortex::encodings::alp::ALPFloat;
    use vortex::encodings::alp::Exponents;
    use vortex::encodings::alp::alp_encode;
    use vortex::encodings::fastlanes::BitPackedArray;
    use vortex::encodings::fastlanes::FoRArray;
    use vortex::encodings::runend::RunEndArray;
    use vortex::encodings::zigzag::ZigZagArray;
    use vortex::error::VortexExpect;
    use vortex::error::VortexResult;
    use vortex::session::VortexSession;

    use super::DynamicDispatchPlan;
    use super::SMEM_TILE_SIZE;
    use super::ScalarOp;
    use super::SourceOp;
    use super::Stage;
    use super::build_plan;
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
        BitPackedArray::encode(&primitive.into_array(), bit_width)
            .vortex_expect("failed to create BitPacked array")
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

        let bitpacked = make_bitpacked_array_u32(bit_width, len);
        let cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())?;
        let packed = bitpacked.packed().clone();
        let device_input = futures::executor::block_on(cuda_ctx.ensure_on_device(packed))?;
        let input_ptr = device_input.cuda_device_ptr()?;

        let scalar_ops: Vec<ScalarOp> = references
            .iter()
            .map(|&r| ScalarOp::frame_of_ref(r as u64))
            .collect();

        let plan = DynamicDispatchPlan::new([Stage::output(
            input_ptr,
            0,
            SourceOp::bitunpack(bit_width, 0),
            &scalar_ops,
        )]);
        assert_eq!(plan.stages[0].num_scalar_ops, 4);

        let actual = run_dynamic_dispatch_plan(&cuda_ctx, len, &plan)?;
        assert_eq!(actual, expected);

        Ok(())
    }

    #[crate::test]
    fn test_plan_structure() {
        // Stage 0: input dict values (BP→FoR) into smem[0..256)
        // Stage 1: output codes (BP→FoR→DICT) into smem[256..2304), gather from smem[0]
        let plan = DynamicDispatchPlan::new([
            Stage::input(
                0xAAAA,
                0,
                256,
                SourceOp::bitunpack(4, 0),
                &[ScalarOp::frame_of_ref(10)],
            ),
            Stage::output(
                0xBBBB,
                256,
                SourceOp::bitunpack(6, 0),
                &[ScalarOp::frame_of_ref(42), ScalarOp::dict(0)],
            ),
        ]);

        assert_eq!(plan.num_stages, 2);

        // Input stage
        assert_eq!(plan.stages[0].smem_offset, 0);
        assert_eq!(plan.stages[0].len, 256);
        assert_eq!(plan.stages[0].input_ptr, 0xAAAA);

        // Output stage
        assert_eq!(plan.stages[1].smem_offset, 256);
        assert_eq!(plan.stages[1].len, SMEM_TILE_SIZE);
        assert_eq!(plan.stages[1].input_ptr, 0xBBBB);
        assert_eq!(plan.stages[1].num_scalar_ops, 2);
        assert_eq!(
            unsafe { plan.stages[1].scalar_ops[1].params.dict.values_smem_offset },
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

        let plan = DynamicDispatchPlan::new([Stage::output(
            input_ptr,
            0,
            SourceOp::load(),
            &[
                ScalarOp::frame_of_ref(reference as u64),
                ScalarOp::zigzag(),
                ScalarOp::alp(alp_f, alp_e),
            ],
        )]);

        let actual = run_dynamic_dispatch_plan(&cuda_ctx, len, &plan)?;
        assert_eq!(actual, expected);

        Ok(())
    }

    /// Runs a dynamic dispatch plan on the GPU.
    fn run_dynamic_dispatch_plan(
        cuda_ctx: &CudaExecutionCtx,
        output_len: usize,
        plan: &DynamicDispatchPlan,
    ) -> VortexResult<Vec<u32>> {
        let smem_bytes = plan.shared_mem_bytes::<u32>();

        let output_slice = cuda_ctx
            .device_alloc::<u32>(output_len)
            .vortex_expect("alloc output");
        let output_buf = CudaDeviceBuffer::new(output_slice);
        let output_view = output_buf.as_view::<u32>();
        let (output_ptr, record_output) = output_view.device_ptr(cuda_ctx.stream());

        let device_plan = Arc::new(
            cuda_ctx
                .stream()
                .clone_htod(std::slice::from_ref(plan))
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
            shared_mem_bytes: smem_bytes,
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
        plan: &DynamicDispatchPlan,
    ) -> VortexResult<Vec<f32>> {
        let actual = run_dynamic_dispatch_plan(cuda_ctx, output_len, plan)?;
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

        let bp = make_bitpacked_array_u32(bit_width, len);
        let cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())?;
        let (plan, _bufs) = build_plan(&bp.into_array(), &cuda_ctx)?;

        let actual = run_dynamic_dispatch_plan(&cuda_ctx, len, &plan)?;
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

        let bp = make_bitpacked_array_u32(bit_width, len);
        let for_arr = FoRArray::try_new(bp.into_array(), Scalar::from(reference))?;

        let cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())?;
        let (plan, _bufs) = build_plan(&for_arr.into_array(), &cuda_ctx)?;

        let actual = run_dynamic_dispatch_plan(&cuda_ctx, len, &plan)?;
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
        let re = RunEndArray::new(ends_arr, values_arr);

        let cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())?;
        let (plan, _bufs) = build_plan(&re.into_array(), &cuda_ctx)?;

        let actual = run_dynamic_dispatch_plan(&cuda_ctx, len, &plan)?;
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
        let dict_bp = BitPackedArray::encode(&dict_prim.into_array(), 6)?;
        let dict_for = FoRArray::try_new(dict_bp.into_array(), Scalar::from(dict_reference))?;

        // BitPack the codes
        let codes_prim = PrimitiveArray::new(Buffer::from(codes), NonNullable);
        let codes_bp = BitPackedArray::encode(&codes_prim.into_array(), 6)?;

        let dict = DictArray::try_new(codes_bp.into_array(), dict_for.into_array())?;

        let cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())?;
        let (plan, _bufs) = build_plan(&dict.into_array(), &cuda_ctx)?;

        let actual = run_dynamic_dispatch_plan(&cuda_ctx, len, &plan)?;
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
        let for_arr = FoRArray::encode(alp.encoded().to_primitive())?;
        let bp = BitPackedArray::encode(for_arr.encoded(), 6)?;

        let tree = ALPArray::new(
            FoRArray::try_new(bp.into_array(), for_arr.reference_scalar().clone())?.into_array(),
            exponents,
            None,
        );

        let cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())?;
        let (plan, _bufs) = build_plan(&tree.into_array(), &cuda_ctx)?;

        let actual = run_dispatch_plan_f32(&cuda_ctx, len, &plan)?;
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
        let bp = BitPackedArray::encode(&prim.into_array(), bit_width)?;
        let zz = ZigZagArray::try_new(bp.into_array())?;

        let cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())?;
        let (plan, _bufs) = build_plan(&zz.into_array(), &cuda_ctx)?;

        let actual = run_dynamic_dispatch_plan(&cuda_ctx, len, &plan)?;
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
        let re = RunEndArray::new(ends_arr, values_arr);
        let for_arr = FoRArray::try_new(re.into_array(), Scalar::from(reference))?;

        let cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())?;
        let (plan, _bufs) = build_plan(&for_arr.into_array(), &cuda_ctx)?;

        let actual = run_dynamic_dispatch_plan(&cuda_ctx, len, &plan)?;
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
        let for_arr = FoRArray::try_new(dict.into_array(), Scalar::from(reference))?;

        let cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())?;
        let (plan, _bufs) = build_plan(&for_arr.into_array(), &cuda_ctx)?;

        let actual = run_dynamic_dispatch_plan(&cuda_ctx, len, &plan)?;
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
        let codes_bp = BitPackedArray::encode(&codes_prim.into_array(), bit_width)?;
        let codes_for = FoRArray::try_new(codes_bp.into_array(), Scalar::from(0u32))?;

        let values_prim = PrimitiveArray::new(Buffer::from(dict_values), NonNullable);
        let dict = DictArray::try_new(codes_for.into_array(), values_prim.into_array())?;

        let cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())?;
        let (plan, _bufs) = build_plan(&dict.into_array(), &cuda_ctx)?;

        let actual = run_dynamic_dispatch_plan(&cuda_ctx, len, &plan)?;
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
        let codes_bp = BitPackedArray::encode(&codes_prim.into_array(), bit_width)?;
        let values_prim = PrimitiveArray::new(Buffer::from(dict_values), NonNullable);

        let dict = DictArray::try_new(codes_bp.into_array(), values_prim.into_array())?;

        let cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())?;
        let (plan, _bufs) = build_plan(&dict.into_array(), &cuda_ctx)?;

        let actual = run_dynamic_dispatch_plan(&cuda_ctx, len, &plan)?;
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
        let (plan, _bufs) = build_plan(&sliced, &cuda_ctx)?;

        let actual = run_dynamic_dispatch_plan(&cuda_ctx, expected.len(), &plan)?;
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
        let bp = BitPackedArray::encode(&prim.into_array(), bit_width)?;
        let zz = ZigZagArray::try_new(bp.into_array())?;

        let sliced = zz.into_array().slice(slice_start..slice_end)?;
        let expected: Vec<u32> = all_decoded[slice_start..slice_end].to_vec();

        let cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())?;
        let (plan, _bufs) = build_plan(&sliced, &cuda_ctx)?;

        let actual = run_dynamic_dispatch_plan(&cuda_ctx, expected.len(), &plan)?;
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
        let (plan, _bufs) = build_plan(&sliced, &cuda_ctx)?;

        let actual = run_dynamic_dispatch_plan(&cuda_ctx, expected.len(), &plan)?;
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
        let bp = BitPackedArray::encode(&prim.into_array(), bit_width)?;

        let sliced = bp.into_array().slice(slice_start..slice_end)?;
        let expected: Vec<u32> = data[slice_start..slice_end].to_vec();

        let cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())?;
        let (plan, _bufs) = build_plan(&sliced, &cuda_ctx)?;

        let actual = run_dynamic_dispatch_plan(&cuda_ctx, expected.len(), &plan)?;
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
        let bp = BitPackedArray::encode(&prim.into_array(), bit_width)?;
        let for_arr = FoRArray::try_new(bp.into_array(), Scalar::from(reference))?;

        let all_decoded: Vec<u32> = encoded_data.iter().map(|&v| v + reference).collect();

        let sliced = for_arr.into_array().slice(slice_start..slice_end)?;
        let expected: Vec<u32> = all_decoded[slice_start..slice_end].to_vec();

        let cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())?;
        let (plan, _bufs) = build_plan(&sliced, &cuda_ctx)?;

        let actual = run_dynamic_dispatch_plan(&cuda_ctx, expected.len(), &plan)?;
        assert_eq!(actual, expected);

        Ok(())
    }

    #[rstest]
    #[case(0, 1024)]
    #[case(0, 3000)]
    #[case(0, 4096)]
    #[case(400, 600)]
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
    fn test_sliced_runend(
        #[case] slice_start: usize,
        #[case] slice_end: usize,
    ) -> VortexResult<()> {
        let ends: Vec<u32> = vec![500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000];
        let values: Vec<u32> = vec![10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
        let len = 5000;

        let all_decoded: Vec<u32> = (0..len)
            .map(|i| {
                let run = ends.iter().position(|&e| (i as u32) < e).unwrap();
                values[run]
            })
            .collect();

        let ends_arr = PrimitiveArray::new(Buffer::from(ends), NonNullable).into_array();
        let values_arr = PrimitiveArray::new(Buffer::from(values), NonNullable).into_array();
        let re = RunEndArray::new(ends_arr, values_arr);

        let sliced = re.into_array().slice(slice_start..slice_end)?;
        let expected: Vec<u32> = all_decoded[slice_start..slice_end].to_vec();

        let cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())?;
        let (plan, _bufs) = build_plan(&sliced, &cuda_ctx)?;

        let actual = run_dynamic_dispatch_plan(&cuda_ctx, expected.len(), &plan)?;
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
        let dict_bp = BitPackedArray::encode(&dict_prim.into_array(), 6)?;
        let dict_for = FoRArray::try_new(dict_bp.into_array(), Scalar::from(dict_reference))?;

        // BitPack the codes
        let codes_prim = PrimitiveArray::new(Buffer::from(codes), NonNullable);
        let codes_bp = BitPackedArray::encode(&codes_prim.into_array(), 6)?;

        let dict = DictArray::try_new(codes_bp.into_array(), dict_for.into_array())?;

        let sliced = dict.into_array().slice(slice_start..slice_end)?;
        let expected: Vec<u32> = all_decoded[slice_start..slice_end].to_vec();

        let cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())?;
        let (plan, _bufs) = build_plan(&sliced, &cuda_ctx)?;

        let actual = run_dynamic_dispatch_plan(&cuda_ctx, expected.len(), &plan)?;
        assert_eq!(actual, expected);

        Ok(())
    }
}
