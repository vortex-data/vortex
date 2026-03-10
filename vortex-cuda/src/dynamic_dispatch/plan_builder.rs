// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Walks an encoding tree and builds a [`DynamicDispatchPlan`].
//!
//! The builder recursively inspects the array's encoding, moves leaf buffers
//! to the device, computes shared memory offsets, and produces a plan that the
//! dynamic dispatch kernel can execute in a single launch.

use futures::executor::block_on;
use vortex::array::ArrayRef;
use vortex::array::DynArray;
use vortex::array::ExecutionCtx;
use vortex::array::arrays::DictVTable;
use vortex::array::arrays::PrimitiveVTable;
use vortex::array::arrays::SliceVTable;
use vortex::array::arrays::primitive::PrimitiveArrayParts;
use vortex::array::buffer::BufferHandle;
use vortex::array::session::ArraySession;
use vortex::dtype::PType;
use vortex::encodings::alp::ALPFloat;
use vortex::encodings::alp::ALPVTable;
use vortex::encodings::fastlanes::BitPackedArrayParts;
use vortex::encodings::fastlanes::BitPackedVTable;
use vortex::encodings::fastlanes::FoRArray;
use vortex::encodings::fastlanes::FoRVTable;
use vortex::encodings::runend::RunEndArrayParts;
use vortex::encodings::runend::RunEndVTable;
use vortex::encodings::zigzag::ZigZagVTable;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_err;
use vortex::session::VortexSession;

use super::DynamicDispatchPlan;
use super::MAX_SCALAR_OPS;
use super::MAX_STAGES;
use super::ScalarOp;
use super::SourceOp;
use super::Stage;
use crate::CudaBufferExt;
use crate::CudaExecutionCtx;

/// The result of walking a subtree: a source op, scalar ops to apply after,
/// and the device pointer to the leaf buffer.
struct Pipeline {
    source: SourceOp,
    scalar_ops: Vec<ScalarOp>,
    input_ptr: u64,
}

/// Walk the encoding tree of `array` and build a [`DynamicDispatchPlan`].
///
/// Leaf buffers are moved to the device if not already there. The returned
/// buffer handles must be kept alive while the plan's device pointers are
/// in use.
///
/// # Plan construction
///
/// The builder walks the encoding tree from root to leaf. Single-child
/// encodings (FoR, ZigZag, ALP) recurse into their child and append an
/// element-wise transform to the pipeline. Leaf encodings (BitPacked,
/// Primitive) produce a source op and a device pointer.
///
/// Encodings with multiple children (Dict, RunEnd) emit an input stage
/// for each child, writing its output to shared memory. The root of the
/// tree becomes the final output stage, which writes directly to global
/// memory instead.
///
/// Shared memory offsets are bump-allocated: each input stage claims
/// the next available region. Since the output stage may reference any
/// input stage's output (e.g., dictionary lookup, run-end resolution),
/// all regions must coexist simultaneously — the total shared memory
/// is `max(smem_offset + len) * sizeof(T)` across all stages.
///
/// # Supported encodings
///
/// - `PrimitiveArray` → `LOAD` source
/// - `BitPackedArray` → `BITUNPACK` source (no patches)
/// - `FoRArray` → recurse + `FoR` scalar op
/// - `ZigZagArray` → recurse + `ZigZag` scalar op
/// - `ALPArray` → recurse + `ALP` scalar op (f32 only, no patches)
/// - `DictArray` → input stage for values + recurse codes + `DICT` scalar op
/// - `RunEndArray` → input stages for ends/values + `RUNEND` source
/// - `SliceArray` → resolve via child's slice reduce/kernel
///
/// # Limitations
///
/// **Nullability**: validity bitmaps are silently ignored. All output elements
/// receive a value regardless of whether the input was null. Only arrays with
/// `NonNullable` or `AllValid` validity produce correct results.
///
/// **Patches**: `BitPackedArray` with patches and `ALPArray` with patches are
/// not supported and will return an error.
///
/// **f64 ALP**: Only f32 ALP is supported. The CUDA kernel's `AlpParams`
/// stores multipliers as `float`, so f64 ALP arrays will return an error.
pub fn build_plan(
    array: &ArrayRef,
    ctx: &CudaExecutionCtx,
) -> VortexResult<(DynamicDispatchPlan, Vec<BufferHandle>)> {
    let mut state = PlanBuilderState {
        ctx,
        stages: Vec::new(),
        smem_cursor: 0,
        device_buffers: Vec::new(),
    };

    let pipeline = state.walk(array.clone())?;
    let output_stage = Stage::output(
        pipeline.input_ptr,
        state.smem_cursor,
        pipeline.source,
        &pipeline.scalar_ops,
    );
    state.stages.push(output_stage);

    assert!(state.stages.len() <= MAX_STAGES as usize);
    assert!(
        state
            .stages
            .iter()
            .all(|&stage| (stage.num_scalar_ops as u32) <= MAX_SCALAR_OPS)
    );

    Ok((DynamicDispatchPlan::new(state.stages), state.device_buffers))
}

/// Internal mutable state for the recursive tree walk.
struct PlanBuilderState<'a> {
    ctx: &'a CudaExecutionCtx,
    /// Stages to process in the dynamic dispatch kernel.
    stages: Vec<Stage>,
    /// Next available element offset in shared memory.
    smem_cursor: u32,
    /// Device buffers to keep alive.
    device_buffers: Vec<BufferHandle>,
}

impl PlanBuilderState<'_> {
    /// Recursively walk the encoding tree.
    fn walk(&mut self, array: ArrayRef) -> VortexResult<Pipeline> {
        let id = array.encoding_id();

        if id == BitPackedVTable::ID {
            self.walk_bitpacked(array)
        } else if id == FoRVTable::ID {
            self.walk_for(array)
        } else if id == ZigZagVTable::ID {
            self.walk_zigzag(array)
        } else if id == ALPVTable::ID {
            self.walk_alp(array)
        } else if id == DictVTable::ID {
            self.walk_dict(array)
        } else if id == RunEndVTable::ID {
            self.walk_runend(array)
        } else if id == PrimitiveVTable::ID {
            self.walk_primitive(array)
        } else if id == SliceVTable::ID {
            self.walk_slice(array)
        } else {
            vortex_bail!(
                "Encoding {:?} not supported by dynamic dispatch plan builder",
                id
            )
        }
    }

    /// SliceArray → resolve the slice via reduce/execute rules.
    ///
    /// When the plan builder encounters a `SliceArray`, it resolves the slice
    /// by invoking the child's `reduce_parent`, `execute_parent`.
    fn walk_slice(&mut self, array: ArrayRef) -> VortexResult<Pipeline> {
        let slice_arr = array.as_::<SliceVTable>();
        let child = slice_arr.child().clone();

        // reduce_parent: (for types with SliceReduceAdaptor, like FoR/ZigZag)
        if let Some(reduced) = child.vtable().reduce_parent(&child, &array, 0)? {
            return self.walk(reduced);
        }

        // execute_parent: (for types with SliceExecuteAdaptor/SliceKernel, like BitPacked)
        let mut ctx = ExecutionCtx::new(VortexSession::empty().with::<ArraySession>());
        if let Some(executed) = child.vtable().execute_parent(&child, &array, 0, &mut ctx)? {
            return self.walk(executed);
        }

        vortex_bail!(
            "Cannot resolve SliceArray wrapping {:?} in dynamic dispatch plan builder",
            child.encoding_id()
        )
    }

    /// Canonical primitive array → LOAD source op.
    ///
    /// The device pointer accounts for buffer slicing, so no offset parameter is needed.
    fn walk_primitive(&mut self, array: ArrayRef) -> VortexResult<Pipeline> {
        let prim = array.to_canonical()?.into_primitive();
        let PrimitiveArrayParts { buffer, .. } = prim.into_parts();
        let device_buf = block_on(self.ctx.ensure_on_device(buffer))?;
        let ptr = device_buf.cuda_device_ptr()?;
        self.device_buffers.push(device_buf);
        Ok(Pipeline {
            source: SourceOp::load(),
            scalar_ops: vec![],
            input_ptr: ptr as u64,
        })
    }

    /// BitPackedArray → BITUNPACK source op.
    ///
    /// The sub-byte element offset (0..=1023) is passed as a kernel parameter
    /// as it cannot be expressed as pointer arithmetic on the device pointer.
    fn walk_bitpacked(&mut self, array: ArrayRef) -> VortexResult<Pipeline> {
        let bp = array
            .try_into::<BitPackedVTable>()
            .map_err(|_| vortex_err!("Expected BitPackedArray"))?;
        let BitPackedArrayParts {
            offset,
            bit_width,
            packed,
            patches,
            ..
        } = bp.into_parts();

        if patches.is_some() {
            vortex_bail!("Dynamic dispatch does not support BitPackedArray with patches");
        }

        let device_buf = block_on(self.ctx.ensure_on_device(packed))?;
        let ptr = device_buf.cuda_device_ptr()?;
        self.device_buffers.push(device_buf);
        Ok(Pipeline {
            source: SourceOp::bitunpack(bit_width, offset),
            scalar_ops: vec![],
            input_ptr: ptr as u64,
        })
    }

    /// FoRArray → recurse into encoded child, add FoR scalar op.
    fn walk_for(&mut self, array: ArrayRef) -> VortexResult<Pipeline> {
        let for_arr = array
            .try_into::<FoRVTable>()
            .map_err(|_| vortex_err!("Expected FoRArray"))?;
        let ref_u64 = extract_for_reference(&for_arr)?;
        let encoded = for_arr.encoded().clone();

        let mut pipeline = self.walk(encoded)?;
        pipeline.scalar_ops.push(ScalarOp::frame_of_ref(ref_u64));
        Ok(pipeline)
    }

    /// ZigZagArray → recurse into encoded child, add ZigZag scalar op.
    fn walk_zigzag(&mut self, array: ArrayRef) -> VortexResult<Pipeline> {
        let zz = array
            .try_into::<ZigZagVTable>()
            .map_err(|_| vortex_err!("Expected ZigZagArray"))?;
        let encoded = zz.encoded().clone();

        let mut pipeline = self.walk(encoded)?;
        pipeline.scalar_ops.push(ScalarOp::zigzag());
        Ok(pipeline)
    }

    /// ALPArray → recurse into encoded child, add ALP scalar op (f32 only).
    fn walk_alp(&mut self, array: ArrayRef) -> VortexResult<Pipeline> {
        let alp = array
            .try_into::<ALPVTable>()
            .map_err(|_| vortex_err!("Expected ALPArray"))?;

        if alp.patches().is_some() {
            vortex_bail!("Dynamic dispatch does not support ALPArray with patches");
        }

        let ptype = alp.dtype().as_ptype();
        if ptype != PType::F32 {
            vortex_bail!(
                "Dynamic dispatch only supports f32 ALP, got {:?}",
                alp.dtype()
            );
        }

        let exponents = alp.exponents();
        let alp_f = <f32 as ALPFloat>::F10[exponents.f as usize];
        let alp_e = <f32 as ALPFloat>::IF10[exponents.e as usize];
        let encoded = alp.encoded().clone();

        let mut pipeline = self.walk(encoded)?;
        pipeline.scalar_ops.push(ScalarOp::alp(alp_f, alp_e));
        Ok(pipeline)
    }

    /// DictArray → add input stage for values, recurse into codes, add DICT scalar op.
    fn walk_dict(&mut self, array: ArrayRef) -> VortexResult<Pipeline> {
        let dict = array
            .try_into::<DictVTable>()
            .map_err(|_| vortex_err!("Expected DictArray"))?;
        let values = dict.values().clone();
        let codes = dict.codes().clone();

        let values_smem_offset = self.add_input_stage(values)?;

        let mut pipeline = self.walk(codes)?;
        pipeline.scalar_ops.push(ScalarOp::dict(values_smem_offset));
        Ok(pipeline)
    }

    /// RunEndArray → add input stages for ends and values, RUNEND source op.
    fn walk_runend(&mut self, array: ArrayRef) -> VortexResult<Pipeline> {
        let re = array
            .try_into::<RunEndVTable>()
            .map_err(|_| vortex_err!("Expected RunEndArray"))?;
        let offset = re.offset() as u64;
        let RunEndArrayParts { ends, values } = re.into_parts();
        let num_runs = ends.len() as u64;

        let ends_smem = self.add_input_stage(ends)?;
        let values_smem = self.add_input_stage(values)?;

        Ok(Pipeline {
            source: SourceOp::runend(ends_smem, values_smem, num_runs, offset),
            scalar_ops: vec![],
            input_ptr: 0,
        })
    }

    /// Recursively walk `array` and add it as an input stage in shared memory.
    /// Claims the next `array.len()` elements from the bump allocator and
    /// returns the smem element offset where this stage's output begins.
    fn add_input_stage(&mut self, array: ArrayRef) -> VortexResult<u32> {
        let smem_offset = self.smem_cursor;
        let len = array.len() as u32;
        let pipeline = self.walk(array)?;
        self.stages.push(Stage::input(
            pipeline.input_ptr,
            smem_offset,
            len,
            pipeline.source,
            &pipeline.scalar_ops,
        ));
        self.smem_cursor += len;
        Ok(smem_offset)
    }
}

/// Extract a FoR reference scalar as u64 bits.
fn extract_for_reference(for_arr: &FoRArray) -> VortexResult<u64> {
    if let Ok(v) = u32::try_from(for_arr.reference_scalar()) {
        Ok(v as u64)
    } else if let Ok(v) = i32::try_from(for_arr.reference_scalar()) {
        Ok(v as u32 as u64)
    } else if let Ok(v) = u64::try_from(for_arr.reference_scalar()) {
        Ok(v)
    } else if let Ok(v) = i64::try_from(for_arr.reference_scalar()) {
        Ok(v as u64)
    } else {
        vortex_bail!("Cannot extract FoR reference as an integer type")
    }
}
