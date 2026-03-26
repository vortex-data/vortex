// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Walks an encoding tree and produces a dispatch plan for a single GPU kernel launch.
//! [`UnmaterializedPlan::new`] builds the plan without a CUDA context, then
//! [`materialize`](UnmaterializedPlan::materialize) copies buffers to the device.
//!
//! For partially-fusable trees (where some nodes can't be fused into the
//! dispatch plan), the free function [`with_subtree_inputs`] builds a plan
//! that incorporates pre-executed subtree outputs as `LOAD` sources.
//!
//! The two-phase design allows callers to query shared memory requirements
//! before committing to device allocation, and keeps the bulk of the logic
//! independent of the CUDA runtime.
//!
//! # Known limitations
//!
//! TODO(0ax1): Optimize device buffer allocation and copying.
//!
//! Ideally, there would be a buffer pool of preallocated device memory such
//! that retrieving a device pointer is O(1) during materialization. In the
//! current setup, we allocate via the global allocator, which does not pin
//! host memory to physical addresses (unlike `cudaHostAlloc`). This means
//! the host-to-device copy is synchronous and cannot be pushed to the CUDA
//! stream as an async operation.

use std::sync::Arc;

use vortex::array::ArrayRef;
use vortex::array::ArrayVisitor;
use vortex::array::DynArray;
use vortex::array::ExecutionCtx;
use vortex::array::arrays::Dict;
use vortex::array::arrays::Primitive;
use vortex::array::arrays::Slice;
use vortex::array::arrays::primitive::PrimitiveArrayParts;
use vortex::array::buffer::BufferHandle;
use vortex::array::session::ArraySession;
use vortex::dtype::PType;
use vortex::encodings::alp::ALP;
use vortex::encodings::alp::ALPFloat;
use vortex::encodings::fastlanes::BitPacked;
use vortex::encodings::fastlanes::BitPackedArrayParts;
use vortex::encodings::fastlanes::FoR;
use vortex::encodings::fastlanes::FoRArray;
use vortex::encodings::runend::RunEnd;
use vortex::encodings::runend::RunEndArrayParts;
use vortex::encodings::sequence::Sequence;
use vortex::encodings::sequence::SequenceArrayParts;
use vortex::encodings::zigzag::ZigZag;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_err;
use vortex::session::VortexSession;

use super::CudaDispatchPlan;
use super::MAX_SCALAR_OPS;
use super::MAX_STAGES;
use super::SMEM_TILE_SIZE;
use super::ScalarOp;
use super::SourceOp;
use super::Stage;
use crate::CudaBufferExt;
use crate::CudaExecutionCtx;

/// A plan whose source buffers have been copied to the device, ready for kernel launch.
pub struct MaterializedPlan {
    /// The C ABI plan struct, ready to upload to the device.
    pub dispatch_plan: CudaDispatchPlan,
    /// Device buffer handles that must be kept alive while the plan is in use.
    pub device_buffers: Vec<BufferHandle>,
    /// Dynamic shared memory bytes needed to launch this plan.
    pub shared_mem_bytes: u32,
}

/// Find encoding-tree nodes that cannot be fused into a dynamic-dispatch plan.
///
/// Each returned node is the root of a branch that must be executed by a
/// separate kernel. Their outputs can then be fed into
/// [`UnmaterializedPlan::new_with_subtree_inputs`] as `LOAD` sources.
///
/// Returns an empty vec if the root itself is not fusable.
pub fn find_unfusable_nodes(array: &ArrayRef) -> Vec<ArrayRef> {
    if !is_dyn_dispatch_compatible(array) {
        return Vec::new();
    }
    let mut out = Vec::new();
    for child in array.children() {
        if is_dyn_dispatch_compatible(&child) {
            out.extend(find_unfusable_nodes(&child));
        } else {
            out.push(child);
        }
    }
    out
}

/// Checks whether the encoding of an array can be fused into a dynamic-dispatch plan.
fn is_dyn_dispatch_compatible(array: &ArrayRef) -> bool {
    let id = array.encoding_id();
    if id == ALP::ID {
        if let Ok(a) = array.clone().try_into::<ALP>() {
            return a.patches().is_none() && a.dtype().as_ptype() == PType::F32;
        }
        return false;
    }
    if id == BitPacked::ID {
        if let Ok(a) = array.clone().try_into::<BitPacked>() {
            return a.patches().is_none();
        }
        return false;
    }
    if id == Dict::ID {
        if let Ok(a) = array.clone().try_into::<Dict>() {
            // As of now the dict dyn dispatch kernel requires
            // codes and values to have the same byte width.
            return match (
                PType::try_from(a.values().dtype()),
                PType::try_from(a.codes().dtype()),
            ) {
                (Ok(values), Ok(codes)) => values.byte_width() == codes.byte_width(),
                _ => false,
            };
        }
        return false;
    }
    if id == RunEnd::ID {
        if let Ok(a) = array.clone().try_into::<RunEnd>() {
            // As of now the run-end dyn dispatch kernel requires
            // ends and values to have the same byte width.
            return match (
                PType::try_from(a.ends().dtype()),
                PType::try_from(a.values().dtype()),
            ) {
                (Ok(e), Ok(v)) => e.byte_width() == v.byte_width(),
                _ => false,
            };
        }
        return false;
    }
    id == FoR::ID
        || id == ZigZag::ID
        || id == Primitive::ID
        || id == Slice::ID
        || id == Sequence::ID
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

/// An unmaterialized stage: a source op, scalar ops, and optional source buffer reference.
struct UnmaterializedStage {
    source: SourceOp,
    scalar_ops: Vec<ScalarOp>,
    /// Index into `UnmaterializedPlan::source_buffers`, or `None`
    /// for sources that don't read from a device buffer.
    source_buffer_index: Option<usize>,
}

impl UnmaterializedStage {
    fn new(source: SourceOp, source_buffer_index: Option<usize>) -> Self {
        Self {
            source,
            scalar_ops: vec![],
            source_buffer_index,
        }
    }
}

type SmemOffset = u32;
type StageLen = u32;
type ArrayId = usize;
type SourceBufferIdx = usize;

/// A dispatch plan before device materialization.
///
/// Created by [`UnmaterializedPlan::new`] or [`new_with_subtree_inputs`](Self::new_with_subtree_inputs).
/// Query shared memory requirements via [`shared_mem_bytes`](Self::shared_mem_bytes),
/// then copy source buffers to the device via [`materialize`](Self::materialize).
pub struct UnmaterializedPlan {
    /// Input stages followed by one output stage.
    stages: Vec<(UnmaterializedStage, SmemOffset, StageLen)>,
    smem_cursor: SmemOffset,
    source_buffers: Vec<BufferHandle>,
    elem_bytes: u32,
    subtree_inputs: Vec<(ArrayId, SourceBufferIdx)>,
}

impl UnmaterializedPlan {
    /// Construct a plan by walking the encoding tree from root to leaf.
    ///
    /// No CUDA context is needed — source buffers are extracted but not
    /// yet copied to the device. Call [`materialize`](Self::materialize)
    /// to do the device copy.
    ///
    /// # Limitations
    ///
    /// - Validity bitmaps are ignored; only `NonNullable`/`AllValid` is supported.
    /// - `BitPackedArray` and `ALPArray` with patches are not supported.
    /// - Only f32 ALP is supported (kernel stores multipliers as `float`).
    pub fn new(array: &ArrayRef) -> VortexResult<Self> {
        Self::new_with_subtree_inputs(array, &[])
    }

    /// Build an [`UnmaterializedPlan`] with pre-executed subtree outputs
    /// injected as `LOAD` sources.
    ///
    /// Used by hybrid dispatch when some nodes in the encoding tree cannot
    /// be fused. Those nodes are executed separately, and their device buffers
    /// are passed here so the remaining fusable tree can reference them.
    pub fn new_with_subtree_inputs(
        array: &ArrayRef,
        subtree_inputs: &[(ArrayRef, BufferHandle)],
    ) -> VortexResult<UnmaterializedPlan> {
        let elem_bytes = PType::try_from(array.dtype())
            .map_err(|_| {
                vortex_err!(
                    "dyn dispatch requires primitive dtype, got {:?}",
                    array.dtype()
                )
            })?
            .byte_width() as u32;

        let subtree_map: Vec<(ArrayId, SourceBufferIdx)> = subtree_inputs
            .iter()
            .enumerate()
            .map(|(leaf_idx, (arr, _handle))| (Arc::as_ptr(arr) as *const () as usize, leaf_idx))
            .collect();

        // Subtree source buffers get indices 0..n
        let source_buffers: Vec<BufferHandle> = subtree_inputs
            .iter()
            .map(|(_, handle)| handle.clone())
            .collect();

        let mut plan = Self {
            stages: Vec::new(),
            smem_cursor: SmemOffset::from(0u32),
            source_buffers,
            elem_bytes,
            subtree_inputs: subtree_map,
        };

        let len = array.len() as u32;
        let output = plan.walk(array.clone())?;
        plan.stages.push((output, plan.smem_cursor, len));

        assert!(plan.stages.len() <= MAX_STAGES as usize);
        assert!(
            plan.stages
                .iter()
                .all(|(s, ..)| (s.scalar_ops.len() as u32) <= MAX_SCALAR_OPS)
        );

        Ok(plan)
    }

    /// Shared memory bytes needed to launch this plan.
    ///
    /// Shared memory holds the *output* of each stage so later stages can
    /// reference it (e.g., dictionary values, run-end endpoints). The total
    /// is the sum of all input stage lengths plus the output tile size,
    /// multiplied by the element byte width.
    pub fn shared_mem_bytes(&self) -> u32 {
        (self.smem_cursor + SMEM_TILE_SIZE) * self.elem_bytes
    }

    /// Copy source buffers to the device, producing a [`MaterializedPlan`].
    pub fn materialize(self, ctx: &CudaExecutionCtx) -> VortexResult<MaterializedPlan> {
        let shared_mem_bytes = self.shared_mem_bytes();
        let mut device_buffers = Vec::new();
        let mut device_ptrs: Vec<u64> = Vec::new();

        // Copy each source buffer to the device and record its pointer.
        for source_buf in self.source_buffers {
            let device_buf = ctx.ensure_on_device_sync(source_buf)?;
            let ptr = device_buf.cuda_device_ptr()?;
            device_ptrs.push(ptr);
            device_buffers.push(device_buf);
        }

        // Resolve the device pointer for a stage's source buffer.
        // RUNEND and SEQUENCE sources don't read from global memory —
        // they use shared memory or generate data on the fly, so
        // `input_ptr = 0` is safe (the kernel never dereferences it).
        let resolve_ptr = |stage: &UnmaterializedStage| -> u64 {
            match stage.source_buffer_index {
                Some(idx) => device_ptrs[idx],
                None => 0,
            }
        };

        let mut stages = Vec::with_capacity(self.stages.len());

        for (stage, smem_offset, len) in &self.stages {
            stages.push(Stage::new(
                resolve_ptr(stage),
                *smem_offset,
                *len,
                stage.source,
                &stage.scalar_ops,
            ));
        }

        Ok(MaterializedPlan {
            dispatch_plan: CudaDispatchPlan::new(stages),
            device_buffers,
            shared_mem_bytes,
        })
    }

    /// Walk the encoding tree, producing an [`UnmaterializedStage`] for the root.
    fn walk(&mut self, array: ArrayRef) -> VortexResult<UnmaterializedStage> {
        // Check if this array matches a pre-executed subtree input.
        let subtree_id = Arc::as_ptr(&array) as *const () as usize;
        if let Some((_, buf_idx)) = self.subtree_inputs.iter().find(|(id, _)| *id == subtree_id) {
            return Ok(UnmaterializedStage::new(SourceOp::load(), Some(*buf_idx)));
        }

        if !is_dyn_dispatch_compatible(&array) {
            vortex_bail!(
                "Encoding {:?} is not compatible with the dynamic dispatch plan builder",
                array.encoding_id()
            );
        }

        let id = array.encoding_id();

        if id == BitPacked::ID {
            self.walk_bitpacked(array)
        } else if id == FoR::ID {
            self.walk_for(array)
        } else if id == ZigZag::ID {
            self.walk_zigzag(array)
        } else if id == ALP::ID {
            self.walk_alp(array)
        } else if id == Dict::ID {
            self.walk_dict(array)
        } else if id == RunEnd::ID {
            self.walk_runend(array)
        } else if id == Primitive::ID {
            self.walk_primitive(array)
        } else if id == Slice::ID {
            self.walk_slice(array)
        } else if id == Sequence::ID {
            self.walk_sequence(array)
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
    fn walk_slice(&mut self, array: ArrayRef) -> VortexResult<UnmaterializedStage> {
        let slice_arr = array.as_::<Slice>();
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

    fn walk_primitive(&mut self, array: ArrayRef) -> VortexResult<UnmaterializedStage> {
        let prim = array.to_canonical()?.into_primitive();
        let PrimitiveArrayParts { buffer, .. } = prim.into_parts();
        let buf_index = self.source_buffers.len();
        self.source_buffers.push(buffer);
        Ok(UnmaterializedStage::new(SourceOp::load(), Some(buf_index)))
    }

    fn walk_bitpacked(&mut self, array: ArrayRef) -> VortexResult<UnmaterializedStage> {
        let bp = array
            .try_into::<BitPacked>()
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

        let buf_index = self.source_buffers.len();
        self.source_buffers.push(packed);
        Ok(UnmaterializedStage::new(
            SourceOp::bitunpack(bit_width, offset),
            Some(buf_index),
        ))
    }

    fn walk_for(&mut self, array: ArrayRef) -> VortexResult<UnmaterializedStage> {
        let for_arr = array
            .try_into::<FoR>()
            .map_err(|_| vortex_err!("Expected FoRArray"))?;
        let ref_u64 = extract_for_reference(&for_arr)?;
        let encoded = for_arr.encoded().clone();

        let mut pipeline = self.walk(encoded)?;
        pipeline.scalar_ops.push(ScalarOp::frame_of_ref(ref_u64));
        Ok(pipeline)
    }

    fn walk_zigzag(&mut self, array: ArrayRef) -> VortexResult<UnmaterializedStage> {
        let zz = array
            .try_into::<ZigZag>()
            .map_err(|_| vortex_err!("Expected ZigZagArray"))?;
        let encoded = zz.encoded().clone();

        let mut pipeline = self.walk(encoded)?;
        pipeline.scalar_ops.push(ScalarOp::zigzag());
        Ok(pipeline)
    }

    fn walk_alp(&mut self, array: ArrayRef) -> VortexResult<UnmaterializedStage> {
        let alp = array
            .try_into::<ALP>()
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

    fn walk_dict(&mut self, array: ArrayRef) -> VortexResult<UnmaterializedStage> {
        let dict = array
            .try_into::<Dict>()
            .map_err(|_| vortex_err!("Expected DictArray"))?;
        let values = dict.values().clone();
        let codes = dict.codes().clone();

        let values_smem_offset = self.add_input_stage(values)?;

        let mut pipeline = self.walk(codes)?;
        pipeline.scalar_ops.push(ScalarOp::dict(values_smem_offset));
        Ok(pipeline)
    }

    fn walk_sequence(&mut self, array: ArrayRef) -> VortexResult<UnmaterializedStage> {
        let seq = array
            .try_into::<Sequence>()
            .map_err(|_| vortex_err!("Expected SequenceArray"))?;
        let SequenceArrayParts {
            base, multiplier, ..
        } = seq.into_parts();

        Ok(UnmaterializedStage::new(
            SourceOp::sequence(base.cast()?, multiplier.cast()?),
            None,
        ))
    }

    fn walk_runend(&mut self, array: ArrayRef) -> VortexResult<UnmaterializedStage> {
        let re = array
            .try_into::<RunEnd>()
            .map_err(|_| vortex_err!("Expected RunEndArray"))?;
        let offset = re.offset() as u64;
        let RunEndArrayParts { ends, values } = re.into_parts();
        let num_runs = ends.len() as u64;

        let ends_smem = self.add_input_stage(ends)?;
        let values_smem = self.add_input_stage(values)?;

        Ok(UnmaterializedStage::new(
            SourceOp::runend(ends_smem, values_smem, num_runs, offset),
            None,
        ))
    }

    fn add_input_stage(&mut self, array: ArrayRef) -> VortexResult<u32> {
        let smem_offset = self.smem_cursor;
        let len = array.len() as u32;
        let spec = self.walk(array)?;
        self.stages.push((spec, smem_offset, len));
        self.smem_cursor += len;
        Ok(smem_offset)
    }
}
