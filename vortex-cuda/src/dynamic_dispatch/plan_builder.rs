// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Walks an encoding tree and produces a [`DispatchPlan`] for a single GPU
//! kernel launch. The tree is inspected in a single pass, identifying unfusable
//! subtrees and computing shared memory requirements upfront — before any
//! device allocation or kernel work.

use itertools::zip_eq;
use tracing::trace;
use vortex::array::ArrayRef;
use vortex::array::arrays::Dict;
use vortex::array::arrays::Primitive;
use vortex::array::arrays::Slice;
use vortex::array::arrays::dict::DictArraySlotsExt;
use vortex::array::arrays::slice::SliceArrayExt;
use vortex::array::buffer::BufferHandle;
use vortex::array::validity::Validity;
use vortex::dtype::PType;
use vortex::encodings::alp::ALP;
use vortex::encodings::alp::ALPArrayExt;
use vortex::encodings::alp::ALPArraySlotsExt;
use vortex::encodings::alp::ALPFloat;
use vortex::encodings::fastlanes::BitPacked;
use vortex::encodings::fastlanes::BitPackedArrayExt;
use vortex::encodings::fastlanes::FoR;
use vortex::encodings::fastlanes::FoRArrayExt;
use vortex::encodings::runend::RunEnd;
use vortex::encodings::runend::RunEndArrayExt;
use vortex::encodings::sequence::Sequence;
use vortex::encodings::zigzag::ZigZag;
use vortex::encodings::zigzag::ZigZagArrayExt;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_err;

use super::CudaDispatchPlan;
use super::MaterializedStage;
use super::PTypeTag;
use super::SMEM_TILE_SIZE;
use super::ScalarOp;
use super::SourceOp;
use super::ptype_to_tag;
use super::tag_to_ptype;
use crate::CudaBufferExt;
use crate::CudaExecutionCtx;

/// A plan whose source buffers have been copied to the device, ready for kernel launch.
pub struct MaterializedPlan {
    /// Packed plan byte buffer, to upload to the device.
    pub dispatch_plan: CudaDispatchPlan,
    /// Device buffer handles that must be kept alive while the plan is in use.
    pub device_buffers: Vec<BufferHandle>,
    /// Dynamic shared memory bytes needed to launch this plan.
    pub shared_mem_bytes: u32,
    /// Validity of the root array, propagated to the output.
    pub validity: Validity,
}

/// Checks whether the encoding of an array can be fused into a dynamic-dispatch plan.
fn is_dyn_dispatch_compatible(array: &ArrayRef) -> bool {
    // The dynamic dispatch kernel only supports F32 floats (via ALP).
    // F16 and F64 have no reinterpret path in the kernel.
    if matches!(PType::try_from(array.dtype()), Ok(PType::F16 | PType::F64)) {
        return false;
    }

    let id = array.encoding_id();
    if id == ALP::ID {
        let arr = array.as_::<ALP>();
        return arr.patches().is_none() && arr.dtype().as_ptype() == PType::F32;
    }
    if id == BitPacked::ID {
        return array.as_::<BitPacked>().patches().is_none();
    }
    if id == Dict::ID {
        let arr = array.as_::<Dict>();
        // Nullable codes could hold garbage values at null positions, causing
        // out-of-bounds shared memory reads in the DICT gather scalar op.
        if arr.codes().dtype().is_nullable() {
            return false;
        }
        // Dict codes and values may have different byte widths.
        // The kernel handles mixed widths via widening input stages,
        // but only when codes are no wider than values (the output type).
        // Wider codes would be truncated by load_element<T>().
        let values_ptype = PType::try_from(arr.values().dtype());
        let codes_ptype = PType::try_from(arr.codes().dtype());
        return match (values_ptype, codes_ptype) {
            (Ok(vp), Ok(cp)) => cp.byte_width() <= vp.byte_width(),
            _ => false,
        };
    }
    if id == RunEnd::ID {
        let arr = array.as_::<RunEnd>();
        // Nullable ends could hold garbage values at null positions, causing
        // unpredictable binary search / forward-scan behavior in the RUNEND
        // source op.
        if arr.ends().dtype().is_nullable() {
            return false;
        }
        // RunEnd ends and values may have different byte widths.
        // The kernel handles mixed widths via widening input stages,
        // but only when ends are no wider than values (the output type).
        // Wider ends would be truncated by load_element<T>().
        let ends_ptype = PType::try_from(arr.ends().dtype());
        let values_ptype = PType::try_from(arr.values().dtype());
        return match (ends_ptype, values_ptype) {
            (Ok(ep), Ok(vp)) => ep.byte_width() <= vp.byte_width(),
            _ => false,
        };
    }
    id == FoR::ID
        || id == ZigZag::ID
        || id == Primitive::ID
        || id == Slice::ID
        || id == Sequence::ID
}

/// An unmaterialized stage: a source op, scalar ops, and optional source buffer reference.
struct Stage {
    source: SourceOp,
    scalar_ops: Vec<ScalarOp>,
    /// Index into `FusedPlan::source_buffers`, or `None`
    /// for sources that don't read from a device buffer.
    source_buffer_index: Option<usize>,
    /// PType tag for the source op's output type.
    source_ptype: PTypeTag,
}

impl Stage {
    fn new(source: SourceOp, source_buffer_index: Option<usize>, source_ptype: PTypeTag) -> Self {
        Self {
            source,
            scalar_ops: vec![],
            source_buffer_index,
            source_ptype,
        }
    }
}

type SmemByteOffset = u32;
type OutputLen = u32;

/// A dispatch plan before device materialization.
///
/// Constructed by [`DispatchPlan::new`], which inspects the encoding tree
/// and determines whether it can be fully fused, partially fused, or not fused at all.
pub enum DispatchPlan {
    /// Entire encoding tree is fusable into a single kernel launch.
    Fused(FusedPlan),
    /// Some subtrees need separate execution before the fused plan can run.
    PartiallyFused {
        /// The fused plan (with placeholder buffer slots for pending subtrees).
        plan: FusedPlan,
        /// Unfusable subtree roots that must be executed separately.
        pending_subtrees: Vec<ArrayRef>,
    },
    /// Tree cannot be fused (incompatible root, non-primitive subtree dtypes,
    /// or shared memory limit exceeded).
    Unfused,
}

/// A fused plan: stages, source buffers and shared-memory.
///
/// Stages are stored in kernel execution order. There are two phases:
///
/// 1. All stages except the last run first and decode their output
///    into shared memory (e.g. all dict values, all run-end endpoints).
///    This data stays resident for the output stage to index into.
///
/// 2. The last stage (the output stage) iterates over the input in tiles
///    of `SMEM_TILE_SIZE` (1024) elements, decoding each tile into a
///    scratch region of shared memory, applying scalar ops (which may
///    reference data from the earlier stages), and writing the result to
///    global memory.
///
/// # Per-stage PType tracking
///
/// Each stage carries a `source_ptype` (`PTypeTag`) that identifies the
/// primitive type produced by its source op (LOAD, BITUNPACK, etc.).
/// Scalar ops may change the type (e.g. DICT transforms codes → values,
/// ALP transforms encoded ints → floats); each `ScalarOp` declares its
/// `output_ptype`. The kernel uses these tags to dispatch typed memory
/// operations and cross-stage references at the correct element width.
///
/// # Shared memory allocation
///
/// Total shared memory = `smem_byte_cursor` + `SMEM_TILE_SIZE` × `output_elem_bytes`.
///
/// `smem_byte_cursor` is tracked in bytes and covers the preceding
/// fully-decoded stages (dict values, run-end endpoints). Each stage's
/// shared memory footprint is `len × final_ptype_byte_width`, where the
/// final ptype is determined by the last scalar op's `output_ptype` (or
/// `source_ptype` if there are no scalar ops).
///
/// All shared memory offsets are byte offsets — the C ABI uses byte
/// offsets and per-field `PTypeTag` values so that stages with different
/// element widths can coexist in the same shared memory pool.
///
/// This is sufficient because:
///
/// - Earlier stages only originate from dict (values) and run-end (ends,
///   values). `push_smem_stage` reserves the appropriate number of bytes
///   in `smem_byte_cursor`, so each stage's source op has room to decode
///   the complete input.
///
/// - The output stage (last) tiles at `SMEM_TILE_SIZE` (1024 elements),
///   so its source op never writes more than 1024 elements into the
///   scratch region, even though each block is responsible for
///   `ELEMENTS_PER_BLOCK` (2048) output elements — it processes them in
///   two passes through the scratch.
///
/// Note: `BITUNPACK` writes full FastLanes blocks (1024 elements), which can
/// exceed `stage.len` by up to 1023 elements. This overflow is absorbed by
/// the scratch region (`SMEM_TILE_SIZE` ≥ `FL_CHUNK_SIZE`).
pub struct FusedPlan {
    /// Stages in kernel execution order; all but the last decode into
    /// shared memory, the last decodes into global memory.
    stages: Vec<(Stage, SmemByteOffset, OutputLen)>,
    /// Shared memory reserved by the non-output stages, in bytes.
    smem_byte_cursor: SmemByteOffset,
    /// Source buffers. `None` entries are placeholder slots for pending subtrees,
    /// filled by [`materialize_with_subtrees`] before device copy.
    source_buffers: Vec<Option<BufferHandle>>,
    /// Bytes per element of the root (output) array.
    output_elem_bytes: u32,
    /// PType of the root (output) array, as a C ABI tag.
    output_ptype: PTypeTag,
    /// Validity of the root array, propagated to the output.
    validity: Validity,
}

impl DispatchPlan {
    /// Construct a plan by inspecting the encoding tree in a single pass.
    ///
    /// # Limitations
    ///
    /// - Validity is propagated from the root array to the output. Nullable
    ///   arrays are supported, but Dict with nullable codes and RunEnd with
    ///   nullable ends are rejected to guard against out-of-bounds access.
    /// - `BitPackedArray` and `ALPArray` with patches are not supported.
    /// - Only f32 ALP is supported (kernel stores multipliers as `float`).
    pub fn new(array: &ArrayRef) -> VortexResult<Self> {
        if PType::try_from(array.dtype()).is_err() || !is_dyn_dispatch_compatible(array) {
            return Ok(Self::Unfused);
        }

        let (plan, pending_subtrees) = match FusedPlan::build(array) {
            Ok(result) => result,
            Err(_) => return Ok(Self::Unfused),
        };

        if plan.exceeds_shared_mem_limit() {
            return Ok(Self::Unfused);
        }

        if pending_subtrees.is_empty() {
            Ok(Self::Fused(plan))
        } else {
            Ok(Self::PartiallyFused {
                plan,
                pending_subtrees,
            })
        }
    }
}

impl FusedPlan {
    /// Maximum shared memory per block in bytes (48 KB, static + dynamic).
    ///
    /// 48 KB is the default per-block shared memory limit across all CUDA
    /// architectures. Higher limits (up to 227 KB on Hopper) require an
    /// explicit opt-in via `cuFuncSetAttribute`.
    const MAX_SHARED_MEM_BYTES: u32 = 48 * 1024;

    /// Fixed shared memory used by the kernel (bytes).
    /// Sourced from the C header via bindgen.
    const FIXED_SHARED_MEM_BYTES: u32 = super::KERNEL_FIXED_SHARED_BYTES;

    /// Build a plan by walking the encoding tree from root to leaf.
    ///
    /// During the walk, incompatible nodes are discovered and recorded in the
    /// returned `Vec<ArrayRef>`.
    fn build(array: &ArrayRef) -> VortexResult<(Self, Vec<ArrayRef>)> {
        let output_ptype_rust = PType::try_from(array.dtype()).map_err(|_| {
            vortex_err!(
                "dyn dispatch requires primitive dtype, got {:?}",
                array.dtype()
            )
        })?;
        if output_ptype_rust == PType::F64 {
            vortex_bail!("dynamic dispatch does not support f64 output");
        }
        let output_elem_bytes = output_ptype_rust.byte_width() as u32;
        let output_ptype = ptype_to_tag(output_ptype_rust);
        let validity = array.validity()?;

        let mut pending_subtrees: Vec<ArrayRef> = Vec::new();
        let mut plan = Self {
            stages: Vec::new(),
            smem_byte_cursor: 0u32,
            source_buffers: Vec::new(),
            output_elem_bytes,
            output_ptype,
            validity,
        };

        let len = array.len() as u32;
        let output = plan.walk(array.clone(), &mut pending_subtrees)?;
        plan.stages.push((output, plan.smem_byte_cursor, len));

        Ok((plan, pending_subtrees))
    }

    /// Dynamic shared memory bytes passed to the CUDA launch config.
    fn dynamic_shared_mem_bytes(&self) -> u32 {
        self.smem_byte_cursor + SMEM_TILE_SIZE * self.output_elem_bytes
    }

    /// Total shared memory (fixed + dynamic) for limit checking.
    fn total_shared_mem_bytes(&self) -> u32 {
        Self::FIXED_SHARED_MEM_BYTES + self.dynamic_shared_mem_bytes()
    }

    /// Returns `true` if this plan's shared memory requirement exceeds
    /// the per-block limit, logging a trace message when it does.
    fn exceeds_shared_mem_limit(&self) -> bool {
        let required = self.total_shared_mem_bytes();
        if required > Self::MAX_SHARED_MEM_BYTES {
            trace!(
                required,
                limit = Self::MAX_SHARED_MEM_BYTES,
                "shared memory limit exceeded, falling back to unfused dispatch"
            );
            return true;
        }
        false
    }

    /// Copy source buffers to the device, producing a [`MaterializedPlan`].
    pub fn materialize(self, ctx: &CudaExecutionCtx) -> VortexResult<MaterializedPlan> {
        let shared_mem_bytes = self.dynamic_shared_mem_bytes();

        let mut device_buffers = Vec::new();
        let mut device_ptrs: Vec<u64> = Vec::new();

        // Copy each source buffer to the device and record its pointer.
        for source_buf in self.source_buffers {
            let source_buf = source_buf.ok_or_else(|| {
                vortex_err!("all source buffer slots must be filled before materialize")
            })?;
            let device_buf = ctx.ensure_on_device_sync(source_buf)?;
            let ptr = device_buf.cuda_device_ptr()?;
            device_ptrs.push(ptr);
            device_buffers.push(device_buf);
        }

        let resolve_ptr = |stage: &Stage| -> u64 {
            match stage.source_buffer_index {
                Some(idx) => device_ptrs[idx],
                None => 0,
            }
        };

        // Byte offsets are passed directly to the C ABI — the kernel now
        // indexes shared memory by byte offset and casts to the correct type
        // using source_ptype / output_ptype.
        let stages: Vec<MaterializedStage> = self
            .stages
            .iter()
            .map(|(stage, smem_byte_offset, len)| {
                MaterializedStage::new(
                    resolve_ptr(stage),
                    *smem_byte_offset,
                    *len,
                    stage.source_ptype,
                    stage.source,
                    &stage.scalar_ops,
                )
            })
            .collect();

        Ok(MaterializedPlan {
            dispatch_plan: CudaDispatchPlan::new(stages, self.output_ptype),
            device_buffers,
            shared_mem_bytes,
            validity: self.validity,
        })
    }

    /// Inject pre-executed subtree buffers into the placeholder (`None`) slots,
    /// then materialize.
    ///
    /// `subtree_buffers` must correspond 1:1 (in DFS order) to the
    /// `pending_subtrees` returned by `build`.
    pub fn materialize_with_subtrees(
        mut self,
        subtree_buffers: Vec<BufferHandle>,
        ctx: &CudaExecutionCtx,
    ) -> VortexResult<MaterializedPlan> {
        for (slot, buf) in zip_eq(
            self.source_buffers.iter_mut().filter(|s| s.is_none()),
            subtree_buffers,
        ) {
            *slot = Some(buf);
        }
        self.materialize(ctx)
    }

    /// Walk the encoding tree, producing a [`Stage`] for the root.
    fn walk(
        &mut self,
        array: ArrayRef,
        pending_subtrees: &mut Vec<ArrayRef>,
    ) -> VortexResult<Stage> {
        if !is_dyn_dispatch_compatible(&array) {
            // Subtree can't be fused — record it as a deferred LOAD source.
            // Bail if dtype is non-primitive (can't become a LOAD stage).
            let ptype = PType::try_from(array.dtype()).map_err(|_| {
                vortex_err!(
                    "unfusable subtree has non-primitive dtype {:?}, cannot partially fuse",
                    array.dtype()
                )
            })?;
            let buf_idx = self.source_buffers.len();
            self.source_buffers.push(None); // placeholder, filled at materialize time
            pending_subtrees.push(array);
            return Ok(Stage::new(
                SourceOp::load(),
                Some(buf_idx),
                ptype_to_tag(ptype),
            ));
        }

        let id = array.encoding_id();

        if id == BitPacked::ID {
            self.walk_bitpacked(array)
        } else if id == FoR::ID {
            self.walk_for(array, pending_subtrees)
        } else if id == ZigZag::ID {
            self.walk_zigzag(array, pending_subtrees)
        } else if id == ALP::ID {
            self.walk_alp(array, pending_subtrees)
        } else if id == Dict::ID {
            self.walk_dict(array, pending_subtrees)
        } else if id == RunEnd::ID {
            self.walk_runend(array, pending_subtrees)
        } else if id == Primitive::ID {
            self.walk_primitive(array)
        } else if id == Slice::ID {
            self.walk_slice(array, pending_subtrees)
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
    fn walk_slice(
        &mut self,
        array: ArrayRef,
        pending_subtrees: &mut Vec<ArrayRef>,
    ) -> VortexResult<Stage> {
        let slice_arr = array.as_::<Slice>();
        let child = slice_arr.child().clone();

        if let Some(reduced) = child.reduce_parent(&array, 0)? {
            return self.walk(reduced, pending_subtrees);
        }

        vortex_bail!(
            "Cannot resolve SliceArray wrapping {:?} in dynamic dispatch plan builder",
            child.encoding_id()
        )
    }

    fn walk_primitive(&mut self, array: ArrayRef) -> VortexResult<Stage> {
        let prim = array.as_::<Primitive>();
        let buf_index = self.source_buffers.len();
        self.source_buffers.push(Some(prim.buffer_handle().clone()));
        Ok(Stage::new(
            SourceOp::load(),
            Some(buf_index),
            ptype_to_tag(prim.ptype()),
        ))
    }

    fn walk_bitpacked(&mut self, array: ArrayRef) -> VortexResult<Stage> {
        let bp = array.as_::<BitPacked>();

        if bp.patches().is_some() {
            vortex_bail!("Dynamic dispatch does not support BitPackedArray with patches");
        }

        let source_ptype = ptype_to_tag(PType::try_from(bp.dtype()).map_err(|_| {
            vortex_err!("BitPacked must have primitive dtype, got {:?}", bp.dtype())
        })?);
        let buf_index = self.source_buffers.len();
        self.source_buffers.push(Some(bp.packed().clone()));
        Ok(Stage::new(
            SourceOp::bitunpack(bp.bit_width(), bp.offset()),
            Some(buf_index),
            source_ptype,
        ))
    }

    fn walk_for(
        &mut self,
        array: ArrayRef,
        pending_subtrees: &mut Vec<ArrayRef>,
    ) -> VortexResult<Stage> {
        let for_arr = array.as_::<FoR>();
        let ref_pvalue = for_arr
            .reference_scalar()
            .as_primitive()
            .pvalue()
            .ok_or_else(|| vortex_err!("FoR reference scalar is null"))?;
        let encoded = for_arr.encoded().clone();
        let output_ptype =
            ptype_to_tag(PType::try_from(array.dtype()).map_err(|_| {
                vortex_err!("FoR must have primitive dtype, got {:?}", array.dtype())
            })?);

        let mut pipeline = self.walk(encoded, pending_subtrees)?;
        let ref_u64 = ref_pvalue
            .reinterpret_cast(ref_pvalue.ptype().to_unsigned())
            .cast::<u64>()?;
        pipeline
            .scalar_ops
            .push(ScalarOp::frame_of_ref(ref_u64, output_ptype));
        Ok(pipeline)
    }

    fn walk_zigzag(
        &mut self,
        array: ArrayRef,
        pending_subtrees: &mut Vec<ArrayRef>,
    ) -> VortexResult<Stage> {
        let zz = array.as_::<ZigZag>();
        let encoded = zz.encoded().clone();
        let output_ptype = ptype_to_tag(PType::try_from(array.dtype()).map_err(|_| {
            vortex_err!("ZigZag must have primitive dtype, got {:?}", array.dtype())
        })?);

        let mut pipeline = self.walk(encoded, pending_subtrees)?;
        pipeline.scalar_ops.push(ScalarOp::zigzag(output_ptype));
        Ok(pipeline)
    }

    fn walk_alp(
        &mut self,
        array: ArrayRef,
        pending_subtrees: &mut Vec<ArrayRef>,
    ) -> VortexResult<Stage> {
        let alp = array.as_::<ALP>();

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

        let mut pipeline = self.walk(encoded, pending_subtrees)?;
        pipeline.scalar_ops.push(ScalarOp::alp(alp_f, alp_e));
        Ok(pipeline)
    }

    /// Handle a child array whose element width differs from the output type.
    ///
    /// If the child is a `Primitive`, its buffer is grabbed directly as a LOAD
    /// source — no separate kernel launch needed, since `load_element<T>()`
    /// handles the widening in-kernel. Otherwise, the child is recorded as a
    /// pending subtree for separate execution.
    fn walk_mixed_width_child(
        &mut self,
        child: ArrayRef,
        pending_subtrees: &mut Vec<ArrayRef>,
    ) -> VortexResult<Stage> {
        let ptype = PType::try_from(child.dtype())?;
        if child.encoding_id() == Primitive::ID {
            return self.walk_primitive(child);
        }
        let buf_idx = self.source_buffers.len();
        self.source_buffers.push(None);
        pending_subtrees.push(child);
        Ok(Stage::new(
            SourceOp::load(),
            Some(buf_idx),
            ptype_to_tag(ptype),
        ))
    }

    fn walk_dict(
        &mut self,
        array: ArrayRef,
        pending_subtrees: &mut Vec<ArrayRef>,
    ) -> VortexResult<Stage> {
        let dict = array.as_::<Dict>();
        let values = dict.values().clone();
        let codes = dict.codes().clone();

        let values_ptype = PType::try_from(values.dtype())?;
        let values_elem_bytes = values_ptype.byte_width() as u32;
        let codes_ptype = PType::try_from(codes.dtype())?;
        let codes_elem_bytes = codes_ptype.byte_width() as u32;

        // If values have a different width than the output type, they
        // can't be fused into the same kernel instantiation. Primitives
        // are handled directly (just grab the buffer); other encodings
        // become pending subtrees executed by a separate kernel.
        let values_len = values.len() as u32;
        let values_spec = if values_elem_bytes != self.output_elem_bytes {
            self.walk_mixed_width_child(values, pending_subtrees)?
        } else {
            self.walk(values, pending_subtrees)?
        };
        let values_smem_byte_offset = self.push_smem_stage(values_spec, values_len);

        // Same for codes.
        let mut pipeline = if codes_elem_bytes != self.output_elem_bytes {
            self.walk_mixed_width_child(codes, pending_subtrees)?
        } else {
            self.walk(codes, pending_subtrees)?
        };
        // DICT scalar op: pass byte offset directly (C ABI uses byte offsets).
        // output_ptype is the values' ptype — DICT transforms codes → values.
        pipeline.scalar_ops.push(ScalarOp::dict(
            values_smem_byte_offset,
            ptype_to_tag(values_ptype),
        ));
        Ok(pipeline)
    }

    fn walk_sequence(&mut self, array: ArrayRef) -> VortexResult<Stage> {
        let seq = array.as_::<Sequence>();

        Ok(Stage::new(
            SourceOp::sequence(seq.base().cast()?, seq.multiplier().cast()?),
            None,
            self.output_ptype,
        ))
    }

    fn walk_runend(
        &mut self,
        array: ArrayRef,
        pending_subtrees: &mut Vec<ArrayRef>,
    ) -> VortexResult<Stage> {
        let re = array.as_::<RunEnd>();
        let offset = re.offset() as u64;
        let ends = re.ends().clone();
        let values = re.values().clone();
        let num_runs = ends.len() as u32;
        let num_values = values.len() as u32;

        let ends_ptype = PType::try_from(ends.dtype())?;
        let ends_elem_bytes = ends_ptype.byte_width() as u32;
        let values_ptype = PType::try_from(values.dtype())?;
        let values_elem_bytes = values_ptype.byte_width() as u32;

        // If ends or values have a different width than the output type,
        // they can't be fused into the same kernel instantiation.
        // Primitives are handled directly; others become pending subtrees.
        let ends_spec = if ends_elem_bytes != self.output_elem_bytes {
            self.walk_mixed_width_child(ends, pending_subtrees)?
        } else {
            self.walk(ends, pending_subtrees)?
        };
        let ends_smem_byte_offset = self.push_smem_stage(ends_spec, num_runs);

        let values_spec = if values_elem_bytes != self.output_elem_bytes {
            self.walk_mixed_width_child(values, pending_subtrees)?
        } else {
            self.walk(values, pending_subtrees)?
        };
        let values_smem_byte_offset = self.push_smem_stage(values_spec, num_values);

        // Pass byte offsets and PTypeTags directly — the C ABI now uses
        // byte offsets and per-field ptype tags for cross-stage references.
        Ok(Stage::new(
            SourceOp::runend(
                ends_smem_byte_offset,
                values_smem_byte_offset,
                num_runs as u64,
                offset,
            ),
            None,
            self.output_ptype,
        ))
    }

    /// Add a stage that decodes fully into shared memory before the output
    /// stage runs. Returns the shared memory byte offset where the data starts.
    ///
    /// The smem region is sized at the stage's output ptype width — i.e.
    /// the ptype after all scalar ops have run. For stages that go through
    /// type-changing scalar ops (e.g. dict values with FoR→ALP), the final
    /// smem footprint is `len × final_ptype_byte_width`. If there are no
    /// scalar ops, the source_ptype determines the width.
    fn push_smem_stage(&mut self, spec: Stage, len: u32) -> u32 {
        let smem_byte_offset = self.smem_byte_cursor;
        // The kernel's execute_input_stage<T> always writes T-wide elements
        // into smem (reinterpret_cast<T*>), so we must allocate at least
        // output_elem_bytes per element — even if the stage's final ptype
        // is narrower. Otherwise the writes overflow into the next region.
        let final_ptype = spec
            .scalar_ops
            .last()
            .map(|op| op.output_ptype)
            .unwrap_or(spec.source_ptype);
        let final_elem_bytes = tag_to_ptype(final_ptype).byte_width() as u32;
        let elem_bytes = final_elem_bytes.max(self.output_elem_bytes);
        let stage_bytes = len * elem_bytes;
        self.stages.push((spec, smem_byte_offset, len));
        self.smem_byte_cursor += stage_bytes;
        smem_byte_offset
    }
}
