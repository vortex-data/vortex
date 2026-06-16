// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Walks an encoding tree and produces a [`DispatchPlan`] for a single GPU
//! kernel launch. The tree is inspected in a single pass, identifying unfusable
//! subtrees and computing shared memory requirements upfront — before any
//! device allocation or kernel work.

use std::ops::Range;

use itertools::zip_eq;
use tracing::trace;
use vortex::array::ArrayRef;
use vortex::array::ArrayVTable;
use vortex::array::arrays::Dict;
use vortex::array::arrays::Primitive;
use vortex::array::arrays::ScalarFn;
use vortex::array::arrays::Slice;
use vortex::array::arrays::dict::DictArraySlotsExt;
use vortex::array::arrays::scalar_fn::ScalarFnArrayExt;
use vortex::array::arrays::slice::SliceArrayExt;
use vortex::array::buffer::BufferHandle;
use vortex::array::patches::Patches;
use vortex::array::validity::Validity;
use vortex::dtype::PType;
use vortex::encodings::alp::ALP;
use vortex::encodings::alp::ALPArrayExt;
use vortex::encodings::alp::ALPArraySlotsExt;
use vortex::encodings::alp::ALPFloat;
use vortex::encodings::alp::Exponents;
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
use vortex::scalar_fn::ScalarFnVTable;
use vortex::scalar_fn::fns::cast::Cast;

use super::CudaDispatchPlan;
use super::MaterializedStage;
use super::PTypeTag;
use super::PTypeTag_PTYPE_F32;
use super::PTypeTag_PTYPE_F64;
use super::SMEM_TILE_SIZE;
use super::ScalarOp;
use super::SourceOp;
use super::ptype_to_tag;
use super::tag_to_ptype;
use crate::CudaBufferExt;
use crate::CudaExecutionCtx;
use crate::executor::CudaDispatchMode;
use crate::kernel::bitpacked_slice_view;
use crate::kernel::load_patches_to_gpu;

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
    // F16 has no reinterpret path in the kernel.
    if matches!(PType::try_from(array.dtype()), Ok(PType::F16)) {
        return false;
    }

    let id = array.encoding_id();
    if id == Cast.id() {
        return is_dyn_dispatch_cast_compatible(array);
    }
    if id == ALP.id() {
        let arr = array.as_::<ALP>();
        return matches!(arr.dtype().as_ptype(), PType::F32 | PType::F64);
    }
    if id == BitPacked.id() {
        return true;
    }
    if id == Dict.id() {
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
    if id == RunEnd.id() {
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
    id == FoR.id()
        || id == ZigZag.id()
        || id == Primitive.id()
        || id == Slice.id()
        || id == Sequence.id()
}

fn is_dyn_dispatch_cast_compatible(array: &ArrayRef) -> bool {
    let cast = array.as_::<ScalarFn>();

    let Ok(source_ptype) = PType::try_from(cast.child_at(0).dtype()) else {
        return false;
    };
    let Ok(target_ptype) = PType::try_from(cast.scalar_fn().as_::<Cast>()) else {
        return false;
    };

    // Implemented as unsigned dictionary-code casts to cuDF's signed index types.
    // LOAD/BITUNPACK materialize directly into the target-width output type.
    matches!(
        (source_ptype, target_ptype),
        (PType::U8, PType::I16) | (PType::U16, PType::I32) | (PType::U32, PType::I64)
    )
}

/// Returns `true` if a registered standalone kernel can decode the entire
/// `array` tree in a single launch without recursing into `execute_cuda`
/// for child encodings.
pub fn has_standalone_kernel(array: &ArrayRef) -> bool {
    let id = array.encoding_id();

    // Leaf encodings: no children to recurse into.
    if id == BitPacked.id() || id == Sequence.id() {
        return true;
    }

    // FoR fuses with BitPacked (FFOR) and Slice(BitPacked) in one launch.
    if id == FoR.id() {
        let for_arr = array.as_::<FoR>();
        let child = for_arr.encoded();
        if child.encoding_id() == BitPacked.id() {
            return true;
        }
        if let Some(slice) = child.as_opt::<Slice>() {
            return slice.child().encoding_id() == BitPacked.id();
        }
        return false;
    }

    false
}

/// Patch payload attached to the op that consumes it.
///
/// `range` is the logical output range to apply when materializing the patch descriptor on the GPU.
/// This lets the planner avoid calling `Patches::slice` when patch metadata may already be
/// device-resident.
#[derive(Clone)]
struct PlanPatches {
    patches: Patches,
    range: Option<Range<usize>>,
}

/// An unmaterialized stage: a source op, scalar ops, and optional source buffer reference.
///
/// Patch descriptors are tied to the op that consumes them, matching the CUDA parameter layout:
/// source patches live on `BitunpackParams`, while scalar-op patches live on `AlpParams`.
/// Patches may also carry a logical slice range when planning has sliced the values but patch
/// metadata must remain device-resident until materialization.
struct Stage {
    source: SourceOp,
    /// Patches from the source op (e.g. BitPacked overflow exceptions).
    source_patches: Option<PlanPatches>,
    /// Scalar ops with optional per-op patches (e.g. ALP float exceptions).
    scalar_ops: Vec<(ScalarOp, Option<PlanPatches>)>,
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
            source_patches: None,
            scalar_ops: vec![],
            source_buffer_index,
            source_ptype,
        }
    }

    fn with_source_patches(mut self, source_patches: Option<PlanPatches>) -> Self {
        self.source_patches = source_patches;
        self
    }
}

type SmemByteOffset = u32;
type OutputLen = u32;

/// A dispatch plan before device materialization.
///
/// Constructed by [`DispatchPlan::new`], which inspects the encoding tree
/// and determines whether it can be fully fused, partially fused, or not fused at all.
pub enum DispatchPlan {
    /// A registered standalone kernel can decode the entire tree in a single
    /// launch without recursing into child encodings.
    Standalone,
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
/// 1. All stages except the last decode into shared memory (dict values,
///    run-end endpoints). The kernel writes `T`-wide elements even when
///    a stage's source ptype is narrower, widening in-place as needed.
///
/// 2. The last stage (the output stage) tiles at `SMEM_TILE_SIZE` (1024)
///    elements, decoding each tile into a scratch region, applying scalar
///    ops (which may reference earlier stages), and streaming to global
///    memory.
///
/// # Shared memory allocation
///
/// Total = `smem_byte_cursor` + `SMEM_TILE_SIZE × output_elem_bytes`.
///
/// Each input stage occupies `len × max(final_width, output_elem_bytes)`
/// bytes, where `final_width` is the byte width of the last scalar op's
/// `output_ptype` (or `source_ptype` if none). The `max` is necessary
/// because `execute_input_stage<T>` writes `T`-wide elements even when
/// the stage's logical type is narrower.
///
/// `BITUNPACK` writes full FastLanes blocks (1024 elements) which may
/// exceed `stage.len` by up to 1023 elements; this overflow is absorbed
/// by the scratch region (`SMEM_TILE_SIZE` ≥ `FL_CHUNK_SIZE`).
pub struct FusedPlan {
    /// Stages in kernel execution order; all but the last decode into
    /// shared memory, the last decodes into global memory.
    stages: Vec<(Stage, SmemByteOffset, OutputLen)>,
    /// Shared memory reserved by the non-output stages, in bytes.
    smem_byte_cursor: SmemByteOffset,
    /// Source buffers. `None` entries are placeholder slots for pending subtrees,
    /// filled by [`Self::materialize_with_subtrees`] before device copy.
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
    /// - **F16 primitives** are not supported (no reinterpret path in the kernel).
    /// - **ALP** is supported for f32 and f64 only (including patches).
    /// - **BitPacked** with patches is supported.
    /// - **Dict** with nullable codes is rejected (garbage at null positions
    ///   could OOB the DICT gather). Dict with codes wider than values is
    ///   also rejected (load would truncate code indices).
    /// - **RunEnd** with nullable ends is rejected (garbage values break the
    ///   binary search). RunEnd with ends wider than values is also rejected.
    /// - Validity is propagated from the root array to the output.
    /// - Unrecognized encodings fall back to `Unfused`.
    pub fn new(array: &ArrayRef, mode: CudaDispatchMode) -> VortexResult<Self> {
        if mode == CudaDispatchMode::Auto && has_standalone_kernel(array) {
            return Ok(Self::Standalone);
        }

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
    pub async fn materialize(self, ctx: &mut CudaExecutionCtx) -> VortexResult<MaterializedPlan> {
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
        let mut stages: Vec<MaterializedStage> = Vec::new();
        for (stage, smem_byte_offset, len) in &self.stages {
            let mut source = stage.source;

            // Upload source patches (e.g. BitPacked exceptions).
            if let Some(patches) = &stage.source_patches {
                let (ptr, bufs) =
                    load_patches_to_gpu(&patches.patches, patches.range.clone(), ctx).await?;
                source.params.bitunpack.patches_ptr = ptr;
                device_buffers.extend(bufs);
            }

            // Upload patches for each scalar op that carries them.
            let mut scalar_ops: Vec<ScalarOp> = Vec::with_capacity(stage.scalar_ops.len());
            for (mut op, patches) in stage.scalar_ops.clone() {
                if let Some(patches) = &patches {
                    let (ptr, bufs) =
                        load_patches_to_gpu(&patches.patches, patches.range.clone(), ctx).await?;
                    op.params.alp.patches_ptr = ptr;
                    device_buffers.extend(bufs);
                }
                scalar_ops.push(op);
            }

            stages.push(MaterializedStage::new(
                resolve_ptr(stage),
                *smem_byte_offset,
                *len,
                stage.source_ptype,
                source,
                &scalar_ops,
            ));
        }

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
    pub async fn materialize_with_subtrees(
        mut self,
        subtree_buffers: Vec<BufferHandle>,
        ctx: &mut CudaExecutionCtx,
    ) -> VortexResult<MaterializedPlan> {
        for (slot, buf) in zip_eq(
            self.source_buffers.iter_mut().filter(|s| s.is_none()),
            subtree_buffers,
        ) {
            *slot = Some(buf);
        }
        self.materialize(ctx).await
    }

    /// Walk the encoding tree, producing a [`Stage`] for the root.
    fn walk(
        &mut self,
        array: ArrayRef,
        pending_subtrees: &mut Vec<ArrayRef>,
    ) -> VortexResult<Stage> {
        if !is_dyn_dispatch_compatible(&array) {
            return self.push_subtree(array, pending_subtrees);
        }

        let id = array.encoding_id();

        if id == BitPacked.id() {
            self.walk_bitpacked(array)
        } else if id == FoR.id() {
            self.walk_for(array, pending_subtrees)
        } else if id == ZigZag.id() {
            self.walk_zigzag(array, pending_subtrees)
        } else if id == ALP.id() {
            self.walk_alp(array, pending_subtrees)
        } else if id == Dict.id() {
            self.walk_dict(array, pending_subtrees)
        } else if id == RunEnd.id() {
            self.walk_runend(array, pending_subtrees)
        } else if id == Primitive.id() {
            self.walk_primitive(array)
        } else if id == Slice.id() {
            self.walk_slice(array, pending_subtrees)
        } else if id == Sequence.id() {
            self.walk_sequence(array)
        } else if id == Cast.id() {
            self.walk_cast(array, pending_subtrees)
        } else {
            vortex_bail!(
                "Encoding {:?} not supported by dynamic dispatch plan builder",
                id
            )
        }
    }

    /// SliceArray → resolve the slice via reduce/execute rules.
    ///
    /// When the plan builder encounters a `SliceArray`, it first asks the child to reduce the
    /// slice. If reduction fails, the planner falls back to encoding-specific handling.
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

        // BitPacked with patches does not reduce through Slice. Slice the
        // packed buffer here, and defer patch slicing to CUDA materialization.
        if child.encoding_id() == BitPacked.id() {
            let bp = child.as_::<BitPacked>();
            let offset = slice_arr.data().slice_range().start;
            let len = array.len();
            let (packed, bitpacked_offset, patch_range) = bitpacked_slice_view(bp, offset, len)?;

            let source_ptype = ptype_to_tag(PType::try_from(bp.dtype()).map_err(|_| {
                vortex_err!("BitPacked must have primitive dtype, got {:?}", bp.dtype())
            })?);
            let buf_index = self.source_buffers.len();
            self.source_buffers.push(Some(packed));
            return Ok(Stage::new(
                SourceOp::bitunpack(bp.bit_width(), bitpacked_offset),
                Some(buf_index),
                source_ptype,
            )
            .with_source_patches(bp.patches().map(|patches| PlanPatches {
                patches,
                range: Some(patch_range),
            })));
        }

        // ALP doesn't implement reduce_parent. Slice the encoded child here,
        // and defer patch slicing to CUDA materialization so device-resident
        // patch buffers stay on device.
        if child.encoding_id() == ALP.id() {
            let alp = child.as_::<ALP>();
            let offset = slice_arr.data().slice_range().start;
            let len = array.len();
            let sliced_encoded = alp.encoded().clone().slice(offset..offset + len)?;
            return self.walk_alp_inner(
                sliced_encoded,
                alp.patches().map(|patches| PlanPatches {
                    patches,
                    range: Some(offset..offset + len),
                }),
                alp.exponents(),
                pending_subtrees,
            );
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

        let source_ptype = ptype_to_tag(PType::try_from(bp.dtype()).map_err(|_| {
            vortex_err!("BitPacked must have primitive dtype, got {:?}", bp.dtype())
        })?);
        let buf_index = self.source_buffers.len();
        self.source_buffers.push(Some(bp.packed().clone()));
        Ok(Stage::new(
            SourceOp::bitunpack(bp.bit_width(), bp.offset()),
            Some(buf_index),
            source_ptype,
        )
        .with_source_patches(bp.patches().map(|patches| PlanPatches {
            patches,
            range: None,
        })))
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
            .push((ScalarOp::frame_of_ref(ref_u64, output_ptype), None));
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
        pipeline
            .scalar_ops
            .push((ScalarOp::zigzag(output_ptype), None));
        Ok(pipeline)
    }

    fn walk_alp(
        &mut self,
        array: ArrayRef,
        pending_subtrees: &mut Vec<ArrayRef>,
    ) -> VortexResult<Stage> {
        let alp = array.as_::<ALP>();
        self.walk_alp_inner(
            alp.encoded().clone(),
            alp.patches().map(|patches| PlanPatches {
                patches,
                range: None,
            }),
            alp.exponents(),
            pending_subtrees,
        )
    }

    /// Shared ALP logic for both `walk_alp` and `walk_slice` (Slice(ALP)).
    fn walk_alp_inner(
        &mut self,
        encoded: ArrayRef,
        patches: Option<PlanPatches>,
        exponents: Exponents,
        pending_subtrees: &mut Vec<ArrayRef>,
    ) -> VortexResult<Stage> {
        let encoded_ptype = PType::try_from(encoded.dtype()).map_err(|_| {
            vortex_err!(
                "ALP encoded child must have primitive dtype, got {:?}",
                encoded.dtype()
            )
        })?;
        // ALP encodes f32 as i32 and f64 as i64. Select the correct
        // exponent tables and output PType based on the encoded integer width.
        let (alp_f, alp_e, output_ptype) = match encoded_ptype {
            PType::I32 => (
                <f32 as ALPFloat>::F10[exponents.f as usize] as f64,
                <f32 as ALPFloat>::IF10[exponents.e as usize] as f64,
                PTypeTag_PTYPE_F32,
            ),
            PType::I64 => (
                <f64 as ALPFloat>::F10[exponents.f as usize],
                <f64 as ALPFloat>::IF10[exponents.e as usize],
                PTypeTag_PTYPE_F64,
            ),
            other => vortex_bail!(
                "ALP encoded ptype must be I32 (f32) or I64 (f64), got {:?}",
                other
            ),
        };

        let mut pipeline = self.walk(encoded, pending_subtrees)?;
        pipeline
            .scalar_ops
            .push((ScalarOp::alp(alp_f, alp_e, output_ptype), patches));
        Ok(pipeline)
    }

    /// Walk a child that may have a different element width than the output.
    ///
    /// Primitives are always handled directly (`load_element<T>()` widens
    /// in-kernel). Non-primitive children are recursively walked; the kernel's
    /// `bitunpack_typed` decodes at the source's native width and widens to
    /// `T` in shared memory, and `push_smem_stage` allocates accordingly.
    fn walk_child(
        &mut self,
        array: ArrayRef,
        pending_subtrees: &mut Vec<ArrayRef>,
    ) -> VortexResult<Stage> {
        if array.encoding_id() == Primitive.id() {
            return self.walk_primitive(array);
        }
        self.walk(array, pending_subtrees)
    }

    /// Reserve a placeholder buffer slot and record the array as a pending subtree.
    ///
    /// Called from [`Self::walk`] when [`is_dyn_dispatch_compatible`] rejects a child.
    /// Cases that require a separate kernel dispatch:
    ///
    /// - **F16 primitives** — no reinterpret path in the kernel.
    /// - **Dict with nullable codes** — garbage at null positions could OOB
    ///   the DICT gather in shared memory.
    /// - **Dict with codes wider than values** — `load_element<T>()` would
    ///   truncate the code indices.
    /// - **RunEnd with nullable ends** — garbage values break the binary
    ///   search / forward-scan.
    /// - **RunEnd with ends wider than values** — same truncation issue.
    /// - **Unrecognized encoding** — anything outside the kernel's allow-list
    ///   (e.g. FSST, Pco, Zstd).
    fn push_subtree(
        &mut self,
        array: ArrayRef,
        pending_subtrees: &mut Vec<ArrayRef>,
    ) -> VortexResult<Stage> {
        let ptype = PType::try_from(array.dtype()).map_err(|_| {
            vortex_err!(
                "unfusable subtree has non-primitive dtype {:?}, cannot partially fuse",
                array.dtype()
            )
        })?;
        let buf_idx = self.source_buffers.len();
        self.source_buffers.push(None);
        pending_subtrees.push(array);
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

        let values_len = values.len() as u32;
        let values_spec = self.walk_child(values, pending_subtrees)?;
        let values_smem_byte_offset = self.push_smem_stage(values_spec, values_len);

        let mut pipeline = self.walk_child(codes, pending_subtrees)?;
        // DICT scalar op: pass byte offset directly (C ABI uses byte offsets).
        // output_ptype is the values' ptype — DICT transforms codes → values.
        pipeline.scalar_ops.push((
            ScalarOp::dict(values_smem_byte_offset, ptype_to_tag(values_ptype)),
            None,
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

    fn walk_cast(
        &mut self,
        array: ArrayRef,
        pending_subtrees: &mut Vec<ArrayRef>,
    ) -> VortexResult<Stage> {
        let cast = array.as_::<ScalarFn>();
        let target_ptype = ptype_to_tag(cast.scalar_fn().as_::<Cast>().as_ptype());
        let mut pipeline = self.walk(cast.child_at(0).clone(), pending_subtrees)?;
        // LOAD/BITUNPACK directly widen into the output type without an additional cast op.
        pipeline
            .scalar_ops
            .push((ScalarOp::cast(target_ptype), None));
        Ok(pipeline)
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

        let ends_spec = self.walk_child(ends, pending_subtrees)?;
        let ends_smem_byte_offset = self.push_smem_stage(ends_spec, num_runs);

        let values_spec = self.walk_child(values, pending_subtrees)?;
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
    /// stage runs. Returns the shared memory byte offset where the data
    /// starts. Allocates `len × max(final_width, output_elem_bytes)` bytes
    /// so that narrower stages widened to `T` by `bitunpack_typed` never
    /// overflow.
    fn push_smem_stage(&mut self, spec: Stage, len: u32) -> u32 {
        let smem_byte_offset = self.smem_byte_cursor;
        // The kernel's execute_input_stage<T> always writes T-wide elements
        // into smem (reinterpret_cast<T*>), so we must allocate at least
        // output_elem_bytes per element — even if the stage's final ptype
        // is narrower. Otherwise the writes overflow into the next region.
        let stage_bytes = Self::smem_stage_bytes(&spec, len, self.output_elem_bytes);
        self.stages.push((spec, smem_byte_offset, len));
        self.smem_byte_cursor += stage_bytes;
        smem_byte_offset
    }

    fn smem_stage_bytes(spec: &Stage, len: u32, output_elem_bytes: u32) -> u32 {
        let final_ptype = spec
            .scalar_ops
            .last()
            .map(|(op, _)| op.output_ptype)
            .unwrap_or(spec.source_ptype);
        let final_elem_bytes = tag_to_ptype(final_ptype).byte_width() as u32;
        len * final_elem_bytes.max(output_elem_bytes)
    }
}

#[cfg(test)]
mod tests {
    use vortex::array::IntoArray;
    use vortex::array::arrays::PrimitiveArray;
    use vortex::array::builtins::ArrayBuiltins;
    use vortex::dtype::DType;
    use vortex::dtype::Nullability;

    use super::*;

    #[test]
    fn cast_to_non_primitive_target_is_not_dyn_dispatch_compatible() -> VortexResult<()> {
        let cast = PrimitiveArray::from_iter([0u8, 1])
            .into_array()
            .cast(DType::Bool(Nullability::NonNullable))?;

        assert!(!is_dyn_dispatch_cast_compatible(&cast));
        assert!(matches!(
            DispatchPlan::new(&cast, CudaDispatchMode::DynDispatchOnly)?,
            DispatchPlan::Unfused
        ));

        Ok(())
    }
}
