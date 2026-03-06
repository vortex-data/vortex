// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Builds a [`DynamicDispatchPlan`] directly from a flatbuffer encoding tree
//! and raw segment bytes, **without constructing any `ArrayRef`**.
//!
//! The standard path is:
//!
//! ```text
//! flatbuffer + segment → ArrayParts::decode() → ArrayRef tree → build_plan() → DynamicDispatchPlan
//! ```
//!
//! This module collapses that into:
//!
//! ```text
//! flatbuffer + segment → build_plan_from_flatbuffer() → DynamicDispatchPlan
//! ```
//!
//! The savings come from skipping:
//! - All `Arc` clone/drop atomics from `ArrayRef` construction (~12% of CPU time)
//! - `vtable.build()` recursive array construction
//! - `StatsSet::from_flatbuffer()` statistics deserialization
//! - Flatbuffer re-verification on every child access
//! - The second tree walk that `build_plan` performs
//!
//! ## Supported encodings
//!
//! The same set as [`super::build_plan`]:
//! `Primitive`, `BitPacked`, `FoR`, `ZigZag`, `ALP`, `Dict`, `RunEnd`,
//! `Slice`, `Chunked`, `Sequence`.

use std::sync::Arc;

use prost::Message;
use vortex::array::ArrayContext;
use vortex::array::buffer::BufferHandle;
use vortex::buffer::Alignment;
use vortex::buffer::ByteBuffer;
use vortex::dtype::DType;
use vortex::dtype::PType;
use vortex::encodings::alp::ALPFloat;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_err;
use vortex::flatbuffers::FlatBuffer;
use vortex::flatbuffers::array as fba;
use vortex::utils::aliases::hash_map::HashMap;

use super::DynamicDispatchPlan;
use super::MAX_STAGES;
use super::ScalarOp;
use super::SourceOp;
use super::Stage;
use crate::CudaBufferExt;
use crate::CudaExecutionCtx;

// ── Encoding IDs (matched by string against the ArrayContext) ───────────

const BITPACKED_ID: &str = "fastlanes.bitpacked";
const FOR_ID: &str = "fastlanes.for";
const ZIGZAG_ID: &str = "vortex.zigzag";
const ALP_ID: &str = "vortex.alp";
const DICT_ID: &str = "vortex.dict";
const RUNEND_ID: &str = "vortex.runend";
const PRIMITIVE_ID: &str = "vortex.primitive";
const SLICE_ID: &str = "vortex.slice";
const CHUNKED_ID: &str = "vortex.chunked";
const SEQUENCE_ID: &str = "vortex.sequence";

// ── Protobuf metadata structs (mirror the prost definitions in each encoding crate) ─

/// Mirror of `BitPackedMetadata` from `vortex-fastlanes`.
#[derive(prost::Message)]
struct BitPackedMetadata {
    #[prost(uint32, tag = "1")]
    bit_width: u32,
    #[prost(uint32, tag = "2")]
    offset: u32,
    // tag 3 = patches; we only need to know if it's present.
    #[prost(message, optional, tag = "3")]
    patches: Option<PatchesMetadata>,
}

#[derive(prost::Message)]
struct PatchesMetadata {
    #[prost(uint32, tag = "1")]
    _num_patches: u32,
}

/// Mirror of `ALPMetadata` from `vortex-alp`.
#[derive(prost::Message)]
struct ALPMetadata {
    #[prost(uint32, tag = "1")]
    exp_e: u32,
    #[prost(uint32, tag = "2")]
    exp_f: u32,
    #[prost(message, optional, tag = "3")]
    patches: Option<PatchesMetadata>,
}

/// Mirror of `RunEndMetadata` from `vortex-runend`.
#[derive(prost::Message)]
struct RunEndMetadata {
    #[prost(int32, tag = "1")]
    _ends_ptype: i32,
    #[prost(uint64, tag = "2")]
    num_runs: u64,
    #[prost(uint64, tag = "3")]
    offset: u64,
}

/// Mirror of `ProstSequenceMetadata` from `vortex-sequence`.
#[derive(prost::Message)]
struct SequenceMetadataProto {
    #[prost(message, tag = "1")]
    base: Option<vortex::proto::scalar::ScalarValue>,
    #[prost(message, tag = "2")]
    multiplier: Option<vortex::proto::scalar::ScalarValue>,
}

/// The result of walking a subtree: a source op, scalar ops, and a device pointer.
struct Pipeline {
    source: SourceOp,
    scalar_ops: Vec<ScalarOp>,
    input_ptr: u64,
}

/// Build a [`DynamicDispatchPlan`] directly from the flatbuffer encoding tree,
/// the raw segment `BufferHandle`, and (optionally) host-resident buffer overrides.
///
/// This is the zero-deser fast path: no `ArrayRef` is constructed, no stats
/// are deserialized, and the flatbuffer is walked exactly once.
///
/// # Arguments
///
/// * `array_tree` – the serialized `Array` flatbuffer (encoding tree + buffer descriptors).
/// * `segment` – the raw segment data containing all encoded buffers.
/// * `host_buffers` – buffers inlined in layout metadata (keyed by global buffer index).
///   These override the corresponding segment slices.
/// * `row_count` – total row count for the array (used as `array_len` if needed).
/// * `dtype` – the column's logical [`DType`], needed to interpret FoR references.
/// * `ctx` – the [`ArrayContext`] that maps encoding indices to encoding IDs.
/// * `cuda_ctx` – the CUDA execution context for device pointer resolution.
///
/// # Returns
///
/// `(plan, keep_alive)` — the dispatch plan and buffer handles that must remain
/// alive while the plan's device pointers are in use.
pub fn build_plan_from_flatbuffer(
    array_tree: &ByteBuffer,
    segment: &BufferHandle,
    host_buffers: &Arc<HashMap<u32, ByteBuffer>>,
    row_count: u64,
    dtype: &DType,
    ctx: &ArrayContext,
    cuda_ctx: &CudaExecutionCtx,
) -> VortexResult<(DynamicDispatchPlan, Vec<BufferHandle>)> {
    // Parse the root flatbuffer (Array) to get the ArrayNode tree and Buffer descriptors.
    let fb_buf = FlatBuffer::align_from(array_tree.clone());
    let fb_array = fba::root_as_array(fb_buf.as_ref())
        .map_err(|e| vortex_err!("Invalid array flatbuffer: {e}"))?;
    let fb_root = fb_array
        .root()
        .ok_or_else(|| vortex_err!("Array must have a root node"))?;

    // Pre-resolve all buffers from the segment (respecting padding/alignment).
    let resolved_buffers = resolve_all_buffers(&fb_array, segment, host_buffers)?;

    let mut state = FbPlanBuilderState {
        ctx,
        cuda_ctx,
        dtype,
        stages: Vec::new(),
        smem_cursor: 0,
        device_buffers: Vec::new(),
        slice_offset: 0,
        resolved_buffers: &resolved_buffers,
    };

    let pipeline = state.walk(fb_root, row_count)?;
    let output_stage = Stage::output(
        pipeline.input_ptr,
        state.smem_cursor,
        pipeline.source,
        &pipeline.scalar_ops,
    );
    state.stages.push(output_stage);

    assert!(state.stages.len() <= MAX_STAGES as usize);

    Ok((DynamicDispatchPlan::new(state.stages), state.device_buffers))
}

/// Pre-resolve all buffer descriptors from the flatbuffer into `BufferHandle`s.
///
/// Each buffer in the `Array.buffers` list is sliced from the segment (with
/// padding and alignment applied), unless a host override exists for that index.
fn resolve_all_buffers(
    fb_array: &fba::Array<'_>,
    segment: &BufferHandle,
    host_buffers: &HashMap<u32, ByteBuffer>,
) -> VortexResult<Vec<BufferHandle>> {
    let segment = segment.clone().ensure_aligned(Alignment::none())?;
    let fb_bufs = fb_array.buffers().unwrap_or_default();

    let mut offset = 0usize;
    let mut result = Vec::with_capacity(fb_bufs.len());

    for i in 0..fb_bufs.len() {
        let fb_buf = fb_bufs.get(i);
        offset += fb_buf.padding() as usize;
        let length = fb_buf.length() as usize;
        let alignment = Alignment::from_exponent(fb_buf.alignment_exponent());

        let idx = i as u32;
        let handle = if let Some(host_data) = host_buffers.get(&idx) {
            BufferHandle::new_host(host_data.clone()).ensure_aligned(alignment)?
        } else {
            segment
                .slice(offset..(offset + length))
                .ensure_aligned(alignment)?
        };

        offset += length;
        result.push(handle);
    }

    Ok(result)
}

/// Internal state for the flatbuffer tree walk.
struct FbPlanBuilderState<'a> {
    ctx: &'a ArrayContext,
    cuda_ctx: &'a CudaExecutionCtx,
    /// The column's logical DType, threaded through for FoR reference interpretation.
    dtype: &'a DType,
    stages: Vec<Stage>,
    smem_cursor: u32,
    device_buffers: Vec<BufferHandle>,
    slice_offset: u64,
    /// All buffers pre-resolved from the segment, indexed by the global buffer index.
    resolved_buffers: &'a [BufferHandle],
}

impl FbPlanBuilderState<'_> {
    /// Resolve the encoding ID string for an ArrayNode's encoding index.
    fn encoding_id(
        &self,
        node: &fba::ArrayNode<'_>,
    ) -> VortexResult<vortex::session::registry::Id> {
        let idx = node.encoding();
        self.ctx
            .resolve(idx)
            .ok_or_else(|| vortex_err!("Unknown encoding index: {}", idx))
    }

    /// Get the metadata bytes for a node.
    fn metadata<'a>(&self, node: &'a fba::ArrayNode<'a>) -> &'a [u8] {
        node.metadata().map(|m| m.bytes()).unwrap_or(&[])
    }

    /// Get the nth child of a node.
    fn child<'a>(
        &self,
        node: &'a fba::ArrayNode<'a>,
        idx: usize,
    ) -> VortexResult<fba::ArrayNode<'a>> {
        let children = node
            .children()
            .ok_or_else(|| vortex_err!("Node has no children"))?;
        if idx >= children.len() {
            vortex_bail!(
                "Child index {} out of bounds (have {})",
                idx,
                children.len()
            );
        }
        Ok(children.get(idx))
    }

    /// Get the buffer for the nth buffer reference of a node.
    fn node_buffer(&self, node: &fba::ArrayNode<'_>, idx: usize) -> VortexResult<BufferHandle> {
        let buf_indices = node
            .buffers()
            .ok_or_else(|| vortex_err!("Node has no buffers"))?;
        let global_idx = buf_indices.get(idx) as usize;
        self.resolved_buffers
            .get(global_idx)
            .cloned()
            .ok_or_else(|| {
                vortex_err!(
                    "Buffer index {} out of range (have {})",
                    global_idx,
                    self.resolved_buffers.len()
                )
            })
    }

    /// Recursively walk an ArrayNode encoding tree.
    fn walk(&mut self, node: fba::ArrayNode<'_>, row_count: u64) -> VortexResult<Pipeline> {
        let id = self.encoding_id(&node)?;

        match id.as_ref() {
            BITPACKED_ID => self.walk_bitpacked(node),
            FOR_ID => self.walk_for(node, row_count),
            ZIGZAG_ID => self.walk_zigzag(node, row_count),
            ALP_ID => self.walk_alp(node, row_count),
            DICT_ID => self.walk_dict(node, row_count),
            RUNEND_ID => self.walk_runend(node),
            PRIMITIVE_ID => self.walk_primitive(node),
            SLICE_ID => self.walk_slice(node, row_count),
            CHUNKED_ID => self.walk_chunked(node, row_count),
            SEQUENCE_ID => self.walk_sequence(node),
            other => vortex_bail!(
                "Encoding {:?} not supported by flatbuffer plan builder",
                other
            ),
        }
    }

    // ── Leaf encodings ───────────────────────────────────────────────

    /// Primitive → LOAD source op.
    fn walk_primitive(&mut self, node: fba::ArrayNode<'_>) -> VortexResult<Pipeline> {
        if !self.dtype.is_primitive() {
            vortex_bail!(
                "Expected primitive dtype for Primitive encoding, got {:?}",
                self.dtype
            );
        }
        let ptype = self.dtype.as_ptype();
        let byte_offset = self.slice_offset * ptype.byte_width() as u64;
        let buffer = self.node_buffer(&node, 0)?;
        let ptr = self.resolve_buffer_ptr(buffer)?;
        Ok(Pipeline {
            source: SourceOp::load(),
            scalar_ops: vec![],
            input_ptr: ptr + byte_offset,
        })
    }

    /// BitPacked → BITUNPACK source op.
    fn walk_bitpacked(&mut self, node: fba::ArrayNode<'_>) -> VortexResult<Pipeline> {
        let meta_bytes = self.metadata(&node);
        let meta = BitPackedMetadata::decode(meta_bytes)
            .map_err(|e| vortex_err!("Failed to decode BitPackedMetadata: {e}"))?;

        if meta.patches.is_some() {
            vortex_bail!("Dynamic dispatch does not support BitPackedArray with patches");
        }

        let total_offset = meta.offset as u64 + self.slice_offset;
        let buffer = self.node_buffer(&node, 0)?;
        let ptr = self.resolve_buffer_ptr(buffer)?;

        Ok(Pipeline {
            source: SourceOp::bitunpack(meta.bit_width as u8, total_offset),
            scalar_ops: vec![],
            input_ptr: ptr,
        })
    }

    /// Sequence → SEQUENCE source op (no input buffer).
    fn walk_sequence(&mut self, node: fba::ArrayNode<'_>) -> VortexResult<Pipeline> {
        let meta_bytes = self.metadata(&node);
        let meta = SequenceMetadataProto::decode(meta_bytes)
            .map_err(|e| vortex_err!("Failed to decode SequenceMetadata: {e}"))?;

        let base = scalar_value_to_u64(
            meta.base
                .as_ref()
                .ok_or_else(|| vortex_err!("Sequence missing base"))?,
        )?;
        let multiplier = scalar_value_to_u64(
            meta.multiplier
                .as_ref()
                .ok_or_else(|| vortex_err!("Sequence missing multiplier"))?,
        )?;

        Ok(Pipeline {
            source: SourceOp::sequence(base, multiplier),
            scalar_ops: vec![],
            input_ptr: 1, // non-null dummy — SEQUENCE doesn't dereference input
        })
    }

    // ── Single-child transform encodings ─────────────────────────────

    /// FoR → recurse into child, add FoR scalar op.
    fn walk_for(&mut self, node: fba::ArrayNode<'_>, row_count: u64) -> VortexResult<Pipeline> {
        let meta_bytes = self.metadata(&node);
        let ref_u64 = for_reference_from_proto(meta_bytes, self.dtype)?;

        let child = self.child(&node, 0)?;
        let mut pipeline = self.walk(child, row_count)?;
        pipeline.scalar_ops.push(ScalarOp::frame_of_ref(ref_u64));
        Ok(pipeline)
    }

    /// ZigZag → recurse into child, add ZigZag scalar op.
    fn walk_zigzag(&mut self, node: fba::ArrayNode<'_>, row_count: u64) -> VortexResult<Pipeline> {
        let child = self.child(&node, 0)?;
        let mut pipeline = self.walk(child, row_count)?;
        pipeline.scalar_ops.push(ScalarOp::zigzag());
        Ok(pipeline)
    }

    /// ALP → recurse into child, add ALP scalar op (f32 only).
    fn walk_alp(&mut self, node: fba::ArrayNode<'_>, row_count: u64) -> VortexResult<Pipeline> {
        let meta_bytes = self.metadata(&node);
        let meta = ALPMetadata::decode(meta_bytes)
            .map_err(|e| vortex_err!("Failed to decode ALPMetadata: {e}"))?;

        if meta.patches.is_some() {
            vortex_bail!("Dynamic dispatch does not support ALPArray with patches");
        }

        if !self.dtype.is_primitive() {
            vortex_bail!(
                "Expected primitive dtype for ALP encoding, got {:?}",
                self.dtype
            );
        }
        let ptype = self.dtype.as_ptype();
        if ptype != PType::F32 {
            vortex_bail!(
                "Dynamic dispatch only supports f32 ALP, got {:?}",
                self.dtype
            );
        }

        let alp_f = <f32 as ALPFloat>::F10[meta.exp_f as usize];
        let alp_e = <f32 as ALPFloat>::IF10[meta.exp_e as usize];

        let child = self.child(&node, 0)?;
        let mut pipeline = self.walk(child, row_count)?;
        pipeline.scalar_ops.push(ScalarOp::alp(alp_f, alp_e));
        Ok(pipeline)
    }

    // ── Multi-child encodings ────────────────────────────────────────

    /// Dict → input stage for values, recurse codes, add DICT scalar op.
    fn walk_dict(&mut self, node: fba::ArrayNode<'_>, row_count: u64) -> VortexResult<Pipeline> {
        // Child 0 = values, Child 1 = codes
        let values_node = self.child(&node, 0)?;
        let codes_node = self.child(&node, 1)?;

        let meta_bytes = self.metadata(&node);
        let meta = DictMetadataProto::decode(meta_bytes)
            .map_err(|e| vortex_err!("Failed to decode DictMetadata: {e}"))?;
        let values_len = meta.values_len as u64;

        let values_smem_offset = self.add_input_stage(values_node, values_len)?;

        let mut pipeline = self.walk(codes_node, row_count)?;
        pipeline.scalar_ops.push(ScalarOp::dict(values_smem_offset));
        Ok(pipeline)
    }

    /// RunEnd → input stages for ends and values, RUNEND source op.
    fn walk_runend(&mut self, node: fba::ArrayNode<'_>) -> VortexResult<Pipeline> {
        let meta_bytes = self.metadata(&node);
        let meta = RunEndMetadata::decode(meta_bytes)
            .map_err(|e| vortex_err!("Failed to decode RunEndMetadata: {e}"))?;

        // Child 0 = ends, Child 1 = values
        let ends_node = self.child(&node, 0)?;
        let values_node = self.child(&node, 1)?;

        let ends_smem = self.add_input_stage(ends_node, meta.num_runs)?;
        let values_smem = self.add_input_stage(values_node, meta.num_runs)?;

        Ok(Pipeline {
            source: SourceOp::runend(ends_smem, values_smem, meta.num_runs, meta.offset),
            scalar_ops: vec![],
            input_ptr: 0,
        })
    }

    // ── Wrapper encodings ────────────────────────────────────────────

    /// Slice → accumulate offset, recurse into child.
    ///
    /// Note: Slice arrays are not directly serialized in Vortex files, but
    /// the layout reader may produce them. We handle them for completeness.
    fn walk_slice(&mut self, node: fba::ArrayNode<'_>, row_count: u64) -> VortexResult<Pipeline> {
        // Slice metadata is a raw Range<usize> serialized as two u64 LE values.
        let meta_bytes = self.metadata(&node);
        if meta_bytes.len() < 16 {
            vortex_bail!("Slice metadata too short: {} bytes", meta_bytes.len());
        }
        let start = u64::from_le_bytes(
            meta_bytes[0..8]
                .try_into()
                .map_err(|_| vortex_err!("Slice metadata start bytes invalid"))?,
        );

        let prev_offset = self.slice_offset;
        self.slice_offset += start;
        let child = self.child(&node, 0)?;
        let result = self.walk(child, row_count);
        self.slice_offset = prev_offset;
        result
    }

    /// Chunked → walk the first chunk (same as build_plan).
    fn walk_chunked(&mut self, node: fba::ArrayNode<'_>, row_count: u64) -> VortexResult<Pipeline> {
        let children = node
            .children()
            .ok_or_else(|| vortex_err!("ChunkedArray has no children"))?;
        if children.is_empty() {
            vortex_bail!("Dynamic dispatch does not support empty ChunkedArray");
        }
        self.walk(children.get(0), row_count)
    }

    // ── Helpers ──────────────────────────────────────────────────────

    /// Walk a subtree and add it as an input stage writing to shared memory.
    fn add_input_stage(&mut self, node: fba::ArrayNode<'_>, len: u64) -> VortexResult<u32> {
        let smem_offset = self.smem_cursor;
        let pipeline = self.walk(node, len)?;
        self.stages.push(Stage::input(
            pipeline.input_ptr,
            smem_offset,
            len as u32,
            pipeline.source,
            &pipeline.scalar_ops,
        ));
        self.smem_cursor += len as u32;
        Ok(smem_offset)
    }

    /// Resolve a buffer handle to a device-visible pointer.
    ///
    /// On GH200 (PageableMemoryAccess=1), host pointers are passed directly.
    /// On discrete GPUs, falls back to H2D copy.
    fn resolve_buffer_ptr(&mut self, buffer: BufferHandle) -> VortexResult<u64> {
        if buffer.is_on_device() {
            let ptr = buffer.cuda_device_ptr()?;
            self.device_buffers.push(buffer);
            Ok(ptr)
        } else if let Some(host_buf) = buffer.as_host_opt() {
            let ptr = host_buf.as_ptr() as u64;
            self.device_buffers.push(buffer);
            Ok(ptr)
        } else {
            let device_buf = futures::executor::block_on(self.cuda_ctx.ensure_on_device(buffer))?;
            let ptr = device_buf.cuda_device_ptr()?;
            self.device_buffers.push(device_buf);
            Ok(ptr)
        }
    }
}

// ── FoR reference extraction ────────────────────────────────────────────

/// Extract the FoR reference value as raw u64 bits directly from the
/// protobuf-serialized `ScalarValue` metadata bytes.
///
/// FoR serializes `ScalarValue::to_proto_bytes(Some(&reference))`, which is a
/// protobuf `ScalarValue` message containing either `int64_value` or
/// `uint64_value`. We decode the proto and reinterpret based on the DType's
/// PType width.
fn for_reference_from_proto(metadata: &[u8], dtype: &DType) -> VortexResult<u64> {
    let proto = vortex::proto::scalar::ScalarValue::decode(metadata)
        .map_err(|e| vortex_err!("Failed to decode FoR reference proto: {e}"))?;

    if !dtype.is_primitive() {
        vortex_bail!(
            "Expected primitive dtype for FoR reference, got {:?}",
            dtype
        );
    }
    let ptype = dtype.as_ptype();

    // The proto stores either int64_value or uint64_value.
    // We need to re-narrow to the correct PType width, matching the
    // behavior of `extract_for_reference` in plan_builder.rs.
    use vortex::proto::scalar::scalar_value::Kind;
    let kind = proto
        .kind
        .as_ref()
        .ok_or_else(|| vortex_err!("FoR reference ScalarValue missing kind"))?;

    match kind {
        Kind::Int64Value(v) => {
            let v = *v;
            match ptype {
                PType::I8 => Ok(v as i8 as u8 as u64),
                PType::I16 => Ok(v as i16 as u16 as u64),
                PType::I32 => Ok(v as i32 as u32 as u64),
                PType::I64 => Ok(v as u64),
                PType::U8 => Ok(v as u8 as u64),
                PType::U16 => Ok(v as u16 as u64),
                PType::U32 => Ok(v as u32 as u64),
                PType::U64 => Ok(v as u64),
                _ => vortex_bail!("Unexpected ptype {:?} for FoR int64 reference", ptype),
            }
        }
        Kind::Uint64Value(v) => {
            let v = *v;
            match ptype {
                PType::U8 => Ok(v & 0xFF),
                PType::U16 => Ok(v & 0xFFFF),
                PType::U32 => Ok(v & 0xFFFF_FFFF),
                PType::U64 => Ok(v),
                PType::I8 => Ok(v as i8 as u8 as u64),
                PType::I16 => Ok(v as i16 as u16 as u64),
                PType::I32 => Ok(v as i32 as u32 as u64),
                PType::I64 => Ok(v),
                _ => vortex_bail!("Unexpected ptype {:?} for FoR uint64 reference", ptype),
            }
        }
        _ => vortex_bail!("FoR reference must be an integer ScalarValue"),
    }
}

/// Extract a raw u64 from a protobuf ScalarValue (for Sequence base/multiplier).
fn scalar_value_to_u64(sv: &vortex::proto::scalar::ScalarValue) -> VortexResult<u64> {
    use vortex::proto::scalar::scalar_value::Kind;
    let kind = sv
        .kind
        .as_ref()
        .ok_or_else(|| vortex_err!("ScalarValue missing kind"))?;

    match kind {
        Kind::Int64Value(v) => Ok(*v as u64),
        Kind::Uint64Value(v) => Ok(*v),
        Kind::F32Value(v) => Ok(v.to_bits() as u64),
        Kind::F64Value(v) => Ok(v.to_bits()),
        _ => vortex_bail!("Cannot convert ScalarValue to u64 bits"),
    }
}

/// Mirror of `DictMetadata` — we only need `values_len`.
#[derive(prost::Message)]
struct DictMetadataProto {
    #[prost(uint32, tag = "1")]
    values_len: u32,
    #[prost(int32, tag = "2")]
    _codes_ptype: i32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_value_to_u64_int() {
        use vortex::proto::scalar::scalar_value::Kind;
        let sv = vortex::proto::scalar::ScalarValue {
            kind: Some(Kind::Int64Value(42)),
        };
        assert_eq!(scalar_value_to_u64(&sv).unwrap(), 42);
    }

    #[test]
    fn test_scalar_value_to_u64_negative() {
        use vortex::proto::scalar::scalar_value::Kind;
        let sv = vortex::proto::scalar::ScalarValue {
            kind: Some(Kind::Int64Value(-1)),
        };
        assert_eq!(scalar_value_to_u64(&sv).unwrap(), u64::MAX);
    }

    #[test]
    fn test_for_reference_u32() {
        use vortex::proto::scalar::scalar_value::Kind;
        let dtype = DType::Primitive(PType::U32, vortex::dtype::Nullability::NonNullable);

        // Encode a u32 reference value as proto bytes
        let proto = vortex::proto::scalar::ScalarValue {
            kind: Some(Kind::Uint64Value(12345)),
        };
        let bytes = proto.encode_to_vec();

        let result = for_reference_from_proto(&bytes, &dtype).unwrap();
        assert_eq!(result, 12345u64);
    }

    #[test]
    fn test_for_reference_i32_negative() {
        use vortex::proto::scalar::scalar_value::Kind;
        let dtype = DType::Primitive(PType::I32, vortex::dtype::Nullability::NonNullable);

        // Encode an i32 reference value as proto bytes
        let proto = vortex::proto::scalar::ScalarValue {
            kind: Some(Kind::Int64Value(-100)),
        };
        let bytes = proto.encode_to_vec();

        let result = for_reference_from_proto(&bytes, &dtype).unwrap();
        // -100i32 as u32 as u64
        assert_eq!(result, (-100i32 as u32) as u64);
    }
}
