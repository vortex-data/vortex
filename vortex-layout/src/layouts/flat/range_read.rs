// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Range read support for FlatLayout segments.
//!
//! When the `array_tree` metadata is inlined in the layout (via `FLAT_LAYOUT_INLINE_ARRAY_NODE`),
//! we can inspect the encoding tree to compute which byte ranges of the segment are needed for a
//! given row range. This allows us to issue a smaller, targeted IO instead of reading the entire
//! segment.
//!
//! Each encoding implements [`VTable::plan_range_read`] to describe how its buffers and children
//! should be handled. The planner dispatches via vtable and recursively walks the encoding tree.

use std::ops::Range;
use std::sync::Arc;

use flatbuffers::root;
use futures::FutureExt;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::serde::ArrayParts;
use vortex_array::session::ArraySessionExt;
use vortex_array::vtable::range_read::BufferSubRange;
use vortex_array::vtable::range_read::ChildRangeRead;
use vortex_array::vtable::range_read::RangeDecodeInfo;
use vortex_buffer::Alignment;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_flatbuffers::array as fba;
use vortex_session::VortexSession;
use vortex_session::registry::ReadContext;

use crate::layouts::SharedArrayFuture;
use crate::segments::SegmentId;
use crate::segments::SegmentSource;

/// Maximum ratio of (range_size / full_segment_size) at which we attempt a range read.
/// If the range read would read more than this fraction of the full segment, we fall back to
/// a full read to avoid the overhead of range computation.
const RANGE_READ_THRESHOLD: f64 = 0.5;

/// A plan describing which bytes to read from a segment for a given row range.
#[derive(Debug)]
pub(super) struct RangeReadPlan {
    /// The byte range to read from the segment.
    segment_byte_range: Range<usize>,
    /// For each buffer index in the array_tree, the byte range within the partial segment
    /// that contains that buffer's data. Empty range means the buffer is not needed.
    buffer_ranges: Vec<Range<usize>>,
    /// The alignment requirement for each buffer.
    buffer_alignments: Vec<Alignment>,
    /// The number of logical rows to pass to `decode`.
    decode_len: usize,
    /// After decoding, slice the result to this range (relative to decode output).
    /// `None` means no post-decode slicing is needed.
    post_slice: Option<Range<usize>>,
}

/// Decode information returned by encoding analysis.
/// Describes how to decode the array after a range read.
#[derive(Debug, Clone)]
struct DecodeInfo {
    /// The number of rows to pass to `decode()` at this level.
    decode_len: usize,
    /// After decoding, slice to this range. `None` means no slicing needed.
    post_slice: Option<Range<usize>>,
}

/// Buffer offset, length, and alignment within the full segment.
#[derive(Debug, Clone)]
struct BufferLocation {
    /// Byte offset within the full segment (after padding).
    offset: usize,
    /// Length in bytes.
    length: usize,
    /// Alignment requirement.
    alignment: Alignment,
}

/// Compute buffer locations from the Array flatbuffer's buffer descriptors.
fn compute_buffer_locations(fb_array: &fba::Array<'_>) -> Vec<BufferLocation> {
    let mut offset = 0usize;
    fb_array
        .buffers()
        .unwrap_or_default()
        .iter()
        .map(|buf| {
            offset += buf.padding() as usize;
            let loc = BufferLocation {
                offset,
                length: buf.length() as usize,
                alignment: Alignment::from_exponent(buf.alignment_exponent()),
            };
            offset += buf.length() as usize;
            loc
        })
        .collect()
}

/// Tracks which buffers are needed and their required byte sub-ranges.
struct NeededBuffers {
    /// For each buffer index: `Some(sub_range_within_buffer)` if needed, `None` if not.
    entries: Vec<Option<Range<usize>>>,
}

impl NeededBuffers {
    fn new(num_buffers: usize) -> Self {
        Self {
            entries: vec![None; num_buffers],
        }
    }

    /// Mark a buffer as fully needed.
    fn need_full(&mut self, buffer_idx: u16) {
        let idx = buffer_idx as usize;
        if idx < self.entries.len() {
            self.entries[idx] = Some(0..usize::MAX);
        }
    }

    /// Mark a sub-range of a buffer as needed.
    fn need_range(&mut self, buffer_idx: u16, range: Range<usize>) {
        let idx = buffer_idx as usize;
        if idx < self.entries.len() {
            match &self.entries[idx] {
                Some(existing) if existing.end == usize::MAX => {
                    // Already marked as fully needed, keep it.
                }
                Some(existing) => {
                    // Merge ranges (union).
                    let start = existing.start.min(range.start);
                    let end = existing.end.max(range.end);
                    self.entries[idx] = Some(start..end);
                }
                None => {
                    self.entries[idx] = Some(range);
                }
            }
        }
    }
}

/// Recursively analyze the encoding tree via vtable dispatch to determine which buffer
/// byte ranges are needed for the given row range.
///
/// Returns `Some(DecodeInfo)` if the encoding tree supports range reads, `None` to fall back.
fn analyze_encoding(
    node: fba::ArrayNode<'_>,
    row_range: Range<usize>,
    row_count: usize,
    dtype: &DType,
    ctx: &ReadContext,
    session: &VortexSession,
    needed: &mut NeededBuffers,
) -> Option<DecodeInfo> {
    let encoding_id = ctx.resolve(node.encoding())?;
    let vtable = session.arrays().registry().find(&encoding_id)?;
    let metadata_bytes = node.metadata().map(|m| m.bytes()).unwrap_or(&[]);

    let plan = vtable.plan_range_read(metadata_bytes, row_range, row_count, dtype, session)?;

    // Apply buffer sub-ranges (local index → global index via flatbuffer).
    apply_buffer_sub_ranges(node, &plan.buffer_sub_ranges, needed);

    // Match plan children against node children and recurse.
    let child_decode_infos = resolve_children(node, &plan.children, ctx, session, needed)?;

    // Resolve decode info from the plan.
    resolve_decode_info(plan.decode_info, &child_decode_infos)
}

/// Map the plan's buffer sub-ranges to global buffer indices via the flatbuffer node.
fn apply_buffer_sub_ranges(
    node: fba::ArrayNode<'_>,
    sub_ranges: &[BufferSubRange],
    needed: &mut NeededBuffers,
) {
    if let Some(buffers) = node.buffers() {
        for (local_idx, sub_range) in sub_ranges.iter().enumerate() {
            if local_idx < buffers.len() {
                let global_idx = buffers.get(local_idx);
                match sub_range {
                    BufferSubRange::Full => needed.need_full(global_idx),
                    BufferSubRange::Range(range) => needed.need_range(global_idx, range.clone()),
                }
            }
        }
    }
}

/// Pair each plan child with its corresponding node child and process them.
///
/// Returns `None` if the plan's children don't exactly match the node's children count,
/// which typically means there is an unhandled validity child.
fn resolve_children(
    node: fba::ArrayNode<'_>,
    plan_children: &[ChildRangeRead],
    ctx: &ReadContext,
    session: &VortexSession,
    needed: &mut NeededBuffers,
) -> Option<Vec<Option<DecodeInfo>>> {
    let node_children: Vec<_> = node
        .children()
        .map_or(vec![], |c| (0..c.len()).map(|i| c.get(i)).collect());

    // Plan must cover every child in the node. Un-covered children would have their
    // buffers missing during decode.
    if plan_children.len() != node_children.len() {
        return None;
    }

    let mut decode_infos = Vec::with_capacity(plan_children.len());
    for (action, child_node) in plan_children.iter().zip(node_children.iter()) {
        match action {
            ChildRangeRead::Recurse {
                row_range,
                row_count,
                dtype,
            } => {
                let info = analyze_encoding(
                    *child_node,
                    row_range.clone(),
                    *row_count,
                    dtype,
                    ctx,
                    session,
                    needed,
                );
                decode_infos.push(info);
            }
            ChildRangeRead::Full => {
                need_all_node_buffers(*child_node, needed);
                decode_infos.push(None);
            }
        }
    }

    Some(decode_infos)
}

/// Compute the final decode parameters from the plan's decode info and children results.
fn resolve_decode_info(
    decode_info: RangeDecodeInfo,
    child_decode_infos: &[Option<DecodeInfo>],
) -> Option<DecodeInfo> {
    match decode_info {
        RangeDecodeInfo::Leaf {
            decode_len,
            post_slice,
        } => Some(DecodeInfo {
            decode_len,
            post_slice,
        }),
        RangeDecodeInfo::FromChild { child_idx, divisor } => {
            let child_info = child_decode_infos.get(child_idx)?.as_ref()?;
            if divisor == 1 {
                return Some(child_info.clone());
            }
            if child_info.decode_len % divisor != 0 {
                return None;
            }
            let scaled_post_slice = match &child_info.post_slice {
                None => None,
                Some(ps) => {
                    if ps.start % divisor != 0 || ps.end % divisor != 0 {
                        return None;
                    }
                    Some(ps.start / divisor..ps.end / divisor)
                }
            };
            Some(DecodeInfo {
                decode_len: child_info.decode_len / divisor,
                post_slice: scaled_post_slice,
            })
        }
    }
}

/// Recursively mark all buffers in a node (and its children) as fully needed.
fn need_all_node_buffers(node: fba::ArrayNode<'_>, needed: &mut NeededBuffers) {
    if let Some(buffers) = node.buffers() {
        for i in 0..buffers.len() {
            needed.need_full(buffers.get(i));
        }
    }
    if let Some(children) = node.children() {
        for i in 0..children.len() {
            need_all_node_buffers(children.get(i), needed);
        }
    }
}

/// Attempt to build a range read plan for the given array tree and row range.
///
/// Returns `None` if:
/// - The encoding does not support range reads (fallback to full segment read).
/// - The row range covers the entire segment (no benefit).
/// - The computed byte range is not significantly smaller than the full segment.
pub(super) fn try_plan_range_read(
    array_tree: &ByteBuffer,
    row_range: Range<usize>,
    row_count: usize,
    dtype: &DType,
    ctx: &ReadContext,
    session: &VortexSession,
) -> VortexResult<Option<RangeReadPlan>> {
    // Should not happen, but guard against empty ranges to avoid downstream arithmetic issues
    // (e.g. saturating_sub overflow in analyze_bitpacked).
    if row_range.is_empty() {
        return Ok(None);
    }

    // No benefit if we need all rows.
    if row_range.start == 0 && row_range.end >= row_count {
        return Ok(None);
    }

    // Parse the flatbuffer.
    let fb_array = root::<fba::Array>(array_tree.as_ref())
        .map_err(|e| vortex_err!("invalid array tree flatbuffer: {e}"))?;

    let buffer_locations = compute_buffer_locations(&fb_array);
    let num_buffers = buffer_locations.len();

    let root_node = fb_array
        .root()
        .ok_or_else(|| vortex_err!("array tree has no root node"))?;

    // Analyze the encoding tree via vtable dispatch.
    let mut needed = NeededBuffers::new(num_buffers);
    let decode_info = match analyze_encoding(
        root_node,
        row_range,
        row_count,
        dtype,
        ctx,
        session,
        &mut needed,
    ) {
        Some(info) => info,
        None => return Ok(None),
    };

    // Resolve the needed ranges against the buffer locations.
    let mut min_offset = usize::MAX;
    let mut max_end = 0usize;
    for (i, entry) in needed.entries.iter_mut().enumerate() {
        if let Some(range) = entry {
            let loc = &buffer_locations[i];
            // Replace sentinel with full range.
            if range.end == usize::MAX {
                *range = 0..loc.length;
            }
            // Clamp to actual buffer length.
            range.end = range.end.min(loc.length);

            let abs_start = loc.offset + range.start;
            let abs_end = loc.offset + range.end;
            min_offset = min_offset.min(abs_start);
            max_end = max_end.max(abs_end);
        }
    }

    if min_offset >= max_end {
        return Ok(None);
    }

    let segment_byte_range = min_offset..max_end;

    // Check if the range read is worth it.
    let full_segment_size: usize = buffer_locations
        .last()
        .map(|loc| loc.offset + loc.length)
        .unwrap_or(0);
    if full_segment_size == 0 {
        return Ok(None);
    }
    let ratio = segment_byte_range.len() as f64 / full_segment_size as f64;
    if ratio > RANGE_READ_THRESHOLD {
        return Ok(None);
    }

    // Compute per-buffer ranges within the partial segment and collect alignments.
    let partial_offset = segment_byte_range.start;
    let mut buffer_ranges = Vec::with_capacity(num_buffers);
    let mut buffer_alignments = Vec::with_capacity(num_buffers);
    for (i, entry) in needed.entries.iter().enumerate() {
        let loc = &buffer_locations[i];
        buffer_alignments.push(loc.alignment);
        if let Some(sub_range) = entry {
            let abs_start = loc.offset + sub_range.start;
            let abs_end = loc.offset + sub_range.end;
            buffer_ranges.push((abs_start - partial_offset)..(abs_end - partial_offset));
        } else {
            buffer_ranges.push(0..0);
        }
    }

    Ok(Some(RangeReadPlan {
        segment_byte_range,
        buffer_ranges,
        buffer_alignments,
        decode_len: decode_info.decode_len,
        post_slice: decode_info.post_slice,
    }))
}

/// Execute a range read plan: issue a targeted IO, build ArrayParts from partial buffers, decode.
pub(super) fn execute_range_read(
    plan: RangeReadPlan,
    array_tree: ByteBuffer,
    segment_id: SegmentId,
    segment_source: Arc<dyn SegmentSource>,
    dtype: DType,
    ctx: ReadContext,
    session: VortexSession,
) -> SharedArrayFuture {
    async move {
        // 1. Issue the targeted read.
        let partial = segment_source
            .request_range(segment_id, plan.segment_byte_range.clone())
            .await?;
        let partial_bytes = partial.try_to_host_sync()?;

        // 2. Slice individual buffers from the partial segment, ensuring alignment.
        let mut buffers = Vec::with_capacity(plan.buffer_ranges.len());
        for (i, buf_range) in plan.buffer_ranges.iter().enumerate() {
            if buf_range.is_empty() {
                buffers.push(BufferHandle::new_host(ByteBuffer::empty()));
            } else {
                let slice = partial_bytes.slice_unaligned(buf_range.clone());
                let aligned =
                    BufferHandle::new_host(slice).ensure_aligned(plan.buffer_alignments[i])?;
                buffers.push(aligned);
            }
        }

        // 3. Build ArrayParts and decode.
        let parts = ArrayParts::from_flatbuffer_with_buffers(array_tree, buffers)?;
        let mut array = parts.decode(&dtype, plan.decode_len, &ctx, &session)?;

        // 4. Post-decode slice if needed (block alignment).
        if let Some(slice_range) = plan.post_slice {
            array = array.slice(slice_range)?;
        }

        Ok(array)
    }
    .map(|r| r.map_err(Arc::new))
    .boxed()
    .shared()
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation, clippy::unnecessary_cast)]
mod tests {
    use std::sync::Arc;

    use rstest::rstest;
    use vortex_array::ArrayContext;
    use vortex_array::DynArray;
    use vortex_array::IntoArray;
    use vortex_array::MaskFuture;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::Dict;
    use vortex_array::arrays::DictArray;
    use vortex_array::arrays::FixedSizeListArray;
    use vortex_array::arrays::NullArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::expr::root;
    use vortex_array::scalar_fn::session::ScalarFnSession;
    use vortex_array::serde::SerializeOptions;
    use vortex_array::session::ArraySessionExt;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_buffer::buffer;
    use vortex_bytebool::ByteBool;
    use vortex_bytebool::ByteBoolArray;
    use vortex_error::VortexResult;
    use vortex_fastlanes::BitPacked;
    use vortex_fastlanes::BitPackedArray;
    use vortex_fastlanes::Delta;
    use vortex_fastlanes::DeltaArray;
    use vortex_fastlanes::FoR;
    use vortex_fastlanes::FoRArray;
    use vortex_fastlanes::delta_compress;
    use vortex_io::runtime::single::block_on;
    use vortex_io::session::RuntimeSession;
    use vortex_session::SessionExt;
    use vortex_session::registry::ReadContext;
    use vortex_zigzag::ZigZag;
    use vortex_zigzag::ZigZagArray;

    use super::*;
    use crate::layouts::flat::FlatLayout;
    use crate::layouts::flat::reader::FlatReader;
    use crate::segments::SegmentSink;
    use crate::segments::TestSegments;
    use crate::sequence::SequenceId;
    use crate::session::LayoutSession;
    use crate::session::RangeReadEnabled;
    use crate::test::SESSION;

    /// Helper: serialize an array and return (array_tree, segment_id, segments).
    fn write_with_array_tree(
        array: &dyn DynArray,
    ) -> VortexResult<(ByteBuffer, SegmentId, Arc<TestSegments>, ReadContext)> {
        let ctx = ArrayContext::empty();
        let buffers = array.serialize(
            &ctx,
            &SerializeOptions {
                offset: 0,
                include_padding: true,
            },
        )?;
        let array_tree = buffers[buffers.len() - 2].clone();
        let segments = Arc::new(TestSegments::default());
        let segment_id =
            block_on(|_| async { segments.write(SequenceId::root().advance(), buffers).await })?;
        let read_ctx = ReadContext::new(ctx.to_ids());
        Ok((array_tree, segment_id, segments, read_ctx))
    }

    #[test]
    fn plan_returns_none_for_full_range() -> VortexResult<()> {
        let array = PrimitiveArray::new(
            buffer![1i32, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            Validity::AllValid,
        )
        .into_array();
        let (array_tree, _, _, read_ctx) = write_with_array_tree(array.as_ref())?;

        let plan = try_plan_range_read(&array_tree, 0..10, 10, array.dtype(), &read_ctx, &SESSION)?;
        assert!(plan.is_none());
        Ok(())
    }

    #[test]
    fn plan_returns_some_for_sub_range() -> VortexResult<()> {
        let array = PrimitiveArray::new(
            buffer![1i32, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            Validity::AllValid,
        )
        .into_array();
        let (array_tree, _, _, read_ctx) = write_with_array_tree(array.as_ref())?;

        let plan = try_plan_range_read(&array_tree, 2..4, 10, array.dtype(), &read_ctx, &SESSION)?;
        let plan = plan.ok_or_else(|| vortex_err!("expected Some plan"))?;
        assert_eq!(plan.decode_len, 2);
        assert!(plan.post_slice.is_none());
        Ok(())
    }

    #[rstest]
    #[case(0..3, &[1i32, 2, 3])]
    #[case(2..5, &[3, 4, 5])]
    #[case(7..10, &[8, 9, 10])]
    #[case(0..1, &[1])]
    #[case(9..10, &[10])]
    fn primitive_range_read_end_to_end(
        #[case] row_range: Range<usize>,
        #[case] expected: &[i32],
    ) -> VortexResult<()> {
        let array = PrimitiveArray::new(
            buffer![1i32, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            Validity::AllValid,
        )
        .into_array();
        let (array_tree, segment_id, segments, read_ctx) = write_with_array_tree(array.as_ref())?;

        let layout = FlatLayout::new_with_metadata(
            10,
            array.dtype().clone(),
            segment_id,
            read_ctx,
            Some(array_tree),
        );

        let expected_array =
            PrimitiveArray::new(Buffer::<i32>::from(expected.to_vec()), Validity::AllValid)
                .into_array();

        block_on(|_| async {
            let reader = FlatReader::new(
                layout,
                "test".into(),
                segments as Arc<dyn SegmentSource>,
                SESSION.clone(),
            );

            let result = crate::LayoutReader::projection_evaluation(
                &reader,
                &(row_range.start as u64..row_range.end as u64),
                &root(),
                MaskFuture::new_true(row_range.len()),
            )?
            .await?;

            assert_arrays_eq!(result, expected_array);
            Ok(())
        })
    }

    #[test]
    fn range_read_disabled_via_session() -> VortexResult<()> {
        let array = PrimitiveArray::new(
            buffer![1i32, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            Validity::AllValid,
        )
        .into_array();
        let (array_tree, segment_id, segments, read_ctx) = write_with_array_tree(array.as_ref())?;

        // Create a session with range read disabled.
        let session = VortexSession::empty()
            .with::<vortex_array::session::ArraySession>()
            .with::<LayoutSession>()
            .with::<ScalarFnSession>()
            .with::<RuntimeSession>();
        session.get_mut::<RangeReadEnabled>().0 = false;

        let layout = FlatLayout::new_with_metadata(
            10,
            array.dtype().clone(),
            segment_id,
            read_ctx,
            Some(array_tree),
        );

        // Even though array_tree is present, range read should be skipped
        // and the reader should fall back to full segment read + slice.
        let expected = PrimitiveArray::new(buffer![3i32, 4], Validity::AllValid).into_array();

        block_on(|_| async {
            let reader = FlatReader::new(
                layout,
                "test".into(),
                segments as Arc<dyn SegmentSource>,
                session,
            );

            let result = crate::LayoutReader::projection_evaluation(
                &reader,
                &(2..4),
                &root(),
                MaskFuture::new_true(2),
            )?
            .await?;

            assert_arrays_eq!(result, expected);
            Ok(())
        })
    }

    #[test]
    fn fallback_without_array_tree() -> VortexResult<()> {
        let array = PrimitiveArray::new(buffer![1i32, 2, 3, 4, 5], Validity::AllValid).into_array();
        let ctx = ArrayContext::empty();
        let buffers = array.serialize(
            &ctx,
            &SerializeOptions {
                offset: 0,
                include_padding: true,
            },
        )?;
        let segments = Arc::new(TestSegments::default());
        let segment_id =
            block_on(|_| async { segments.write(SequenceId::root().advance(), buffers).await })?;

        // No array_tree → range read not possible, should fall back to full read.
        let layout = FlatLayout::new(
            5,
            array.dtype().clone(),
            segment_id,
            ReadContext::new(ctx.to_ids()),
        );

        let expected = PrimitiveArray::new(buffer![3i32, 4], Validity::AllValid).into_array();

        block_on(|_| async {
            let reader = FlatReader::new(
                layout,
                "test".into(),
                segments as Arc<dyn SegmentSource>,
                SESSION.clone(),
            );

            let result = crate::LayoutReader::projection_evaluation(
                &reader,
                &(2..4),
                &root(),
                MaskFuture::new_true(2),
            )?
            .await?;

            assert_arrays_eq!(result, expected);
            Ok(())
        })
    }

    // For each supported encoding, write the array with array_tree metadata,
    // read a sub-range via FlatReader, and compare with the canonical array
    // sliced to the same range.

    const N: usize = 4096;

    /// Helper: round-trip an encoded array through write → FlatReader sub-range read,
    /// and compare with `canonical.slice(row_range)`.
    fn roundtrip_range(
        encoded: &dyn DynArray,
        canonical: &dyn DynArray,
        row_range: Range<usize>,
    ) -> VortexResult<()> {
        let (array_tree, segment_id, segments, read_ctx) = write_with_array_tree(encoded)?;
        let row_count = encoded.len() as u64;
        let layout = FlatLayout::new_with_metadata(
            row_count,
            encoded.dtype().clone(),
            segment_id,
            read_ctx,
            Some(array_tree),
        );
        let expected = canonical.slice(row_range.clone())?;

        // Register encodings needed by these tests.
        SESSION.arrays().register(Dict::ID, Dict);
        SESSION.arrays().register(BitPacked::ID, BitPacked);
        SESSION.arrays().register(FoR::ID, FoR);
        SESSION.arrays().register(Delta::ID, Delta);
        SESSION.arrays().register(ZigZag::ID, ZigZag);

        block_on(|_| async {
            let reader = FlatReader::new(
                layout,
                "test".into(),
                segments as Arc<dyn SegmentSource>,
                SESSION.clone(),
            );
            let result = crate::LayoutReader::projection_evaluation(
                &reader,
                &(row_range.start as u64..row_range.end as u64),
                &root(),
                MaskFuture::new_true(row_range.len()),
            )?
            .await?;
            assert_arrays_eq!(result, expected);
            Ok(())
        })
    }

    /// Non-fallback: range read IS used for these encodings.
    #[test]
    fn non_fallback_range_reads() -> VortexResult<()> {
        let ranges: &[Range<usize>] = &[
            0..1,
            42..43,
            512..513,
            1000..1001,
            1023..1024,
            1024..1025,
            2048..2049,
            3000..3001,
            3500..3501,
            4095..4096,
            1500..1510,
        ];

        // Bool (byte-aligned starts only).
        let bool_ranges: &[Range<usize>] = &[
            0..1,
            8..9,
            16..17,
            64..65,
            128..129,
            256..257,
            512..513,
            1024..1025,
            2048..2049,
            4088..4089,
            1024..1034,
        ];
        let bool_arr = BoolArray::from_iter((0..N).map(|i| i % 3 == 0));
        for r in bool_ranges {
            roundtrip_range(bool_arr.as_ref(), bool_arr.as_ref(), r.clone())?;
        }

        // BitPacked (no patches, no validity).
        let bp_values: PrimitiveArray = (0..N).map(|i| (i % 256) as u32).collect();
        let bp_canonical = bp_values.into_array();
        let bp_encoded = BitPackedArray::encode(&bp_canonical, 8)?;
        for r in ranges {
            roundtrip_range(bp_encoded.as_ref(), bp_canonical.as_ref(), r.clone())?;
        }

        // FoR (transparent wrapper → Primitive child).
        let for_build = || -> PrimitiveArray { (0..N).map(|i| (i + 1000) as i32).collect() };
        let for_canonical = for_build().into_array();
        let for_encoded = FoRArray::encode(for_build())?;
        for r in ranges {
            roundtrip_range(for_encoded.as_ref(), for_canonical.as_ref(), r.clone())?;
        }

        // Delta.
        let delta_build = || -> PrimitiveArray { (0..N).map(|i| (i * 3 + 100) as u32).collect() };
        let delta_canonical = delta_build().into_array();
        let delta_encoded = DeltaArray::try_from_primitive_array(&delta_build())?;
        for r in ranges {
            roundtrip_range(delta_encoded.as_ref(), delta_canonical.as_ref(), r.clone())?;
        }

        // Dict (high positions so codes sub-range + values < 50% of segment).
        let dict_ranges: &[Range<usize>] = &[
            3000..3001,
            3200..3201,
            3400..3401,
            3600..3601,
            3800..3801,
            4095..4096,
            3800..3810,
        ];
        let codes: PrimitiveArray = (0..N).map(|i| (i % 5) as u8).collect();
        let values: PrimitiveArray = (0..5).map(|i| (i * 10) as i32).collect();
        let dict_encoded = DictArray::new(codes.into_array(), values.into_array());
        let dict_canonical: PrimitiveArray = (0..N).map(|i| ((i % 5) * 10) as i32).collect();
        for r in dict_ranges {
            roundtrip_range(dict_encoded.as_ref(), dict_canonical.as_ref(), r.clone())?;
        }

        // FixedSizeList (non-nullable, Primitive child).
        let list_size = 4u32;
        let total = N * list_size as usize;
        let build_fsl = || {
            let flat: PrimitiveArray = (0..total).map(|i| i as u32).collect();
            FixedSizeListArray::try_new(flat.into_array(), list_size, Validity::NonNullable, N)
                .unwrap()
        };
        let fsl_encoded = build_fsl();
        let fsl_canonical = build_fsl();
        for r in ranges {
            roundtrip_range(fsl_encoded.as_ref(), fsl_canonical.as_ref(), r.clone())?;
        }

        // ── Nested: FoR → BitPacked ──
        // Manually construct: BitPack shifted values, then wrap in FoR.
        let for_bp_canonical: PrimitiveArray = (0..N).map(|i| (i + 1000) as i32).collect();
        let shifted: PrimitiveArray = (0..N).map(|i| i as i32).collect();
        let bp = BitPackedArray::encode(&shifted.into_array(), 12)?;
        let for_bp = FoRArray::try_new(bp.into_array(), 1000i32.into())?;
        for r in ranges {
            roundtrip_range(for_bp.as_ref(), for_bp_canonical.as_ref(), r.clone())?;
        }

        // ── Nested: Delta → BitPacked deltas ──
        let delta_bp_build =
            || -> PrimitiveArray { (0..N).map(|i| (i * 3 + 100) as u32).collect() };
        let delta_bp_canonical = delta_bp_build().into_array();
        let (bases, deltas) = delta_compress(&delta_bp_build())?;
        let deltas_len = deltas.len();
        let bp_deltas = BitPackedArray::encode(&deltas.into_array(), 16)?;
        let delta_bp =
            DeltaArray::try_new(bases.into_array(), bp_deltas.into_array(), 0, deltas_len)?;
        for r in ranges {
            roundtrip_range(delta_bp.as_ref(), delta_bp_canonical.as_ref(), r.clone())?;
        }

        // ── Nested: FixedSizeList → BitPacked child ──
        let fsl_bp_canonical = build_fsl();
        let flat_bp: PrimitiveArray = (0..total).map(|i| i as u32).collect();
        let bp_flat = BitPackedArray::encode(&flat_bp.into_array(), 16)?;
        let fsl_bp =
            FixedSizeListArray::try_new(bp_flat.into_array(), list_size, Validity::NonNullable, N)?;
        for r in ranges {
            roundtrip_range(fsl_bp.as_ref(), fsl_bp_canonical.as_ref(), r.clone())?;
        }

        // ByteBool (1 byte per element).
        SESSION.arrays().register(ByteBool::ID, ByteBool);
        let bb_values: Vec<bool> = (0..N).map(|i| i % 3 == 0).collect();
        let bb = ByteBoolArray::from(bb_values);
        // ByteBool roundtrips through itself (not BoolArray) since it decodes to bool?.
        for r in ranges {
            roundtrip_range(bb.as_ref(), bb.as_ref(), r.clone())?;
        }

        // Null (no buffers, no children).
        let null_arr = NullArray::new(N);
        for r in ranges {
            roundtrip_range(null_arr.as_ref(), null_arr.as_ref(), r.clone())?;
        }

        Ok(())
    }

    /// Fallback: range read falls back to full segment read + slice.
    #[test]
    fn fallback_range_reads() -> VortexResult<()> {
        let ranges: &[Range<usize>] = &[
            0..1,
            42..43,
            512..513,
            1000..1001,
            1023..1024,
            1024..1025,
            2048..2049,
            3000..3001,
            3500..3501,
            4095..4096,
            1500..1510,
        ];

        // Nullable Primitive (validity child → range read unsupported).
        let values: Vec<i32> = (0..N).map(|i| (i * 7 + 13) as i32).collect();
        let validity = Validity::from_iter((0..N).map(|i| i % 7 != 0));
        let nullable = PrimitiveArray::new(Buffer::from(values), validity);
        for r in ranges {
            roundtrip_range(nullable.as_ref(), nullable.as_ref(), r.clone())?;
        }

        // Bool with non-byte-aligned start.
        let unaligned_ranges: &[Range<usize>] = &[
            1..2,
            3..4,
            5..6,
            7..8,
            13..14,
            42..43,
            100..101,
            999..1000,
            2047..2048,
            3999..4000,
            1025..1035,
        ];
        let bool_arr = BoolArray::from_iter((0..N).map(|i| i % 3 == 0));
        for r in unaligned_ranges {
            roundtrip_range(bool_arr.as_ref(), bool_arr.as_ref(), r.clone())?;
        }

        // ── Nested fallback: FoR → nullable Primitive ──
        // FoR is transparent, recurses into nullable Primitive which has validity child.
        let validity = Validity::from_iter((0..N).map(|i| i % 5 != 0));
        let shifted: Vec<i32> = (0..N).map(|i| i as i32).collect();
        let shifted_arr = PrimitiveArray::new(Buffer::from(shifted), validity.clone());
        let for_nullable = FoRArray::try_new(shifted_arr.into_array(), 1000i32.into())?;
        let for_canonical_values: Vec<i32> = (0..N).map(|i| (i + 1000) as i32).collect();
        let for_canonical = PrimitiveArray::new(Buffer::from(for_canonical_values), validity);
        for r in ranges {
            roundtrip_range(for_nullable.as_ref(), for_canonical.as_ref(), r.clone())?;
        }

        // ── Nested fallback: ZigZag → nullable Primitive ──
        // ZigZag is transparent, recurses into nullable Primitive.
        let validity = Validity::from_iter((0..N).map(|i| i % 5 != 0));
        let encoded_values: Vec<u32> = (0..N).map(|i| (i * 2) as u32).collect();
        let encoded_arr = PrimitiveArray::new(Buffer::from(encoded_values), validity.clone());
        let zigzag = ZigZagArray::try_new(encoded_arr.into_array())?;
        // zigzag_decode(2*i) = i
        let zz_canonical_values: Vec<i32> = (0..N).map(|i| i as i32).collect();
        let zz_canonical = PrimitiveArray::new(Buffer::from(zz_canonical_values), validity);
        for r in ranges {
            roundtrip_range(zigzag.as_ref(), zz_canonical.as_ref(), r.clone())?;
        }

        Ok(())
    }
}
