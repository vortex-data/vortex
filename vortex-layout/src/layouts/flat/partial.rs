// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeSet;
use std::ops::Range;
use std::sync::Arc;

use futures::FutureExt;
use futures::future::try_join_all;
use prost::Message;
use vortex_alp::ALPRD;
use vortex_alp::ALPRDMetadata;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::FixedSizeList;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::Struct;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::patches::Patches;
use vortex_array::patches::PatchesMetadata;
use vortex_array::serde::SerializedArray;
use vortex_array::serde::SerializedBuffer;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_fastlanes::BitPacked;
use vortex_fastlanes::BitPackedMetadata;
use vortex_mask::AllOr;
use vortex_mask::Mask;
use vortex_session::VortexSession;
use vortex_session::registry::ReadContext;

use crate::layouts::flat::FlatLayout;
use crate::segments::SegmentFuture;
use crate::segments::SegmentId;
use crate::segments::SegmentSource;

#[derive(Clone)]
pub(super) struct PartialReadPlan {
    array_tree: ByteBuffer,
    bytes_per_row: usize,
    row_granularity: usize,
    kind: PartialReadKind,
}

#[derive(Clone)]
enum PartialReadKind {
    Fixed(Arc<[PlannedBuffer]>),
    Alprd(Box<ALPRDReadPlan>),
}

#[derive(Clone)]
struct PlannedBuffer {
    descriptor: SerializedBuffer,
    bytes_per_row: usize,
    row_granularity: usize,
    bytes_per_granule: usize,
}

#[derive(Clone)]
struct ALPRDReadPlan {
    serialized: SerializedArray,
    descriptors: Arc<[SerializedBuffer]>,
    left: BitPackedReadPlan,
    right: BitPackedReadPlan,
    patch_buffers: Arc<[SerializedBuffer]>,
    patch_metadata: PatchesMetadata,
    patch_indices_dtype: DType,
    left_parts_dtype: DType,
    left_parts_dictionary: Buffer<u16>,
    right_bit_width: u8,
    element_dtype: DType,
    list_size: u32,
    row_count: usize,
}

struct PageResolveContext<'a> {
    dtype: &'a DType,
    row_range: &'a Range<usize>,
    mask: &'a Mask,
    ctx: &'a ReadContext,
    session: &'a VortexSession,
}

#[derive(Clone)]
struct BitPackedReadPlan {
    descriptor: SerializedBuffer,
    ptype: vortex_array::dtype::PType,
    bit_width: u8,
    offset: u16,
}

pub(super) struct RegisteredPartialRead {
    array_tree: ByteBuffer,
    kind: RegisteredReadKind,
}

enum RegisteredReadKind {
    Fixed {
        pages: Vec<RegisteredPage>,
    },
    Alprd {
        pages: Vec<RegisteredALPRDPage>,
        patch_buffers: Vec<(SegmentFuture, SerializedBuffer)>,
        plan: ALPRDReadPlan,
    },
}

struct RegisteredALPRDPage {
    rows: Range<usize>,
    left: SegmentFuture,
    right: SegmentFuture,
}

struct RegisteredPage {
    rows: Range<usize>,
    buffers: Vec<(SegmentFuture, SerializedBuffer)>,
}

impl PartialReadPlan {
    pub(super) fn supports_mask(mask: &Mask) -> bool {
        !mask.all_true()
    }

    pub(super) fn try_new(layout: &FlatLayout) -> VortexResult<Option<Self>> {
        let Some(array_tree) = layout.array_tree().cloned() else {
            return Ok(None);
        };
        let serialized = SerializedArray::from_array_tree(array_tree.clone())?;
        let descriptors: Arc<[SerializedBuffer]> = serialized.buffer_descriptors()?.into();
        let row_count = usize::try_from(layout.row_count())?;

        if let Some((plan, bytes_per_row)) = try_alprd_plan(
            &serialized,
            layout.dtype(),
            layout.array_ctx(),
            row_count,
            Arc::clone(&descriptors),
        )? {
            return Ok(Some(Self {
                array_tree,
                bytes_per_row,
                row_granularity: 1,
                kind: PartialReadKind::Alprd(Box::new(plan)),
            }));
        }

        let mut planned = Vec::new();
        if !collect_raw_buffers(
            &serialized,
            layout.dtype(),
            layout.array_ctx(),
            1,
            row_count,
            &descriptors,
            &mut planned,
        )? {
            return Ok(None);
        }
        planned.sort_unstable_by_key(|buffer| buffer.descriptor.index());
        if planned.len() != descriptors.len()
            || planned
                .iter()
                .enumerate()
                .any(|(index, buffer)| buffer.descriptor.index() != index)
        {
            return Ok(None);
        }
        for buffer in &planned {
            let expected = row_count
                .div_ceil(buffer.row_granularity)
                .checked_mul(buffer.bytes_per_granule)
                .ok_or_else(|| vortex_err!("Partial buffer length overflow"))?;
            if buffer.descriptor.range().len() != expected {
                return Ok(None);
            }
        }
        let bytes_per_row = planned.iter().try_fold(0usize, |sum, buffer| {
            sum.checked_add(buffer.bytes_per_row)
                .ok_or_else(|| vortex_err!("Partial row width overflow"))
        })?;
        if bytes_per_row == 0 {
            return Ok(None);
        }
        let row_granularity = planned
            .iter()
            .map(|buffer| buffer.row_granularity)
            .try_fold(1usize, checked_lcm)?;
        Ok(Some(Self {
            array_tree,
            bytes_per_row,
            row_granularity,
            kind: PartialReadKind::Fixed(planned.into()),
        }))
    }

    pub(super) fn register(
        &self,
        source: &Arc<dyn SegmentSource>,
        segment_id: SegmentId,
        layout_len: usize,
        row_range: &Range<usize>,
        mask: &Mask,
    ) -> Option<RegisteredPartialRead> {
        if !Self::supports_mask(mask) {
            return None;
        }
        let preferred_read_size = usize::try_from(source.preferred_read_size()?).ok()?;
        let segment_len = usize::try_from(source.segment_len(segment_id)?).ok()?;
        let desired_rows = (preferred_read_size / self.bytes_per_row).max(1);
        let page_rows = desired_rows
            .div_ceil(self.row_granularity)
            .saturating_mul(self.row_granularity);
        let pages = selected_pages(page_rows, layout_len, row_range, mask)?;
        let pages = match &self.kind {
            PartialReadKind::Alprd(_) => selected_page_runs(&pages, row_range, mask)?,
            PartialReadKind::Fixed(_) => pages,
        };
        let (partial_bytes, request_count) = self.estimated_partial_io(&pages, layout_len)?;
        let partial_cost = partial_bytes.checked_add(
            request_count
                .saturating_sub(1)
                .checked_mul(preferred_read_size)?,
        )?;
        if partial_cost >= segment_len {
            tracing::trace!(
                layout_len,
                page_rows,
                page_count = pages.len(),
                partial_bytes,
                request_count,
                partial_cost,
                segment_len,
                "Flat partial read rejected by I/O cost"
            );
            return None;
        }
        tracing::trace!(
            layout_len,
            page_rows,
            page_count = pages.len(),
            partial_bytes,
            request_count,
            partial_cost,
            segment_len,
            "Flat partial read registered"
        );

        let kind = match &self.kind {
            PartialReadKind::Fixed(buffers) => {
                let page_specs = pages
                    .into_iter()
                    .map(|rows| {
                        let ranges = buffers
                            .iter()
                            .map(|buffer| {
                                let start = buffer.descriptor.range().start
                                    + (rows.start / buffer.row_granularity)
                                        * buffer.bytes_per_granule;
                                let end = buffer.descriptor.range().start
                                    + rows.end.div_ceil(buffer.row_granularity)
                                        * buffer.bytes_per_granule;
                                Some((
                                    u64::try_from(start).ok()?..u64::try_from(end).ok()?,
                                    buffer.descriptor.clone(),
                                ))
                            })
                            .collect::<Option<Vec<_>>>()?;
                        Some((rows, ranges))
                    })
                    .collect::<Option<Vec<_>>>()?;
                let requests = source.request_ranges(
                    segment_id,
                    page_specs
                        .iter()
                        .flat_map(|(_, ranges)| ranges.iter().map(|(range, _)| range.clone()))
                        .collect(),
                );
                let mut requests = requests.into_iter();
                let pages = page_specs
                    .into_iter()
                    .map(|(rows, ranges)| {
                        let buffers = ranges
                            .into_iter()
                            .map(|(_, descriptor)| Some((requests.next()?, descriptor)))
                            .collect::<Option<Vec<_>>>()?;
                        Some(RegisteredPage { rows, buffers })
                    })
                    .collect::<Option<Vec<_>>>()?;
                RegisteredReadKind::Fixed { pages }
            }
            PartialReadKind::Alprd(plan) => {
                let values_per_row = usize::try_from(plan.list_size).ok()?;
                let page_specs = pages
                    .into_iter()
                    .map(|rows| {
                        let inner_start = rows.start.checked_mul(values_per_row)?;
                        let inner_end = rows.end.checked_mul(values_per_row)?;
                        let left = bitpacked_range(&plan.left, inner_start..inner_end)?;
                        let right = bitpacked_range(&plan.right, inner_start..inner_end)?;
                        Some((
                            rows,
                            u64::try_from(left.start).ok()?..u64::try_from(left.end).ok()?,
                            u64::try_from(right.start).ok()?..u64::try_from(right.end).ok()?,
                        ))
                    })
                    .collect::<Option<Vec<_>>>()?;
                let patch_specs = plan
                    .patch_buffers
                    .iter()
                    .map(|descriptor| {
                        Some((
                            u64::try_from(descriptor.range().start).ok()?
                                ..u64::try_from(descriptor.range().end).ok()?,
                            descriptor.clone(),
                        ))
                    })
                    .collect::<Option<Vec<_>>>()?;
                let ranges = page_specs
                    .iter()
                    .flat_map(|(_, left, right)| [left.clone(), right.clone()])
                    .chain(patch_specs.iter().map(|(range, _)| range.clone()))
                    .collect();
                let mut requests = source.request_ranges(segment_id, ranges).into_iter();
                let pages = page_specs
                    .into_iter()
                    .map(|(rows, ..)| {
                        Some(RegisteredALPRDPage {
                            rows,
                            left: requests.next()?,
                            right: requests.next()?,
                        })
                    })
                    .collect::<Option<Vec<_>>>()?;
                let patch_buffers = patch_specs
                    .into_iter()
                    .map(|(_, descriptor)| Some((requests.next()?, descriptor)))
                    .collect::<Option<Vec<_>>>()?;
                RegisteredReadKind::Alprd {
                    pages,
                    patch_buffers,
                    plan: plan.as_ref().clone(),
                }
            }
        };

        Some(RegisteredPartialRead {
            array_tree: self.array_tree.clone(),
            kind,
        })
    }

    fn estimated_partial_io(
        &self,
        pages: &[Range<usize>],
        _layout_len: usize,
    ) -> Option<(usize, usize)> {
        match &self.kind {
            PartialReadKind::Fixed(buffers) => {
                let bytes = pages.iter().try_fold(0usize, |total, rows| {
                    buffers.iter().try_fold(total, |total, buffer| {
                        let granules = rows
                            .end
                            .div_ceil(buffer.row_granularity)
                            .checked_sub(rows.start / buffer.row_granularity)?;
                        total.checked_add(granules.checked_mul(buffer.bytes_per_granule)?)
                    })
                })?;
                Some((bytes, pages.len().checked_mul(buffers.len())?))
            }
            PartialReadKind::Alprd(plan) => {
                let values_per_row = usize::try_from(plan.list_size).ok()?;
                let page_bytes = pages.iter().try_fold(0usize, |total, rows| {
                    let values = rows.start.checked_mul(values_per_row)?
                        ..rows.end.checked_mul(values_per_row)?;
                    let left = bitpacked_range(&plan.left, values.clone())?;
                    let right = bitpacked_range(&plan.right, values)?;
                    total.checked_add(left.len())?.checked_add(right.len())
                })?;
                let patch_bytes = plan
                    .patch_buffers
                    .iter()
                    .try_fold(0usize, |total, buffer| {
                        total.checked_add(buffer.range().len())
                    })?;
                Some((
                    page_bytes.checked_add(patch_bytes)?,
                    pages
                        .len()
                        .checked_mul(2)?
                        .checked_add(plan.patch_buffers.len())?,
                ))
            }
        }
    }
}

impl RegisteredPartialRead {
    pub(super) async fn resolve(
        self,
        dtype: &DType,
        row_range: &Range<usize>,
        mask: &Mask,
        ctx: &ReadContext,
        session: &VortexSession,
    ) -> VortexResult<ArrayRef> {
        let chunks = match self.kind {
            RegisteredReadKind::Fixed { pages } => {
                resolve_fixed_pages(
                    self.array_tree,
                    pages,
                    PageResolveContext {
                        dtype,
                        row_range,
                        mask,
                        ctx,
                        session,
                    },
                )
                .await?
            }
            RegisteredReadKind::Alprd {
                pages,
                patch_buffers,
                plan,
            } => {
                resolve_alprd_pages(
                    pages,
                    patch_buffers,
                    plan,
                    dtype,
                    row_range,
                    mask,
                    ctx,
                    session,
                )
                .await?
            }
        };
        finish_chunks(chunks, dtype, session)
    }
}

async fn resolve_fixed_pages(
    array_tree: ByteBuffer,
    pages: Vec<RegisteredPage>,
    context: PageResolveContext<'_>,
) -> VortexResult<Vec<ArrayRef>> {
    let mut page_futures = Vec::new();
    for page in pages {
        let local_mask = page_mask(&page.rows, context.row_range, context.mask.indices())?;
        if local_mask.all_false() {
            continue;
        }
        let array_tree = array_tree.clone();
        let dtype = context.dtype.clone();
        let ctx = context.ctx.clone();
        let session = context.session.clone();
        page_futures.push(async move {
            let buffers = try_join_all(page.buffers.into_iter().map(
                |(future, descriptor)| async move {
                    future.await?.ensure_aligned(descriptor.alignment())
                },
            ))
            .await?;
            let array = SerializedArray::from_flatbuffer_with_buffers(array_tree, buffers)?
                .decode(&dtype, page.rows.len(), &ctx, &session)?;
            clear_stats(&array);
            apply_page_mask(array, local_mask)
        });
    }
    try_join_all(page_futures).await
}

#[allow(clippy::too_many_arguments)]
async fn resolve_alprd_pages(
    pages: Vec<RegisteredALPRDPage>,
    patch_requests: Vec<(SegmentFuture, SerializedBuffer)>,
    plan: ALPRDReadPlan,
    dtype: &DType,
    row_range: &Range<usize>,
    mask: &Mask,
    ctx: &ReadContext,
    session: &VortexSession,
) -> VortexResult<Vec<ArrayRef>> {
    let patch_handles = try_join_all(patch_requests.into_iter().map(
        |(future, descriptor)| async move {
            Ok::<_, vortex_error::VortexError>((
                descriptor.index(),
                future.await?.ensure_aligned(descriptor.alignment())?,
            ))
        },
    ));
    let page_handles = try_join_all(pages.into_iter().filter_map(|page| {
        let local_mask = match page_mask(&page.rows, row_range, mask.indices()) {
            Ok(local_mask) if !local_mask.all_false() => local_mask,
            Ok(_) => return None,
            Err(error) => return Some(futures::future::ready(Err(error)).left_future()),
        };
        let left_alignment = plan.left.descriptor.alignment();
        let right_alignment = plan.right.descriptor.alignment();
        Some(
            async move {
                let (left, right) = futures::try_join!(page.left, page.right)?;
                Ok::<_, vortex_error::VortexError>((
                    page.rows,
                    local_mask,
                    left.ensure_aligned(left_alignment)?,
                    right.ensure_aligned(right_alignment)?,
                ))
            }
            .right_future(),
        )
    }));

    // Every range for this Flat layout is registered before resolution. Poll the complete set
    // together so the driver can issue and coalesce patch, left-part, and right-part reads in one
    // I/O round; array reconstruction starts only after that set has resolved.
    let (patch_handles, page_handles) = futures::try_join!(patch_handles, page_handles)?;
    let mut handles = empty_handles(plan.descriptors.len());
    for (index, handle) in patch_handles {
        handles[index] = handle;
    }
    let serialized = plan.serialized.with_buffers(handles);
    let alprd = serialized.child(0);
    let patch_len = plan.patch_metadata.len()?;
    let patch_indices =
        alprd
            .child(2)
            .decode(&plan.patch_indices_dtype, patch_len, ctx, session)?;
    let patch_indices = patch_indices
        .execute::<PrimitiveArray>(&mut session.create_execution_ctx())?
        .into_array();
    let patch_values = alprd.child(3).decode(
        &plan.left_parts_dtype.as_nonnullable(),
        patch_len,
        ctx,
        session,
    )?;
    let full_inner_len = usize::try_from(plan.list_size)?
        .checked_mul(plan.row_count)
        .ok_or_else(|| vortex_err!("ALPRD inner length overflow"))?;
    let full_patches = Patches::new(
        full_inner_len,
        plan.patch_metadata.offset()?,
        patch_indices,
        patch_values,
        None,
    )?;

    page_handles
        .into_iter()
        .map(|(rows, local_mask, left, right)| {
            let inner_start = rows.start * plan.list_size as usize;
            let inner_end = rows.end * plan.list_size as usize;
            let inner_len = inner_end - inner_start;
            let left = BitPacked::try_new(
                left,
                plan.left.ptype,
                Validity::from(plan.left_parts_dtype.nullability()),
                None,
                plan.left.bit_width,
                inner_len,
                0,
            )?
            .into_array();
            let right = BitPacked::try_new(
                right,
                plan.right.ptype,
                Validity::NonNullable,
                None,
                plan.right.bit_width,
                inner_len,
                0,
            )?
            .into_array();
            let patches = full_patches.slice(inner_start..inner_end)?;
            let elements = ALPRD::try_new(
                plan.element_dtype.clone(),
                left,
                plan.left_parts_dictionary.clone(),
                right,
                plan.right_bit_width,
                patches,
            )?
            .into_array();
            let array = FixedSizeListArray::try_new(
                elements,
                plan.list_size,
                Validity::from(dtype.nullability()),
                rows.len(),
            )?
            .into_array();
            clear_stats(&array);
            apply_page_mask(array, local_mask)
        })
        .collect()
}

fn finish_chunks(
    mut chunks: Vec<ArrayRef>,
    dtype: &DType,
    session: &VortexSession,
) -> VortexResult<ArrayRef> {
    match chunks.len() {
        0 => Ok(Canonical::empty(dtype).into_array()),
        1 => Ok(chunks.remove(0)),
        _ => {
            let chunks = ChunkedArray::try_new(chunks, dtype.clone())?.into_array();
            let mut ctx = session.create_execution_ctx();
            Ok(chunks.execute::<Canonical>(&mut ctx)?.into_array())
        }
    }
}

fn apply_page_mask(array: ArrayRef, mask: Mask) -> VortexResult<ArrayRef> {
    if mask.all_true() {
        Ok(array)
    } else if let AllOr::Some([(start, end)]) = mask.slices() {
        array.slice(*start..*end)
    } else {
        array.filter(mask)
    }
}

fn clear_stats(array: &ArrayRef) {
    for child in array.depth_first_traversal() {
        child.statistics().clear_all();
    }
}

fn empty_handles(len: usize) -> Vec<BufferHandle> {
    (0..len)
        .map(|_| BufferHandle::new_host(ByteBuffer::empty()))
        .collect()
}

fn selected_pages(
    page_rows: usize,
    layout_len: usize,
    row_range: &Range<usize>,
    mask: &Mask,
) -> Option<Vec<Range<usize>>> {
    let mut page_indices = BTreeSet::new();
    match mask.slices() {
        AllOr::All => return None,
        AllOr::None => {}
        AllOr::Some(slices) => {
            for &(start, end) in slices {
                if start >= end {
                    continue;
                }
                let global_start = row_range.start.checked_add(start)?;
                let global_end = row_range.start.checked_add(end)?;
                if global_end > row_range.end || global_end > layout_len {
                    return None;
                }
                page_indices.extend(global_start / page_rows..=(global_end - 1) / page_rows);
            }
        }
    }
    Some(
        page_indices
            .into_iter()
            .map(|page_index| {
                let start = page_index * page_rows;
                start..start.saturating_add(page_rows).min(layout_len)
            })
            .collect(),
    )
}

fn selected_page_runs(
    pages: &[Range<usize>],
    row_range: &Range<usize>,
    mask: &Mask,
) -> Option<Vec<Range<usize>>> {
    let selected = mask.slices();
    let mut runs = Vec::new();
    for page in pages {
        match selected {
            AllOr::All => {
                let start = page.start.max(row_range.start);
                let end = page.end.min(row_range.end);
                if start < end {
                    runs.push(start..end);
                }
            }
            AllOr::None => {}
            AllOr::Some(slices) => {
                for &(start, end) in slices {
                    let global_start = row_range.start.checked_add(start)?;
                    let global_end = row_range.start.checked_add(end)?;
                    let run_start = page.start.max(global_start);
                    let run_end = page.end.min(global_end);
                    if run_start < run_end {
                        runs.push(run_start..run_end);
                    }
                }
            }
        }
    }
    Some(runs)
}

fn page_mask(
    page_rows: &Range<usize>,
    row_range: &Range<usize>,
    selected: AllOr<&[usize]>,
) -> VortexResult<Mask> {
    match selected {
        AllOr::None => Ok(Mask::new_false(page_rows.len())),
        AllOr::All => {
            let start = page_rows.start.max(row_range.start);
            let end = page_rows.end.min(row_range.end);
            Ok(Mask::from_indices(
                page_rows.len(),
                (start..end).map(|row| row - page_rows.start),
            ))
        }
        AllOr::Some(indices) => Ok(Mask::from_indices(
            page_rows.len(),
            indices.iter().filter_map(|&index| {
                let row = row_range.start.checked_add(index)?;
                page_rows.contains(&row).then(|| row - page_rows.start)
            }),
        )),
    }
}

fn try_alprd_plan(
    node: &SerializedArray,
    dtype: &DType,
    ctx: &ReadContext,
    row_count: usize,
    descriptors: Arc<[SerializedBuffer]>,
) -> VortexResult<Option<(ALPRDReadPlan, usize)>> {
    if ctx.resolve(node.encoding_id()) != Some(FixedSizeList.id())
        || node.nbuffers() != 0
        || node.nchildren() != 1
    {
        return Ok(None);
    }
    let DType::FixedSizeList(element_dtype, list_size, _) = dtype else {
        return Ok(None);
    };
    let list_size_usize = usize::try_from(*list_size)?;
    if list_size_usize == 0 || !list_size_usize.is_multiple_of(1024) {
        return Ok(None);
    }
    let DType::Primitive(element_ptype, element_nullability) = element_dtype.as_ref() else {
        return Ok(None);
    };
    if !matches!(
        element_ptype,
        vortex_array::dtype::PType::F32 | vortex_array::dtype::PType::F64
    ) {
        return Ok(None);
    }

    let alprd = node.child(0);
    if ctx
        .resolve(alprd.encoding_id())
        .is_none_or(|id| id.as_str() != "vortex.alprd")
        || alprd.nbuffers() != 0
        || alprd.nchildren() != 4
    {
        return Ok(None);
    }
    let metadata = ALPRDMetadata::decode(alprd.metadata())?;
    let Some(patch_metadata) = metadata.patches().copied() else {
        return Ok(None);
    };
    let left_parts_dtype = DType::Primitive(metadata.left_parts_ptype(), *element_nullability);
    let right_ptype = match element_ptype {
        vortex_array::dtype::PType::F32 => vortex_array::dtype::PType::U32,
        vortex_array::dtype::PType::F64 => vortex_array::dtype::PType::U64,
        _ => unreachable!(),
    };
    let inner_len = row_count
        .checked_mul(list_size_usize)
        .ok_or_else(|| vortex_err!("ALPRD inner length overflow"))?;
    let left = try_bitpacked_plan(
        &alprd.child(0),
        left_parts_dtype.as_ptype(),
        ctx,
        inner_len,
        &descriptors,
    )?;
    let right = try_bitpacked_plan(&alprd.child(1), right_ptype, ctx, inner_len, &descriptors)?;
    let (Some(left), Some(right)) = (left, right) else {
        return Ok(None);
    };

    let mut patch_indices = BTreeSet::new();
    collect_buffer_indices(&alprd.child(2), &mut patch_indices);
    collect_buffer_indices(&alprd.child(3), &mut patch_indices);
    let Some(patch_buffers) = patch_indices
        .into_iter()
        .map(|index| descriptors.get(index).cloned())
        .collect::<Option<Vec<_>>>()
    else {
        return Ok(None);
    };
    if patch_buffers.is_empty() {
        return Ok(None);
    }
    let expected_indices: BTreeSet<_> = [left.descriptor.index(), right.descriptor.index()]
        .into_iter()
        .chain(patch_buffers.iter().map(SerializedBuffer::index))
        .collect();
    if expected_indices.len() != descriptors.len()
        || expected_indices.iter().copied().ne(0..descriptors.len())
    {
        return Ok(None);
    }

    let blocks_per_row = list_size_usize / 1024;
    let bytes_per_row = blocks_per_row
        .checked_mul(128)
        .and_then(|value| value.checked_mul(left.bit_width as usize + right.bit_width as usize))
        .ok_or_else(|| vortex_err!("ALPRD row width overflow"))?;
    Ok(Some((
        ALPRDReadPlan {
            serialized: node.clone(),
            descriptors,
            left,
            right,
            patch_buffers: patch_buffers.into(),
            patch_metadata,
            patch_indices_dtype: patch_metadata.indices_dtype()?,
            left_parts_dtype,
            left_parts_dictionary: metadata.left_parts_dictionary()?,
            right_bit_width: metadata.right_bit_width()?,
            element_dtype: element_dtype.as_ref().clone(),
            list_size: *list_size,
            row_count,
        },
        bytes_per_row,
    )))
}

fn try_bitpacked_plan(
    node: &SerializedArray,
    ptype: vortex_array::dtype::PType,
    ctx: &ReadContext,
    len: usize,
    descriptors: &[SerializedBuffer],
) -> VortexResult<Option<BitPackedReadPlan>> {
    if ctx
        .resolve(node.encoding_id())
        .is_none_or(|id| id.as_str() != "fastlanes.bitpacked")
        || node.nchildren() != 0
        || node.buffer_indices().len() != 1
    {
        return Ok(None);
    }
    let metadata = BitPackedMetadata::decode(node.metadata())?;
    if metadata.patches().is_some() || metadata.offset()? != 0 {
        return Ok(None);
    }
    let bit_width = metadata.bit_width()?;
    let Some(descriptor) = descriptors.get(node.buffer_indices()[0]).cloned() else {
        return Ok(None);
    };
    let expected_len = len
        .div_ceil(1024)
        .checked_mul(128 * bit_width as usize)
        .ok_or_else(|| vortex_err!("Bit-packed buffer length overflow"))?;
    if descriptor.range().len() != expected_len {
        return Ok(None);
    }
    Ok(Some(BitPackedReadPlan {
        descriptor,
        ptype,
        bit_width,
        offset: 0,
    }))
}

fn bitpacked_range(plan: &BitPackedReadPlan, values: Range<usize>) -> Option<Range<usize>> {
    if plan.offset != 0 || !values.start.is_multiple_of(1024) || !values.end.is_multiple_of(1024) {
        return None;
    }
    let bytes_per_block = 128usize.checked_mul(plan.bit_width as usize)?;
    let start = plan
        .descriptor
        .range()
        .start
        .checked_add((values.start / 1024).checked_mul(bytes_per_block)?)?;
    let end = plan
        .descriptor
        .range()
        .start
        .checked_add((values.end / 1024).checked_mul(bytes_per_block)?)?;
    Some(start..end)
}

fn collect_buffer_indices(node: &SerializedArray, output: &mut BTreeSet<usize>) {
    output.extend(node.buffer_indices());
    for index in 0..node.nchildren() {
        collect_buffer_indices(&node.child(index), output);
    }
}

fn collect_raw_buffers(
    node: &SerializedArray,
    dtype: &DType,
    ctx: &ReadContext,
    row_multiplier: usize,
    root_row_count: usize,
    descriptors: &[SerializedBuffer],
    output: &mut Vec<PlannedBuffer>,
) -> VortexResult<bool> {
    let Some(id) = ctx.resolve(node.encoding_id()) else {
        return Ok(false);
    };

    if id == Primitive.id() {
        let DType::Primitive(ptype, _) = dtype else {
            return Ok(false);
        };
        if node.nchildren() != 0 || node.buffer_indices().len() != 1 {
            return Ok(false);
        }
        let index = node.buffer_indices()[0];
        let Some(descriptor) = descriptors.get(index) else {
            return Ok(false);
        };
        output.push(PlannedBuffer {
            descriptor: descriptor.clone(),
            bytes_per_row: row_multiplier
                .checked_mul(ptype.byte_width())
                .ok_or_else(|| vortex_err!("Partial primitive row width overflow"))?,
            row_granularity: 1,
            bytes_per_granule: row_multiplier
                .checked_mul(ptype.byte_width())
                .ok_or_else(|| vortex_err!("Partial primitive row width overflow"))?,
        });
        return Ok(true);
    }

    if id == FixedSizeList.id() {
        let DType::FixedSizeList(element_dtype, list_size, _) = dtype else {
            return Ok(false);
        };
        if node.nbuffers() != 0 || node.nchildren() != 1 {
            return Ok(false);
        }
        let multiplier = row_multiplier
            .checked_mul(*list_size as usize)
            .ok_or_else(|| vortex_err!("Partial fixed-size-list width overflow"))?;
        return collect_raw_buffers(
            &node.child(0),
            element_dtype,
            ctx,
            multiplier,
            root_row_count,
            descriptors,
            output,
        );
    }

    if id == Struct.id() {
        let DType::Struct(fields, _) = dtype else {
            return Ok(false);
        };
        if node.nbuffers() != 0 || node.nchildren() != fields.nfields() {
            return Ok(false);
        }
        for (index, field_dtype) in fields.fields().enumerate() {
            if !collect_raw_buffers(
                &node.child(index),
                &field_dtype,
                ctx,
                row_multiplier,
                root_row_count,
                descriptors,
                output,
            )? {
                return Ok(false);
            }
        }
        return Ok(true);
    }

    if id.as_str() == "vortex.alprd" {
        if !matches!(dtype, DType::Primitive(_, _))
            || node.nbuffers() != 0
            || node.nchildren() != 2
            || row_multiplier == 0
            || root_row_count == 0
        {
            return Ok(false);
        }
        let granularity = 1024 / gcd(1024, row_multiplier);
        for child_index in 0..2 {
            let child = node.child(child_index);
            let Some(child_id) = ctx.resolve(child.encoding_id()) else {
                return Ok(false);
            };
            if child_id.as_str() != "fastlanes.bitpacked"
                || child.nchildren() != 0
                || child.buffer_indices().len() != 1
            {
                return Ok(false);
            }
            let index = child.buffer_indices()[0];
            let Some(descriptor) = descriptors.get(index) else {
                return Ok(false);
            };
            let granules = root_row_count.div_ceil(granularity);
            if descriptor.range().len() % granules != 0 {
                return Ok(false);
            }
            let bytes_per_granule = descriptor.range().len() / granules;
            output.push(PlannedBuffer {
                descriptor: descriptor.clone(),
                bytes_per_row: bytes_per_granule.div_ceil(granularity),
                row_granularity: granularity,
                bytes_per_granule,
            });
        }
        return Ok(true);
    }

    Ok(false)
}

fn gcd(mut left: usize, mut right: usize) -> usize {
    while right != 0 {
        (left, right) = (right, left % right);
    }
    left
}

fn checked_lcm(left: usize, right: usize) -> VortexResult<usize> {
    left.checked_div(gcd(left, right))
        .and_then(|value| value.checked_mul(right))
        .ok_or_else(|| vortex_err!("Partial row granularity overflow"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selected_page_runs_are_exact_and_coalesced() {
        let pages = [10..14, 14..18];
        let mask = Mask::from_indices(8, [1, 2, 6]);

        assert_eq!(
            selected_page_runs(&pages, &(10..18), &mask),
            Some(vec![11..13, 16..17])
        );
    }
}
