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
use vortex_array::dtype::PType;
use vortex_array::expr::stats::Stat;
use vortex_array::patches::Patches;
use vortex_array::patches::PatchesMetadata;
use vortex_array::serde::SerializedArray;
use vortex_array::serde::SerializedBuffer;
use vortex_array::validity::Validity;
use vortex_buffer::Alignment;
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
use crate::layouts::flat::Stripes;
use crate::layouts::flat::striped::StripeMap;
use crate::segments::SegmentFuture;
use crate::segments::SegmentId;
use crate::segments::SegmentSource;

#[derive(Clone)]
pub(super) struct PartialReadPlan {
    array_tree: SerializedArray,
    bytes_per_row: usize,
    row_granularity: usize,
    kind: PartialReadKind,
    /// The physical layout of a striped segment, whose reads are estimated in physical bytes.
    stripe_map: Option<Arc<StripeMap>>,
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
    /// The patches, when they are plain primitive buffers that can be read in part.
    patch_columns: Option<PatchColumns>,
    /// How a striped segment splits the patches by stripe, so pages read only their own.
    patch_stripes: Option<PatchStripes>,
}

/// Patches stored as one primitive buffer of indices and one of values.
#[derive(Clone)]
pub(super) struct PatchColumns {
    pub indices: SerializedBuffer,
    pub index_ptype: PType,
    pub values: SerializedBuffer,
    pub value_ptype: PType,
    /// The patch offset: index `i` patches value `i - offset`.
    pub offset: usize,
    /// Patched values per row.
    pub values_per_row: usize,
}

#[derive(Clone)]
struct PatchStripes {
    rows_per_stripe: usize,
    map: Arc<StripeMap>,
}

impl PatchStripes {
    /// The patches of the stripes overlapping `rows`.
    fn patches(&self, rows: &Range<usize>) -> Option<Range<usize>> {
        let starts = self.map.element_starts()?;
        let first = rows.start / self.rows_per_stripe;
        let end = rows
            .end
            .div_ceil(self.rows_per_stripe)
            .min(starts.len() - 1);
        Some(usize::try_from(*starts.get(first)?).ok()?..usize::try_from(starts[end]).ok()?)
    }
}

impl PatchColumns {
    /// The byte ranges of the indices and values of `patches`.
    fn ranges(&self, patches: &Range<usize>) -> (Range<usize>, Range<usize>) {
        let bytes = |buffer: &SerializedBuffer, width: usize| {
            buffer.range().start + patches.start * width..buffer.range().start + patches.end * width
        };
        (
            bytes(&self.indices, self.index_ptype.byte_width()),
            bytes(&self.values, self.value_ptype.byte_width()),
        )
    }
}

impl ALPRDReadPlan {
    /// The patch buffers and their stripes, when pages read only their own patches.
    fn page_patches(&self) -> Option<(&PatchColumns, &PatchStripes)> {
        Some((self.patch_columns.as_ref()?, self.patch_stripes.as_ref()?))
    }
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
    ptype: PType,
    bit_width: u8,
    offset: u16,
}

pub(super) struct RegisteredPartialRead {
    array_tree: SerializedArray,
    kind: RegisteredReadKind,
}

enum RegisteredReadKind {
    Fixed {
        pages: Vec<RegisteredPage>,
    },
    Alprd {
        pages: Vec<RegisteredALPRDPage>,
        patch_buffers: Vec<(SegmentFuture, SerializedBuffer)>,
        plan: Box<ALPRDReadPlan>,
    },
}

struct RegisteredALPRDPage {
    rows: Range<usize>,
    left: SegmentFuture,
    right: SegmentFuture,
    /// The indices and values of the page's own patches, when the segment stripes them.
    patches: Option<(SegmentFuture, SegmentFuture)>,
}

struct RegisteredPage {
    rows: Range<usize>,
    buffers: Vec<(SegmentFuture, SerializedBuffer)>,
}

/// A buffer whose bytes are proportional to rows: `bytes_per_granule` bytes for every
/// `rows_per_granule` rows.
pub(super) struct RowBuffer {
    pub range: Range<usize>,
    pub rows_per_granule: usize,
    pub bytes_per_granule: usize,
}

impl PartialReadPlan {
    /// The buffers that can be split into runs of rows.
    pub(super) fn row_buffers(&self) -> VortexResult<Vec<RowBuffer>> {
        match &self.kind {
            PartialReadKind::Fixed(buffers) => Ok(buffers
                .iter()
                .map(|buffer| RowBuffer {
                    range: buffer.descriptor.range().clone(),
                    rows_per_granule: buffer.row_granularity,
                    bytes_per_granule: buffer.bytes_per_granule,
                })
                .collect()),
            PartialReadKind::Alprd(plan) => {
                let list_size = usize::try_from(plan.list_size)?;
                let values_per_granule = checked_lcm(1024, list_size)?;
                Ok([&plan.left, &plan.right]
                    .into_iter()
                    .filter(|bitpacked| bitpacked.offset == 0)
                    .map(|bitpacked| RowBuffer {
                        range: bitpacked.descriptor.range().clone(),
                        rows_per_granule: values_per_granule / list_size,
                        bytes_per_granule: values_per_granule / 1024
                            * 128
                            * usize::from(bitpacked.bit_width),
                    })
                    .collect())
            }
        }
    }

    /// The patches, when they are plain primitive buffers that a writer can split by stripe.
    pub(super) fn patch_columns(&self) -> Option<&PatchColumns> {
        match &self.kind {
            PartialReadKind::Alprd(plan) => plan.patch_columns.as_ref(),
            PartialReadKind::Fixed(_) => None,
        }
    }

    pub(super) fn supports_mask(mask: &Mask) -> bool {
        !mask.all_false()
    }

    pub(super) fn try_new(layout: &FlatLayout) -> VortexResult<Option<Self>> {
        let Some(array_tree) = layout.array_tree().cloned() else {
            return Ok(None);
        };
        let serialized = SerializedArray::from_array_tree(array_tree)?;
        let descriptors: Arc<[SerializedBuffer]> = serialized.buffer_descriptors()?.into();
        let row_count = usize::try_from(layout.row_count())?;

        let stripe_map = layout.stripes().map(|stripes| Arc::clone(stripes.map()));
        if let Some((plan, bytes_per_row)) = try_alprd_plan(
            &serialized,
            layout.dtype(),
            layout.array_ctx(),
            row_count,
            Arc::clone(&descriptors),
            layout.stripes(),
        )? {
            return Ok(Some(Self {
                array_tree: serialized.clone(),
                bytes_per_row,
                row_granularity: 1,
                kind: PartialReadKind::Alprd(Box::new(plan)),
                stripe_map,
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
            array_tree: serialized.clone(),
            bytes_per_row,
            row_granularity,
            kind: PartialReadKind::Fixed(planned.into()),
            stripe_map,
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
        // Reads are rounded to I/O blocks per buffer rather than to row pages spanning every
        // buffer, so a single row costs a block or two of each buffer, not a page of each.
        let block = (preferred_read_size / 16).max(1);
        // Striped segments keep the bytes of a row together, so they read exactly those bytes.
        let request_block = if self.stripe_map.is_some() { 1 } else { block };
        // Runs of selected rows closer than a block of bytes are read and decoded together.
        let merge_gap_rows = block / self.bytes_per_row;
        let pages = selected_runs(
            self.row_granularity,
            merge_gap_rows,
            layout_len,
            row_range,
            mask,
        )?;
        let page_rows = pages.iter().map(Range::len).max().unwrap_or(0);
        // Each extra read costs a request, an allocation and a decode, while a whole segment read
        // usually coalesces with its neighbours into one, and cached bytes are cheap to copy.
        // Charge a page of bytes per extra read. Encodings that decode per row (ALP-RD,
        // bit-packing) also save decode work in proportion to the rows they skip, so they go
        // partial once the reads cost at most half the segment; uncompressed buffers decode for
        // free and need an eighth.
        let min_saving_factor =
            if matches!(self.kind, PartialReadKind::Alprd(_)) || self.row_granularity > 1 {
                2
            } else {
                8
            };
        let (partial_bytes, request_count) =
            self.estimated_partial_io(&pages, preferred_read_size / 4, block, segment_len)?;
        let partial_cost = partial_bytes.checked_add(
            request_count
                .saturating_sub(1)
                .checked_mul(preferred_read_size)?,
        )?;
        // The bytes the selected rows occupy, before rounding to pages and adding whole buffers:
        // a lower bound for what a read must fetch.
        let needed_bytes = mask.true_count().saturating_mul(self.bytes_per_row)
            + match &self.kind {
                PartialReadKind::Alprd(plan) if plan.patch_stripes.is_none() => {
                    plan.patch_buffers.iter().map(|b| b.range().len()).sum()
                }
                _ => 0,
            };
        if partial_cost.saturating_mul(min_saving_factor) > segment_len {
            tracing::trace!(
                layout_len,
                page_rows,
                page_count = pages.len(),
                partial_bytes,
                request_count,
                partial_cost,
                needed_bytes,
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
            needed_bytes,
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
                                Some((start..end, buffer.descriptor.clone()))
                            })
                            .collect::<Option<Vec<_>>>()?;
                        Some((rows, ranges))
                    })
                    .collect::<Option<Vec<_>>>()?;
                let requests = request_block_aligned(
                    source,
                    segment_id,
                    request_block,
                    segment_len,
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
                let page_patches = plan.page_patches();
                let page_specs = pages
                    .into_iter()
                    .map(|rows| {
                        let inner_start = rows.start.checked_mul(values_per_row)?;
                        let inner_end = rows.end.checked_mul(values_per_row)?;
                        let left = bitpacked_range(&plan.left, inner_start..inner_end)?;
                        let right = bitpacked_range(&plan.right, inner_start..inner_end)?;
                        let patches = match page_patches {
                            Some((columns, stripes)) => {
                                let patches = stripes.patches(&rows)?;
                                (!patches.is_empty()).then(|| columns.ranges(&patches))
                            }
                            None => None,
                        };
                        Some((rows, left, right, patches))
                    })
                    .collect::<Option<Vec<_>>>()?;
                let patch_specs = if page_patches.is_some() {
                    Vec::new()
                } else {
                    plan.patch_buffers
                        .iter()
                        .map(|descriptor| (descriptor.range().clone(), descriptor.clone()))
                        .collect::<Vec<_>>()
                };
                let ranges = page_specs
                    .iter()
                    .flat_map(|(_, left, right, patches)| {
                        [left.clone(), right.clone()].into_iter().chain(
                            patches
                                .iter()
                                .flat_map(|(indices, values)| [indices.clone(), values.clone()]),
                        )
                    })
                    .chain(patch_specs.iter().map(|(range, _)| range.clone()))
                    .collect();
                let mut requests =
                    request_block_aligned(source, segment_id, request_block, segment_len, ranges)
                        .into_iter();
                let pages = page_specs
                    .into_iter()
                    .map(|(rows, _, _, patches)| {
                        let left = requests.next()?;
                        let right = requests.next()?;
                        let patches = match patches {
                            Some(_) => Some((requests.next()?, requests.next()?)),
                            None => None,
                        };
                        Some(RegisteredALPRDPage {
                            rows,
                            left,
                            right,
                            patches,
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
                    plan: plan.clone(),
                }
            }
        };

        Some(RegisteredPartialRead {
            array_tree: self.array_tree.clone(),
            kind,
        })
    }

    /// Estimate the I/O of reading `pages` as `(bytes, reads)` after the file source coalesces
    /// ranges whose gap is at most `coalesce_distance`.
    fn estimated_partial_io(
        &self,
        pages: &[Range<usize>],
        coalesce_distance: usize,
        block: usize,
        segment_len: usize,
    ) -> Option<(usize, usize)> {
        let mut ranges = Vec::new();
        match &self.kind {
            PartialReadKind::Fixed(buffers) => {
                for rows in pages {
                    for buffer in buffers.iter() {
                        let base = buffer.descriptor.range().start;
                        ranges.push(
                            base.checked_add(
                                (rows.start / buffer.row_granularity)
                                    .checked_mul(buffer.bytes_per_granule)?,
                            )?
                                ..base.checked_add(
                                    rows.end
                                        .div_ceil(buffer.row_granularity)
                                        .checked_mul(buffer.bytes_per_granule)?,
                                )?,
                        );
                    }
                }
            }
            PartialReadKind::Alprd(plan) => {
                let values_per_row = usize::try_from(plan.list_size).ok()?;
                let page_patches = plan.page_patches();
                for rows in pages {
                    let values = rows.start.checked_mul(values_per_row)?
                        ..rows.end.checked_mul(values_per_row)?;
                    ranges.push(bitpacked_range(&plan.left, values.clone())?);
                    ranges.push(bitpacked_range(&plan.right, values)?);
                    if let Some((columns, stripes)) = page_patches {
                        let (indices, values) = columns.ranges(&stripes.patches(rows)?);
                        ranges.extend([indices, values]);
                    }
                }
                if page_patches.is_none() {
                    ranges.extend(plan.patch_buffers.iter().map(|b| b.range().clone()));
                }
            }
        }
        let mut ranges = match &self.stripe_map {
            // A striped segment reads the exact physical pieces of each range.
            Some(map) => ranges
                .into_iter()
                .filter(|range| !range.is_empty())
                .flat_map(|range| map.physical_ranges(range.start as u64..range.end as u64))
                .map(|range| usize::try_from(range.start).ok()..usize::try_from(range.end).ok())
                .map(|range| Some(range.start?..range.end?))
                .collect::<Option<Vec<_>>>()?,
            None => ranges
                .into_iter()
                .map(|range| block_aligned(range, block, segment_len))
                .collect::<Vec<_>>(),
        };
        ranges.sort_unstable_by_key(|range| range.start);

        let mut bytes = 0usize;
        let mut reads = 0usize;
        let mut current: Option<Range<usize>> = None;
        for range in ranges {
            match &mut current {
                Some(cur) if range.start <= cur.end.saturating_add(coalesce_distance) => {
                    cur.end = cur.end.max(range.end);
                }
                _ => {
                    if let Some(cur) = current.replace(range) {
                        bytes = bytes.checked_add(cur.len())?;
                        reads += 1;
                    }
                }
            }
        }
        if let Some(cur) = current {
            bytes = bytes.checked_add(cur.len())?;
            reads += 1;
        }
        Some((bytes, reads))
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
                    self.array_tree,
                    pages,
                    patch_buffers,
                    *plan,
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
    array_tree: SerializedArray,
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
            let array =
                array_tree
                    .with_buffers(buffers)
                    .decode(&dtype, page.rows.len(), &ctx, &session)?;
            clear_stats(&array);
            apply_page_mask(array, local_mask)
        });
    }
    try_join_all(page_futures).await
}

#[allow(clippy::too_many_arguments)]
async fn resolve_alprd_pages(
    array_tree: SerializedArray,
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
                let patches = async move {
                    match page.patches {
                        Some((indices, values)) => futures::try_join!(indices, values).map(Some),
                        None => Ok(None),
                    }
                };
                let (left, right, patches) = futures::try_join!(page.left, page.right, patches)?;
                Ok::<_, vortex_error::VortexError>((
                    page.rows,
                    local_mask,
                    left.ensure_aligned(left_alignment)?,
                    right.ensure_aligned(right_alignment)?,
                    patches,
                ))
            }
            .right_future(),
        )
    }));

    // Every range for this Flat layout is registered before resolution. Poll the complete set
    // together so the driver can issue and coalesce patch, left-part, and right-part reads in one
    // I/O round; array reconstruction starts only after that set has resolved.
    let (patch_handles, page_handles) = futures::try_join!(patch_handles, page_handles)?;
    let full_inner_len = usize::try_from(plan.list_size)?
        .checked_mul(plan.row_count)
        .ok_or_else(|| vortex_err!("ALPRD inner length overflow"))?;
    // Unless each page read its own patches, decode the segment's patches once for all pages.
    let full_patches = if patch_handles.is_empty() {
        None
    } else {
        let mut handles = empty_handles(plan.descriptors.len());
        for (index, handle) in patch_handles {
            handles[index] = handle;
        }
        let serialized = array_tree.with_buffers(handles);
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
        Some(Patches::new(
            full_inner_len,
            plan.patch_metadata.offset()?,
            patch_indices,
            patch_values,
            None,
        )?)
    };

    page_handles
        .into_iter()
        .map(|(rows, local_mask, left, right, page_patches)| {
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
            let patches = match (page_patches, &full_patches, &plan.patch_columns) {
                (Some((indices, values)), _, Some(columns)) => Patches::new(
                    full_inner_len,
                    columns.offset,
                    primitive(indices, columns.index_ptype)?,
                    primitive(values, columns.value_ptype)?,
                    None,
                )?
                .slice(inner_start..inner_end)?,
                (None, Some(full_patches), _) => full_patches.slice(inner_start..inner_end)?,
                _ => None,
            };
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

/// A non-nullable primitive array of `ptype` over the bytes in `handle`.
fn primitive(handle: BufferHandle, ptype: PType) -> VortexResult<ArrayRef> {
    let handle = handle.ensure_aligned(Alignment::new(ptype.byte_width()))?;
    Ok(PrimitiveArray::from_buffer_handle(handle, ptype, Validity::NonNullable).into_array())
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
        for stat in Stat::all() {
            child.statistics().clear(stat);
        }
    }
}

fn empty_handles(len: usize) -> Vec<BufferHandle> {
    (0..len)
        .map(|_| BufferHandle::new_host(ByteBuffer::empty()))
        .collect()
}

/// Expand `range` to whole `block`s, without passing the end of the segment.
fn block_aligned(range: Range<usize>, block: usize, segment_len: usize) -> Range<usize> {
    let start = range.start / block * block;
    let end = range
        .end
        .div_ceil(block)
        .saturating_mul(block)
        .min(segment_len)
        .max(range.end);
    start..end
}

/// Request `ranges` rounded out to whole I/O blocks, resolving each to its exact bytes.
fn request_block_aligned(
    source: &Arc<dyn SegmentSource>,
    segment_id: SegmentId,
    block: usize,
    segment_len: usize,
    ranges: Vec<Range<usize>>,
) -> Vec<SegmentFuture> {
    if block <= 1 {
        return source.request_ranges(
            segment_id,
            ranges
                .into_iter()
                .map(|range| range.start as u64..range.end as u64)
                .collect(),
        );
    }
    let wide = ranges
        .iter()
        .map(|range| block_aligned(range.clone(), block, segment_len))
        .collect::<Vec<_>>();
    source
        .request_ranges(
            segment_id,
            wide.iter()
                .map(|range| range.start as u64..range.end as u64)
                .collect(),
        )
        .into_iter()
        .zip(ranges)
        .zip(wide)
        .map(|((read, exact), wide)| {
            let start = exact.start - wide.start;
            let len = exact.len();
            async move { Ok(read.await?.slice(start..start + len)) }.boxed()
        })
        .collect()
}

/// The runs of rows to read: selected rows rounded out to the encoding's row granularity, with
/// runs fewer than `merge_gap_rows` apart merged.
fn selected_runs(
    granularity: usize,
    merge_gap_rows: usize,
    layout_len: usize,
    row_range: &Range<usize>,
    mask: &Mask,
) -> Option<Vec<Range<usize>>> {
    let all = [(0, mask.len())];
    let slices: &[(usize, usize)] = match mask.slices() {
        AllOr::None => &[],
        AllOr::All => &all,
        AllOr::Some(slices) => slices,
    };
    let mut runs: Vec<Range<usize>> = Vec::new();
    for &(start, end) in slices {
        if start >= end {
            continue;
        }
        let global_start = row_range.start.checked_add(start)?;
        let global_end = row_range.start.checked_add(end)?;
        if global_end > row_range.end || global_end > layout_len {
            return None;
        }
        let start = global_start / granularity * granularity;
        let end = global_end
            .div_ceil(granularity)
            .saturating_mul(granularity)
            .min(layout_len);
        match runs.last_mut() {
            Some(run) if start <= run.end.saturating_add(merge_gap_rows) => {
                run.end = run.end.max(end);
            }
            _ => runs.push(start..end),
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
    stripes: Option<&Stripes>,
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
    if !matches!(element_ptype, PType::F32 | PType::F64) {
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
        PType::F32 => PType::U32,
        PType::F64 => PType::U64,
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

    let patch_columns = match (
        primitive_buffer(&alprd.child(2), ctx, &descriptors),
        primitive_buffer(&alprd.child(3), ctx, &descriptors),
    ) {
        (Some(indices), Some(values)) => Some(PatchColumns {
            indices,
            index_ptype: patch_metadata.indices_dtype()?.as_ptype(),
            values,
            value_ptype: left_parts_dtype.as_ptype(),
            offset: patch_metadata.offset()?,
            values_per_row: list_size_usize,
        }),
        _ => None,
    };
    let patch_stripes = patch_columns
        .as_ref()
        .and_then(|columns| patch_stripes(columns, stripes));

    let blocks_per_row = list_size_usize / 1024;
    let bytes_per_row = blocks_per_row
        .checked_mul(128)
        .and_then(|value| value.checked_mul(left.bit_width as usize + right.bit_width as usize))
        .ok_or_else(|| vortex_err!("ALPRD row width overflow"))?;
    Ok(Some((
        ALPRDReadPlan {
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
            patch_columns,
            patch_stripes,
        },
        bytes_per_row,
    )))
}

/// The buffer of a primitive array node without children.
fn primitive_buffer(
    node: &SerializedArray,
    ctx: &ReadContext,
    descriptors: &[SerializedBuffer],
) -> Option<SerializedBuffer> {
    if ctx.resolve(node.encoding_id())? != Primitive.id()
        || node.nchildren() != 0
        || node.buffer_indices().len() != 1
    {
        return None;
    }
    descriptors.get(node.buffer_indices()[0]).cloned()
}

/// How `stripes` split `columns`, when they split the indices and values by element.
fn patch_stripes(columns: &PatchColumns, stripes: Option<&Stripes>) -> Option<PatchStripes> {
    let stripes = stripes?;
    let rows_per_stripe = usize::try_from(stripes.rows_per_stripe()?).ok()?;
    let map = stripes.map();
    map.element_starts()?;
    let split = |buffer: &SerializedBuffer, ptype: PType| {
        stripes.buffers.iter().any(|striped| {
            striped.offset == buffer.range().start as u64
                && striped.bytes_per_element == ptype.byte_width() as u64
        })
    };
    (rows_per_stripe > 0
        && split(&columns.indices, columns.index_ptype)
        && split(&columns.values, columns.value_ptype))
    .then(|| PatchStripes {
        rows_per_stripe,
        map: Arc::clone(map),
    })
}

fn try_bitpacked_plan(
    node: &SerializedArray,
    ptype: PType,
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

pub(super) fn checked_lcm(left: usize, right: usize) -> VortexResult<usize> {
    left.checked_div(gcd(left, right))
        .and_then(|value| value.checked_mul(right))
        .ok_or_else(|| vortex_err!("Partial row granularity overflow"))
}
