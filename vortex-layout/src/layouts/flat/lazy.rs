// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Encoding-agnostic partial reads of a flat segment through lazy buffers.
//!
//! The array is decoded over [`LazySegmentBuffer`]s, which record where a buffer lives in the
//! segment but read nothing. Slicing the array to each run of selected rows pushes the slice into
//! the buffers through each encoding's own slice rules, narrowing their byte ranges. Only those
//! narrowed ranges are then read, and the tree is rebuilt over the host buffers.

use std::any::Any;
use std::fmt::Debug;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;

use futures::FutureExt;
use futures::future::BoxFuture;
use futures::future::try_join_all;
use parking_lot::Mutex;
use vortex_array::ArrayRef;
use vortex_array::ArraySlots;
use vortex_array::IntoArray;
use vortex_array::arrays::ChunkedArray;
use vortex_array::buffer::BufferHandle;
use vortex_array::buffer::DeviceBuffer;
use vortex_array::dtype::DType;
use vortex_array::serde::SerializedArray;
use vortex_array::serde::SerializedBuffer;
use vortex_buffer::Alignment;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_mask::AllOr;
use vortex_mask::Mask;
use vortex_session::VortexSession;
use vortex_session::registry::ReadContext;

use crate::segments::SegmentId;
use crate::segments::SegmentSource;

/// A segment buffer that has not been read yet.
///
/// Slicing narrows the byte range without I/O. Host access fails rather than blocking, so a
/// lazy read that needs bytes it has not materialized falls back to reading the segment.
#[derive(Clone)]
pub(super) struct LazySegmentBuffer {
    source: Arc<dyn SegmentSource>,
    segment_id: SegmentId,
    range: Range<usize>,
    alignment: Alignment,
}

impl Debug for LazySegmentBuffer {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LazySegmentBuffer")
            .field("segment_id", &self.segment_id)
            .field("range", &self.range)
            .field("alignment", &self.alignment)
            .finish()
    }
}

impl PartialEq for LazySegmentBuffer {
    fn eq(&self, other: &Self) -> bool {
        self.segment_id == other.segment_id
            && self.range == other.range
            && self.alignment == other.alignment
    }
}

impl Eq for LazySegmentBuffer {}

impl Hash for LazySegmentBuffer {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.segment_id.hash(state);
        self.range.hash(state);
        self.alignment.hash(state);
    }
}

impl DeviceBuffer for LazySegmentBuffer {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn len(&self) -> usize {
        self.range.len()
    }

    fn alignment(&self) -> Alignment {
        self.alignment
    }

    fn copy_to_host_sync(&self, _alignment: Alignment) -> VortexResult<ByteBuffer> {
        Err(vortex_err!(
            "Lazy segment buffer {:?} must be materialized before host access",
            self
        ))
    }

    fn copy_to_host(
        &self,
        alignment: Alignment,
    ) -> VortexResult<BoxFuture<'static, VortexResult<ByteBuffer>>> {
        let read = self.source.request_range(
            self.segment_id,
            self.range.start as u64..self.range.end as u64,
        );
        Ok(async move { read.await?.ensure_aligned(alignment)?.try_into_host_sync() }.boxed())
    }

    fn slice(&self, range: Range<usize>) -> Arc<dyn DeviceBuffer> {
        Arc::new(Self {
            source: Arc::clone(&self.source),
            segment_id: self.segment_id,
            range: self.range.start + range.start..self.range.start + range.end,
            alignment: self.alignment,
        })
    }

    fn aligned(self: Arc<Self>, alignment: Alignment) -> VortexResult<Arc<dyn DeviceBuffer>> {
        // Alignment is applied when the bytes are materialized.
        Ok(Arc::new(Self {
            alignment,
            ..self.as_ref().clone()
        }))
    }
}

fn as_lazy(handle: &BufferHandle) -> Option<&LazySegmentBuffer> {
    handle
        .as_device_opt()
        .and_then(|device| device.as_any().downcast_ref::<LazySegmentBuffer>())
}

/// Collect the lazy buffers of `array` depth first: slots before the node's own buffers.
fn collect_lazy(array: &ArrayRef, out: &mut Vec<LazySegmentBuffer>) {
    for child in array.slots().iter().flatten() {
        collect_lazy(child, out);
    }
    out.extend(array.buffer_handles().iter().filter_map(as_lazy).cloned());
}

/// Rebuild `array`, replacing its lazy buffers in [`collect_lazy`] order.
fn rebuild(
    array: &ArrayRef,
    materialized: &mut impl Iterator<Item = BufferHandle>,
) -> VortexResult<ArrayRef> {
    let slots = array.slots();
    let new_slots = slots
        .iter()
        .map(|slot| {
            slot.as_ref()
                .map(|child| rebuild(child, materialized))
                .transpose()
        })
        .collect::<VortexResult<ArraySlots>>()?;
    let slots_changed = slots
        .iter()
        .zip(new_slots.iter())
        .any(|(old, new)| match (old, new) {
            (Some(old), Some(new)) => !ArrayRef::ptr_eq(old, new),
            _ => false,
        });
    let mut current = if slots_changed {
        // SAFETY: each child is replaced by the same logical values over host buffers.
        unsafe { array.clone().with_slots(new_slots)? }
    } else {
        array.clone()
    };

    let handles = current.buffer_handles();
    if handles.iter().any(|handle| as_lazy(handle).is_some()) {
        let handles = handles
            .into_iter()
            .map(|handle| match as_lazy(&handle) {
                Some(_) => materialized
                    .next()
                    .ok_or_else(|| vortex_err!("Lazy buffer was not materialized")),
                None => Ok(handle),
            })
            .collect::<VortexResult<Vec<_>>>()?;
        // SAFETY: each lazy buffer is replaced by the bytes it describes.
        current = unsafe { current.with_buffers(handles)? };
    }
    Ok(current)
}

/// Small buffers read up front, by their index in the array tree.
type EagerBuffers = Arc<[(usize, BufferHandle)]>;

/// What a flat reader learns about its segment across lazy reads.
#[derive(Default)]
pub(super) struct LazyState {
    /// The small buffers read up front, kept so later reads skip that I/O round.
    eager: Mutex<Option<EagerBuffers>>,
    /// Set once slicing is seen not to narrow this encoding's buffers.
    disabled: AtomicBool,
}

impl LazyState {
    pub fn is_disabled(&self) -> bool {
        self.disabled.load(Ordering::Relaxed)
    }
}

/// Everything a flat reader needs to read its segment lazily.
pub(super) struct LazyRead<'a> {
    pub template: &'a SerializedArray,
    pub descriptors: &'a [SerializedBuffer],
    pub source: &'a Arc<dyn SegmentSource>,
    pub segment_id: SegmentId,
    pub segment_len: usize,
    /// The source's preferred read size. Buffers up to a quarter of it are read up front, so
    /// slicing code that inspects small auxiliary buffers (patches, dictionaries, run ends) never
    /// touches a lazy buffer; reading them costs about as much as the gap the reads coalesce.
    pub page: usize,
    pub dtype: &'a DType,
    pub row_count: usize,
    pub ctx: &'a ReadContext,
    pub session: &'a VortexSession,
    pub state: &'a LazyState,
}

impl LazyRead<'_> {
    /// Read the rows of `row_range` selected by `mask`, or `None` when reading the whole segment
    /// would be cheaper.
    pub async fn read(
        &self,
        row_range: &Range<usize>,
        mask: &Mask,
    ) -> VortexResult<Option<ArrayRef>> {
        let all = [(0, mask.len())];
        let runs: &[(usize, usize)] = match mask.slices() {
            AllOr::None => return Ok(None),
            AllOr::All => &all,
            AllOr::Some(slices) => slices,
        };

        let (eager, lazy): (Vec<_>, Vec<_>) = self
            .descriptors
            .iter()
            .partition(|descriptor| descriptor.range().len() <= self.page / 4);
        let eager_bytes: usize = eager.iter().map(|d| d.range().len()).sum();
        if lazy.is_empty() || eager_bytes.saturating_mul(2) > self.segment_len {
            return Ok(None);
        }

        let cached = self.state.eager.lock().clone();
        let eager_buffers = match cached {
            Some(eager_buffers) => eager_buffers,
            None => {
                let eager_reads = self.source.request_ranges(
                    self.segment_id,
                    eager
                        .iter()
                        .map(|d| d.range().start as u64..d.range().end as u64)
                        .collect(),
                );
                let eager_buffers: EagerBuffers = try_join_all(eager.iter().zip(eager_reads).map(
                    |(descriptor, read)| async move {
                        Ok::<_, vortex_error::VortexError>((
                            descriptor.index(),
                            read.await?.ensure_aligned(descriptor.alignment())?,
                        ))
                    },
                ))
                .await?
                .into();
                *self.state.eager.lock() = Some(Arc::clone(&eager_buffers));
                eager_buffers
            }
        };

        let mut buffers = self
            .descriptors
            .iter()
            .map(|descriptor| {
                BufferHandle::new_device(Arc::new(LazySegmentBuffer {
                    source: Arc::clone(self.source),
                    segment_id: self.segment_id,
                    range: descriptor.range().clone(),
                    alignment: descriptor.alignment(),
                }))
            })
            .collect::<Vec<_>>();
        for (index, buffer) in eager_buffers.iter() {
            buffers[*index] = buffer.clone();
        }

        let array = self.template.with_buffers(buffers).decode(
            self.dtype,
            self.row_count,
            self.ctx,
            self.session,
        )?;

        // Slice each run of selected rows; the encodings push the slice into their buffers.
        let lazy_total: usize = lazy.iter().map(|d| d.range().len()).sum();
        let mut sliced = Vec::with_capacity(runs.len());
        let mut lazy_buffers = Vec::new();
        let mut lazy_bytes = 0usize;
        for &(start, end) in runs {
            let run = array.slice(row_range.start + start..row_range.start + end)?;
            let collected = lazy_buffers.len();
            collect_lazy(&run, &mut lazy_buffers);
            let run_bytes: usize = lazy_buffers[collected..]
                .iter()
                .map(|buffer| buffer.range.len())
                .sum();
            // A short run that still needs nearly every lazily held byte means this encoding
            // does not push slices into its buffers, so stop trying for this segment.
            if (end - start).saturating_mul(4) < self.row_count
                && run_bytes.saturating_mul(10) >= lazy_total.saturating_mul(9)
            {
                self.state.disabled.store(true, Ordering::Relaxed);
                return Ok(None);
            }
            lazy_bytes += run_bytes;
            // Only read partially when it costs at most half the segment.
            if (eager_bytes + lazy_bytes).saturating_mul(2) > self.segment_len {
                tracing::trace!(
                    runs = runs.len(),
                    eager_bytes,
                    lazy_bytes,
                    segment_len = self.segment_len,
                    "Flat lazy read rejected by I/O cost"
                );
                return Ok(None);
            }
            sliced.push(run);
        }
        tracing::trace!(
            runs = runs.len(),
            eager_bytes,
            lazy_bytes,
            segment_len = self.segment_len,
            "Flat lazy read registered"
        );

        let reads = self.source.request_ranges(
            self.segment_id,
            lazy_buffers
                .iter()
                .map(|buffer| buffer.range.start as u64..buffer.range.end as u64)
                .collect(),
        );
        let materialized = try_join_all(
            lazy_buffers
                .iter()
                .zip(reads)
                .map(|(buffer, read)| async move { read.await?.ensure_aligned(buffer.alignment) }),
        )
        .await?;

        let mut materialized = materialized.into_iter();
        let chunks = sliced
            .iter()
            .map(|run| rebuild(run, &mut materialized))
            .collect::<VortexResult<Vec<_>>>()?;
        Ok(Some(if chunks.len() == 1 {
            chunks
                .into_iter()
                .next()
                .ok_or_else(|| vortex_err!("one chunk"))?
        } else {
            ChunkedArray::try_new(chunks, self.dtype.clone())?.into_array()
        }))
    }
}
