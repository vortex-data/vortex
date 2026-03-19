// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::fmt::Debug;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;
use std::ops::Range;
use std::sync::Arc;

use futures::FutureExt;
use futures::future::BoxFuture;
use vortex_array::ArrayRef;
use vortex_array::ArrayVisitor;
use vortex_array::DynArray;
use vortex_array::buffer::BufferHandle;
use vortex_array::buffer::DeviceBuffer;
use vortex_array::serde::ArrayParts;
use vortex_buffer::Alignment;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;

use crate::segments::SegmentId;
use crate::segments::SegmentSource;

/// A lazy buffer handle that defers segment I/O until materialization.
///
/// Wraps a [`SegmentSource`] and [`SegmentId`] together with an optional byte
/// selection. Operations like [`slice`](Self::slice) and [`filter`](Self::filter)
/// accumulate without triggering I/O, allowing the system to determine the exact
/// byte ranges needed before reading.
#[derive(Clone)]
pub struct LazyBufferHandle {
    source: Arc<dyn SegmentSource>,
    segment_id: SegmentId,
    selection: Selection,
    alignment: Alignment,
}

/// Byte selection within a segment buffer.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
enum Selection {
    /// The entire segment is selected.
    All,
    /// A single contiguous byte range within the segment.
    Range(Range<usize>),
    /// Multiple non-overlapping, sorted byte ranges within the segment.
    Ranges(Arc<[Range<usize>]>),
}

#[allow(clippy::same_name_method)]
impl LazyBufferHandle {
    /// Create a new lazy handle selecting the entire segment.
    pub fn new(
        source: Arc<dyn SegmentSource>,
        segment_id: SegmentId,
        alignment: Alignment,
    ) -> Self {
        Self {
            source,
            segment_id,
            selection: Selection::All,
            alignment,
        }
    }

    /// Returns the length of the selected byte range(s).
    ///
    /// # Panics
    ///
    /// Panics if the entire segment is selected ([`Selection::All`]) since the
    /// length is not known without performing I/O.
    pub fn len(&self) -> usize {
        match &self.selection {
            Selection::All => {
                vortex_panic!("len() is not available for Selection::All; slice first")
            }
            Selection::Range(r) => r.len(),
            Selection::Ranges(rs) => rs.iter().map(|r| r.len()).sum(),
        }
    }

    /// Returns whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the alignment of the buffer.
    pub fn alignment(&self) -> Alignment {
        self.alignment
    }

    /// Returns the segment ID.
    pub fn segment_id(&self) -> SegmentId {
        self.segment_id
    }

    /// Returns the byte ranges that will be read from the segment, or `None` if the
    /// entire segment is selected.
    pub fn byte_ranges(&self) -> Option<&[Range<usize>]> {
        match &self.selection {
            Selection::All => None,
            Selection::Range(r) => Some(std::slice::from_ref(r)),
            Selection::Ranges(ranges) => Some(ranges),
        }
    }

    /// Narrow to a contiguous byte range within the current selection.
    ///
    /// The range is interpreted relative to the current selection's logical byte
    /// offsets (i.e., offsets into the bytes that would be produced by materializing
    /// the current selection).
    ///
    /// # Panics
    ///
    /// Panics if the range exceeds the bounds of the current selection (when
    /// those bounds are known).
    pub fn slice(&self, range: Range<usize>) -> Self {
        let selection = match &self.selection {
            Selection::All => Selection::Range(range),
            Selection::Range(base) => {
                let start = base.start + range.start;
                let end = base.start + range.end;
                assert!(
                    end <= base.end,
                    "slice range {}..{} exceeds current selection 0..{}",
                    range.start,
                    range.end,
                    base.len(),
                );
                Selection::Range(start..end)
            }
            Selection::Ranges(existing) => slice_into_ranges(existing, range),
        };
        Self {
            source: Arc::clone(&self.source),
            segment_id: self.segment_id,
            selection,
            alignment: self.alignment,
        }
    }

    /// Select multiple byte ranges within the current view.
    ///
    /// Ranges are interpreted relative to the current selection's logical byte
    /// offsets and must be sorted and non-overlapping.
    ///
    /// # Panics
    ///
    /// Panics if any range exceeds the bounds of the current selection (when
    /// those bounds are known).
    pub fn filter(&self, ranges: &[Range<usize>]) -> Self {
        let selection = match &self.selection {
            Selection::All => Selection::Ranges(Arc::from(ranges)),
            Selection::Range(base) => {
                let absolute: Arc<[Range<usize>]> = ranges
                    .iter()
                    .map(|r| {
                        let abs = (base.start + r.start)..(base.start + r.end);
                        assert!(
                            abs.end <= base.end,
                            "filter range {}..{} exceeds current selection 0..{}",
                            r.start,
                            r.end,
                            base.len(),
                        );
                        abs
                    })
                    .collect();
                Selection::Ranges(absolute)
            }
            Selection::Ranges(existing) => {
                // Each input range is relative to the concatenated output of
                // the existing ranges. Map them back to absolute segment offsets.
                let mut result = Vec::new();
                for r in ranges {
                    match slice_into_ranges(existing, r.clone()) {
                        Selection::All => unreachable!(),
                        Selection::Range(abs) => result.push(abs),
                        Selection::Ranges(abs) => result.extend_from_slice(&abs),
                    }
                }
                Selection::Ranges(result.into())
            }
        };
        Self {
            source: Arc::clone(&self.source),
            segment_id: self.segment_id,
            selection,
            alignment: self.alignment,
        }
    }

    /// Materialize the lazy buffer by performing I/O and applying the selection.
    ///
    /// # Errors
    ///
    /// Returns an error if the segment cannot be loaded or the selection cannot be
    /// applied.
    pub async fn materialize(&self) -> VortexResult<BufferHandle> {
        let buffer = self.source.request(self.segment_id).await?;
        match &self.selection {
            Selection::All => Ok(buffer),
            Selection::Range(range) => Ok(buffer.slice(range.clone())),
            Selection::Ranges(ranges) => buffer.filter(ranges),
        }
    }
}

impl Debug for LazyBufferHandle {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LazyBufferHandle")
            .field("segment_id", &self.segment_id)
            .field("selection", &self.selection)
            .field("alignment", &self.alignment)
            .finish()
    }
}

impl PartialEq for LazyBufferHandle {
    fn eq(&self, other: &Self) -> bool {
        self.segment_id == other.segment_id
            && self.selection == other.selection
            && self.alignment == other.alignment
    }
}

impl Eq for LazyBufferHandle {}

impl Hash for LazyBufferHandle {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.segment_id.hash(state);
        self.selection.hash(state);
        self.alignment.hash(state);
    }
}

impl DeviceBuffer for LazyBufferHandle {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn len(&self) -> usize {
        self.len()
    }

    fn alignment(&self) -> Alignment {
        self.alignment
    }

    fn copy_to_host_sync(&self, alignment: Alignment) -> VortexResult<ByteBuffer> {
        futures::executor::block_on(async {
            let handle = self.materialize().await?;
            Ok(handle.try_into_host_sync()?.aligned(alignment))
        })
    }

    fn copy_to_host(
        &self,
        alignment: Alignment,
    ) -> VortexResult<BoxFuture<'static, VortexResult<ByteBuffer>>> {
        let this = self.clone();
        Ok(async move {
            let handle = this.materialize().await?;
            Ok(handle.try_into_host_sync()?.aligned(alignment))
        }
        .boxed())
    }

    fn slice(&self, range: Range<usize>) -> Arc<dyn DeviceBuffer> {
        Arc::new(LazyBufferHandle::slice(self, range))
    }

    fn filter(&self, ranges: &[Range<usize>]) -> VortexResult<Arc<dyn DeviceBuffer>> {
        Ok(Arc::new(LazyBufferHandle::filter(self, ranges)))
    }

    fn aligned(self: Arc<Self>, alignment: Alignment) -> VortexResult<Arc<dyn DeviceBuffer>> {
        if self.alignment.is_aligned_to(alignment) {
            Ok(self)
        } else {
            Ok(Arc::new(LazyBufferHandle {
                source: Arc::clone(&self.source),
                segment_id: self.segment_id,
                selection: self.selection.clone(),
                alignment,
            }))
        }
    }
}

/// Build an [`ArrayParts`] with lazy device buffers that defer segment I/O.
///
/// Each buffer descriptor in the flatbuffer is turned into a [`LazyBufferHandle`]
/// that records the segment source, segment ID, byte range, and alignment but
/// does **not** perform any I/O. The returned [`ArrayParts`] can be decoded into
/// an array tree and manipulated (sliced, filtered, optimized) before the lazy
/// buffers are materialized with [`materialize_recursive`].
pub fn create_lazy_array_parts(
    array_tree: ByteBuffer,
    source: Arc<dyn SegmentSource>,
    segment_id: SegmentId,
) -> VortexResult<ArrayParts> {
    use flatbuffers::root;
    use vortex_flatbuffers::FlatBuffer;
    use vortex_flatbuffers::array as fba;

    let fb_aligned = FlatBuffer::align_from(array_tree.clone());
    let fb_array = root::<fba::Array>(fb_aligned.as_ref())?;

    let mut offset: usize = 0;
    let buffers: Vec<BufferHandle> = fb_array
        .buffers()
        .unwrap_or_default()
        .iter()
        .map(|fb_buf| {
            offset += fb_buf.padding() as usize;
            let buffer_len = fb_buf.length() as usize;
            let alignment = Alignment::from_exponent(fb_buf.alignment_exponent());

            let lazy = LazyBufferHandle::new(Arc::clone(&source), segment_id, alignment)
                .slice(offset..offset + buffer_len);

            offset += buffer_len;
            BufferHandle::new_device(Arc::new(lazy))
        })
        .collect();

    ArrayParts::from_flatbuffer_with_buffers(array_tree, buffers)
}

/// Recursively walk the array tree and materialize any [`LazyBufferHandle`]
/// device buffers by performing I/O, returning a new tree with host-resident
/// buffers.
pub async fn materialize_recursive(array: &ArrayRef) -> VortexResult<ArrayRef> {
    // 1. Recursively materialize children.
    let children = array.children();
    let mut new_children = Vec::with_capacity(children.len());
    let mut any_child_changed = false;
    for child in &children {
        let new_child = Box::pin(materialize_recursive(child)).await?;
        any_child_changed |= !Arc::ptr_eq(child, &new_child);
        new_children.push(new_child);
    }
    let current = if any_child_changed {
        array.with_children(new_children)?
    } else {
        array.clone()
    };

    // 2. Check for lazy device buffers.
    let handles = current.buffer_handles();
    let any_lazy = handles.iter().any(|h| {
        h.as_device_opt()
            .and_then(|d| d.as_any().downcast_ref::<LazyBufferHandle>())
            .is_some()
    });
    if !any_lazy {
        return Ok(current);
    }

    // 3. Materialize lazy buffers, ensuring proper alignment.
    let mut materialized = Vec::with_capacity(handles.len());
    for handle in &handles {
        if let Some(lazy) = handle
            .as_device_opt()
            .and_then(|d| d.as_any().downcast_ref::<LazyBufferHandle>())
        {
            let buf = lazy.materialize().await?;
            materialized.push(buf.ensure_aligned(lazy.alignment())?);
        } else {
            materialized.push(handle.clone());
        }
    }
    current.with_buffers(materialized)
}

/// Map a logical byte range into the given set of existing absolute ranges.
///
/// The `range` is interpreted as an offset into the concatenated output of
/// `existing`. The result contains the corresponding absolute segment byte
/// ranges.
///
/// # Example
///
/// Given `existing = [10..20, 30..50]` (30 logical bytes),
/// `slice_into_ranges(existing, 5..25)` returns `Ranges([15..20, 30..40])`.
fn slice_into_ranges(existing: &[Range<usize>], range: Range<usize>) -> Selection {
    let mut result = Vec::new();
    let mut offset: usize = 0;

    for er in existing {
        let er_len = er.len();
        let next_offset = offset + er_len;

        // Skip ranges entirely before the slice start.
        if next_offset <= range.start {
            offset = next_offset;
            continue;
        }

        // Stop once past the slice end.
        if offset >= range.end {
            break;
        }

        // Intersect [range.start, range.end) with the logical span [offset, next_offset)
        // and map back to absolute segment bytes.
        let rel_start = range.start.saturating_sub(offset);
        let rel_end = (range.end - offset).min(er_len);
        result.push((er.start + rel_start)..(er.start + rel_end));

        offset = next_offset;
    }

    match result.len() {
        0 => Selection::Ranges(Arc::from([])),
        1 => Selection::Range(result.remove(0)),
        _ => Selection::Ranges(result.into()),
    }
}

#[cfg(test)]
mod tests {
    use std::ops::Range;
    use std::sync::Arc;

    use futures::FutureExt;
    use vortex_array::buffer::BufferHandle;
    use vortex_buffer::Alignment;
    use vortex_buffer::ByteBuffer;
    use vortex_error::VortexResult;
    use vortex_io::runtime::single::block_on;

    use super::*;
    use crate::segments::SegmentFuture;
    use crate::segments::SegmentId;
    use crate::segments::SegmentSource;

    /// A trivial in-memory segment source for tests.
    struct SingleSegment(BufferHandle);

    impl SegmentSource for SingleSegment {
        fn request(&self, _id: SegmentId) -> SegmentFuture {
            let handle = self.0.clone();
            async move { Ok(handle) }.boxed()
        }
    }

    fn lazy(data: &[u8]) -> LazyBufferHandle {
        let buf = BufferHandle::new_host(ByteBuffer::copy_from(data));
        LazyBufferHandle::new(
            Arc::new(SingleSegment(buf)),
            SegmentId::from(0u32),
            Alignment::none(),
        )
    }

    #[test]
    fn materialize_all() -> VortexResult<()> {
        block_on(|_| async {
            let handle = lazy(&[1, 2, 3, 4, 5, 6]).materialize().await?;
            assert_eq!(handle.unwrap_host().as_slice(), &[1, 2, 3, 4, 5, 6]);
            Ok(())
        })
    }

    #[test]
    fn slice_single() -> VortexResult<()> {
        block_on(|_| async {
            let handle = lazy(&[1, 2, 3, 4, 5, 6]).slice(1..5).materialize().await?;
            assert_eq!(handle.unwrap_host().as_slice(), &[2, 3, 4, 5]);
            Ok(())
        })
    }

    #[test]
    fn slice_of_slice() -> VortexResult<()> {
        block_on(|_| async {
            let handle = lazy(&[1, 2, 3, 4, 5, 6])
                .slice(1..5)
                .slice(1..3)
                .materialize()
                .await?;
            assert_eq!(handle.unwrap_host().as_slice(), &[3, 4]);
            Ok(())
        })
    }

    #[test]
    fn filter_from_all() -> VortexResult<()> {
        block_on(|_| async {
            let handle = lazy(&[1, 2, 3, 4, 5, 6])
                .filter(&[0..2, 4..6])
                .materialize()
                .await?;
            assert_eq!(handle.unwrap_host().as_slice(), &[1, 2, 5, 6]);
            Ok(())
        })
    }

    #[test]
    fn filter_of_slice() -> VortexResult<()> {
        block_on(|_| async {
            let handle = lazy(&[1, 2, 3, 4, 5, 6])
                .slice(1..5)
                .filter(&[0..1, 2..4])
                .materialize()
                .await?;
            // slice(1..5) → [2, 3, 4, 5]
            // filter([0..1, 2..4]) → [2, 4, 5]
            assert_eq!(handle.unwrap_host().as_slice(), &[2, 4, 5]);
            Ok(())
        })
    }

    #[test]
    fn slice_of_filter() -> VortexResult<()> {
        block_on(|_| async {
            let handle = lazy(&[10, 20, 30, 40, 50, 60])
                .filter(&[0..2, 4..6])
                .slice(1..3)
                .materialize()
                .await?;
            // filter([0..2, 4..6]) selects [10, 20, 50, 60] (4 logical bytes)
            // slice(1..3) → logical bytes 1..3 → [20, 50]
            assert_eq!(handle.unwrap_host().as_slice(), &[20, 50]);
            Ok(())
        })
    }

    #[test]
    fn filter_of_filter() -> VortexResult<()> {
        block_on(|_| async {
            let handle = lazy(&[10, 20, 30, 40, 50, 60])
                .filter(&[0..2, 4..6])
                .filter(&[0..1, 3..4])
                .materialize()
                .await?;
            // First filter selects [10, 20, 50, 60] (logical bytes 0..4)
            // Second filter selects logical [0..1, 3..4] → [10, 60]
            assert_eq!(handle.unwrap_host().as_slice(), &[10, 60]);
            Ok(())
        })
    }

    #[test]
    fn byte_ranges_none_for_all() {
        let lazy = lazy(&[1, 2, 3]);
        assert!(lazy.byte_ranges().is_none());
    }

    #[test]
    fn byte_ranges_after_slice() {
        let lazy = lazy(&[1, 2, 3, 4, 5]).slice(1..4);
        let expected = [Range { start: 1, end: 4 }];
        assert_eq!(lazy.byte_ranges(), Some(expected.as_slice()));
    }

    #[test]
    fn byte_ranges_after_filter() {
        let lazy = lazy(&[1, 2, 3, 4, 5]).filter(&[0..2, 3..5]);
        let expected = [Range { start: 0, end: 2 }, Range { start: 3, end: 5 }];
        assert_eq!(lazy.byte_ranges(), Some(expected.as_slice()));
    }
}
