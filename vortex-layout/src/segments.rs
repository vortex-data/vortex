use std::collections::BTreeMap;
use std::fmt::Display;
use std::ops::{Deref, Range, RangeBounds};
use std::sync::{Arc, RwLock};
use std::task::Poll;

use async_trait::async_trait;
use futures::channel::mpsc;
use futures::{SinkExt, Stream, StreamExt};
use range_union_find::RangeUnionFind;
use vortex_buffer::{Buffer, ByteBuffer};
use vortex_error::{VortexExpect, VortexResult, vortex_err};

use crate::range_intersection;

/// The identifier for a single segment.
// TODO(ngates): should this be a `[u8]` instead? Allowing for arbitrary segment identifiers?
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SegmentId(u32);

impl From<u32> for SegmentId {
    fn from(value: u32) -> Self {
        Self(value)
    }
}

impl Deref for SegmentId {
    type Target = u32;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Display for SegmentId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SegmentId({})", self.0)
    }
}

#[async_trait]
pub trait AsyncSegmentReader: 'static + Send + Sync {
    /// Attempt to get the data associated with a given segment ID.
    async fn get(&self, id: SegmentId) -> VortexResult<ByteBuffer>;
}

pub trait SegmentWriter {
    /// Write the given data into a segment and return its identifier.
    /// The provided buffers are concatenated together to form the segment.
    ///
    // TODO(ngates): in order to support aligned Direct I/O, it is preferable for all segments to
    //  be aligned to the logical block size (typically 512, but could be 4096). For this reason,
    //  if we know we're going to read an entire FlatLayout together, then we should probably
    //  serialize it into a single segment that is 512 byte aligned? Or else, we should guarantee
    //  to align the the first segment to 512, and then assume that coalescing captures the rest.
    fn put(&mut self, buffer: &[ByteBuffer]) -> SegmentId;
}

#[derive(Debug, Default, PartialEq, Eq, Hash, Clone, Copy, PartialOrd, Ord)]
pub enum RequiredSegmentKind {
    PRUNING = 1,
    FILTER = 2,
    #[default]
    PROJECTION = 3,
}

#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy, PartialOrd, Ord)]
pub struct SegmentPriority {
    row_end: u64, // sort by row_end first
    kind: RequiredSegmentKind,
    row_start: u64,
}

type SegmentStore = BTreeMap<SegmentPriority, Vec<SegmentId>>;

#[derive(Default)]
pub struct SegmentCollector {
    store: Arc<RwLock<SegmentStore>>,
    pub kind: RequiredSegmentKind,
}

impl SegmentCollector {
    pub fn with_priority_hint(&self, kind: RequiredSegmentKind) -> Self {
        SegmentCollector {
            store: self.store.clone(),
            // highest priority wins
            kind: kind.min(self.kind),
        }
    }

    pub fn push(&mut self, row_start: u64, row_end: u64, segment: SegmentId) {
        let (start, end) = match self.kind {
            // row offset inside the stats table is not our concern
            RequiredSegmentKind::PRUNING => (0, 0),
            _ => (row_start, row_end),
        };
        let priority = SegmentPriority {
            row_start: start,
            row_end: end,
            kind: self.kind,
        };
        self.store
            .write()
            .vortex_expect("poisoned lock")
            .entry(priority)
            .or_default()
            .push(segment);
    }

    pub fn finish(self) -> (RowRangePruner, SegmentStream) {
        let (cancellations_tx, cancellations_rx) = mpsc::unbounded();
        (
            RowRangePruner {
                store: self.store.clone(),
                cancellations_tx,
                excluded_ranges: Default::default(),
            },
            SegmentStream {
                store: self.store,
                cancellations_rx,
                current_key: TOP_PRIORITY,
                current_idx: 0,
            },
        )
    }
}

#[derive(Debug, Clone)]
pub struct RowRangePruner {
    store: Arc<RwLock<SegmentStore>>,
    cancellations_tx: mpsc::UnboundedSender<SegmentId>,
    excluded_ranges: Arc<RwLock<RangeUnionFind<u64>>>,
}

impl RowRangePruner {
    // Remove all segments fully encompassed by the given row range. Removals
    // of each matching segment is notified to the cancellation channel.
    pub async fn remove(&mut self, to_exclude: Range<u64>) -> VortexResult<()> {
        let to_exclude = {
            let mut excluded_ranges = self.excluded_ranges.write().vortex_expect("poisoned lock");
            excluded_ranges
                .insert_range(&to_exclude)
                .map_err(|e| vortex_err!("invalid range: {e}"))?;
            excluded_ranges
                .find_range_with_element(&to_exclude.start)
                .map_err(|_| vortex_err!("can not find range just inserted"))?
        };

        let cancelled_segments: Vec<_> = {
            let mut store = self.store.write().vortex_expect("poisoned lock");
            let to_remove: Vec<_> = store
                .keys()
                .skip_while(|key| !to_exclude.contains(&key.row_start))
                .take_while(|key| to_exclude.contains(&key.row_end))
                .copied()
                .collect();
            to_remove
                .iter()
                .flat_map(|key| store.remove(key).unwrap_or_default())
                .collect()
        };
        for id in cancelled_segments {
            self.cancellations_tx
                .send(id)
                .await
                .map_err(|_| vortex_err!("channel closed"))?;
        }
        Ok(())
    }

    /// Bulk remove row_indices. It is intended to be used for
    /// pruning row indices known to be excluded before the scan.
    /// It does not notify the cancellation channel.
    pub fn retain_matching(&mut self, row_indices: Buffer<u64>) {
        if row_indices.is_empty() {
            return;
        }
        self.store
            .write()
            .vortex_expect("poisoned lock")
            .retain(|key, _| {
                if key.kind == RequiredSegmentKind::PRUNING {
                    return true; // keep segments required for pruning
                }
                range_intersection(&(key.row_start..key.row_end), &row_indices).is_some()
            });
    }
}

pub struct SegmentStream {
    store: Arc<RwLock<SegmentStore>>,
    cancellations_rx: mpsc::UnboundedReceiver<SegmentId>,
    current_key: SegmentPriority,
    current_idx: usize,
}

pub enum SegmentEvent {
    Cancel(SegmentId),
    Request(SegmentId),
}

impl Stream for SegmentStream {
    type Item = SegmentEvent;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        // cancellations take priority over the next segment in store
        let channel_closed = match self.cancellations_rx.poll_next_unpin(cx) {
            Poll::Ready(Some(segment)) => return Poll::Ready(Some(SegmentEvent::Cancel(segment))),
            Poll::Ready(None) => true,
            Poll::Pending => false,
        };

        {
            let store_clone = self.store.clone();
            let store_guard = store_clone.read().vortex_expect("poisoned lock");
            let store_iter = store_guard.range(self.current_key..);
            for (&key, segments) in store_iter {
                match key == self.current_key {
                    true if self.current_idx >= segments.len() => continue,
                    false => {
                        self.current_idx = 0;
                        self.current_key = key;
                    }
                    _ => {}
                };
                let segment_to_yield = segments[self.current_idx];
                self.current_idx += 1;
                return Poll::Ready(Some(SegmentEvent::Request(segment_to_yield)));
            }
        }
        // store is exhausted if we are here
        if channel_closed {
            return Poll::Ready(None);
        }
        match self.cancellations_rx.poll_next_unpin(cx) {
            Poll::Ready(Some(segment)) => Poll::Ready(Some(SegmentEvent::Cancel(segment))),
            Poll::Ready(None) => Poll::Ready(None), // channel closed, end stream
            Poll::Pending => Poll::Pending,
        }
    }
}

#[cfg(test)]
pub mod test {
    use vortex_buffer::ByteBufferMut;
    use vortex_error::{VortexExpect, vortex_err};

    use super::*;

    #[derive(Default)]
    pub struct TestSegments {
        segments: Vec<ByteBuffer>,
    }

    impl SegmentWriter for TestSegments {
        fn put(&mut self, data: &[ByteBuffer]) -> SegmentId {
            let id = u32::try_from(self.segments.len())
                .vortex_expect("Cannot store more than u32::MAX segments");

            // Combine all the buffers since we're only a test implementation
            let mut buffer = ByteBufferMut::empty();
            for segment in data {
                buffer.extend_from_slice(segment.as_ref());
            }
            self.segments.push(buffer.freeze());

            id.into()
        }
    }

    #[async_trait]
    impl AsyncSegmentReader for TestSegments {
        async fn get(&self, id: SegmentId) -> VortexResult<ByteBuffer> {
            self.segments
                .get(*id as usize)
                .cloned()
                .ok_or_else(|| vortex_err!("Segment not found"))
        }
    }
}
