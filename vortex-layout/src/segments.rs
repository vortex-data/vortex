use std::collections::BTreeMap;
use std::fmt::Display;
use std::ops::{Bound, Deref, Range, RangeBounds};
use std::sync::{Arc, RwLock};
use std::task::Poll;

use async_trait::async_trait;
use futures::channel::mpsc;
use futures::{SinkExt, Stream, StreamExt};
use range_union_find::RangeUnionFind;
use vortex_buffer::{Buffer, ByteBuffer};
use vortex_error::{VortexExpect, VortexResult, vortex_err};
use vortex_metrics::VortexMetrics;

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

impl SegmentPriority {
    fn new(row_start: u64, row_end: u64, kind: RequiredSegmentKind) -> Self {
        SegmentPriority {
            row_end,
            kind,
            row_start,
        }
    }
}

const TOP_PRIORITY: SegmentPriority = SegmentPriority {
    row_end: 0,
    kind: RequiredSegmentKind::PRUNING,
    row_start: 0,
};

type SegmentStore = BTreeMap<SegmentPriority, Vec<SegmentId>>;

#[derive(Default)]
pub struct SegmentCollector {
    store: Arc<RwLock<SegmentStore>>,
    pub kind: RequiredSegmentKind,
    metrics: VortexMetrics,
}

impl SegmentCollector {
    pub fn new(metrics: VortexMetrics) -> Self {
        Self {
            metrics,
            ..Default::default()
        }
    }

    pub fn with_priority_hint(&self, kind: RequiredSegmentKind) -> Self {
        SegmentCollector {
            store: self.store.clone(),
            // highest priority wins
            kind: kind.min(self.kind),
            metrics: self.metrics.clone(),
        }
    }

    pub fn push(&mut self, row_start: u64, row_end: u64, segment: SegmentId) {
        let (start, end) = match self.kind {
            // row offset inside the stats table is not our concern
            RequiredSegmentKind::PRUNING => (0, 0),
            _ => (row_start, row_end),
        };
        self.increment_metrics();
        let priority = SegmentPriority::new(start, end, self.kind);
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
                metrics: self.metrics.clone(),
            },
            SegmentStream {
                store: self.store,
                cancellations_rx,
                current_key: TOP_PRIORITY,
                current_idx: 0,
            },
        )
    }

    fn increment_metrics(&self) {
        self.metrics
            .counter("vortex.scan.segments.count.total")
            .inc();
        self.metrics
            .counter(format!("vortex.scan.segments.count.{:?}", self.kind))
            .inc();
    }
}

#[derive(Debug, Clone)]
pub struct RowRangePruner {
    store: Arc<RwLock<SegmentStore>>,
    cancellations_tx: mpsc::UnboundedSender<SegmentId>,
    excluded_ranges: Arc<RwLock<RangeUnionFind<u64>>>,
    metrics: VortexMetrics,
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
        let first_row = match to_exclude.start_bound() {
            Bound::Included(idx) => *idx,
            Bound::Excluded(idx) => *idx + 1,
            Bound::Unbounded => 0,
        };

        let last_row = match to_exclude.end_bound() {
            Bound::Included(idx) => *idx + 1,
            Bound::Excluded(idx) => *idx,
            Bound::Unbounded => u64::MAX,
        };

        let cancelled_segments: Vec<_> = {
            let mut store = self.store.write()?;
            let to_remove: Vec<_> = store
                .keys()
                .filter(|key| key.kind != RequiredSegmentKind::PRUNING)
                .skip_while(|key| key.row_end < first_row)
                .take_while(|key| key.row_end <= last_row)
                .filter(|key| first_row <= key.row_start)
                .copied()
                .collect();
            to_remove
                .iter()
                .flat_map(|key| store.remove(key).unwrap_or_default())
                .collect()
        };
        self.metrics
            .counter("vortex.scan.segments.cancel_sent")
            .add(cancelled_segments.len() as i64);
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
                let keep =
                    range_intersection(&(key.row_start..key.row_end), &row_indices).is_some();
                if !keep {
                    self.metrics
                        .counter("vortex.scan.segment.pruned_by_row_indices")
                        .inc();
                }
                keep
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
    use futures::executor::block_on;
    use vortex_array::aliases::hash_map::HashMap;
    use vortex_array::aliases::hash_set::HashSet;
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

    fn setup_store() -> Arc<RwLock<SegmentStore>> {
        let mut store = BTreeMap::new();

        // Add segments that span different ranges
        store.insert(
            SegmentPriority::new(0, 100, RequiredSegmentKind::PROJECTION),
            vec![SegmentId(1)],
        );
        store.insert(
            SegmentPriority::new(50, 150, RequiredSegmentKind::PROJECTION),
            vec![SegmentId(2)],
        );
        store.insert(
            SegmentPriority::new(150, 250, RequiredSegmentKind::FILTER),
            vec![SegmentId(3)],
        );
        store.insert(
            SegmentPriority::new(200, 300, RequiredSegmentKind::PROJECTION),
            vec![SegmentId(4)],
        );
        store.insert(
            SegmentPriority::new(0, 0, RequiredSegmentKind::PRUNING),
            vec![SegmentId(5)],
        );

        Arc::new(RwLock::new(store))
    }

    #[test]
    fn test_remove_fully_encompassed_segments() {
        block_on(async {
            // Setup
            let store = setup_store();
            let (tx, mut rx) = mpsc::unbounded();
            let mut pruner = RowRangePruner {
                store: store.clone(),
                cancellations_tx: tx,
                excluded_ranges: Default::default(),
                metrics: Default::default(),
            };

            // Test removing segments in range 0..200
            let result = pruner.remove(0..200).await;
            assert!(result.is_ok(), "Removal operation should succeed");

            // Check that the correct segments were removed from the store
            let store_lock = store.read().vortex_expect("poisoned lock");
            assert!(!store_lock.contains_key(&SegmentPriority::new(
                0,
                100,
                RequiredSegmentKind::PROJECTION
            )));
            assert!(!store_lock.contains_key(&SegmentPriority::new(
                50,
                150,
                RequiredSegmentKind::PROJECTION
            )));
            assert!(store_lock.contains_key(&SegmentPriority::new(
                150,
                250,
                RequiredSegmentKind::FILTER
            ))); // Not fully encompassed
            assert!(store_lock.contains_key(&SegmentPriority::new(
                200,
                300,
                RequiredSegmentKind::PROJECTION
            )));
            assert!(store_lock.contains_key(&SegmentPriority::new(
                0,
                0,
                RequiredSegmentKind::PRUNING
            )));

            // Check that the correct cancellation messages were sent
            let mut received_cancellations = HashSet::new();
            while let Ok(Some(id)) = rx.try_next() {
                received_cancellations.insert(id);
            }

            assert!(received_cancellations.contains(&SegmentId(1)));
            assert!(received_cancellations.contains(&SegmentId(2)));
            assert!(!received_cancellations.contains(&SegmentId(3))); // Not fully encompassed
            assert!(!received_cancellations.contains(&SegmentId(4)));
            assert!(!received_cancellations.contains(&SegmentId(5)));
        })
    }

    #[test]
    fn test_no_double_cancellation() {
        block_on(async {
            // Setup
            let store = setup_store();
            let (tx, mut rx) = mpsc::unbounded();
            let mut pruner = RowRangePruner {
                store: store.clone(),
                cancellations_tx: tx,
                excluded_ranges: Default::default(),
                metrics: Default::default(),
            };

            // First removal (0..100)
            let result = pruner.remove(0..100).await;
            assert!(result.is_ok(), "First removal operation should succeed");

            // Second removal with overlapping range (50..150)
            let result = pruner.remove(50..150).await;
            assert!(result.is_ok(), "Second removal operation should succeed");

            // Third removal with broader range (0..200)
            let result = pruner.remove(0..200).await;
            assert!(result.is_ok(), "Third removal operation should succeed");

            // Check all cancellation messages
            let mut received_cancellations = Vec::new();
            while let Ok(Some(id)) = rx.try_next() {
                received_cancellations.push(id);
            }

            // Count occurrences of each segment ID
            let mut id_counts = HashMap::new();
            for id in received_cancellations {
                *id_counts.entry(id).or_insert(0) += 1;
            }

            // Verify no segment was cancelled more than once
            for (id, count) in id_counts {
                assert_eq!(count, 1, "Segment {:?} was cancelled {} times", id, count);
            }
        })
    }

    #[test]
    fn test_range_merging() {
        block_on(async {
            // Setup
            let store = setup_store();
            let (tx, _rx) = mpsc::unbounded();
            let mut pruner = RowRangePruner {
                store: store.clone(),
                cancellations_tx: tx,
                excluded_ranges: Default::default(),
                metrics: Default::default(),
            };

            // First removal (0..75)
            let result = pruner.remove(0..75).await;
            assert!(result.is_ok());

            // Second removal with adjacent range (75..150)
            let result = pruner.remove(75..150).await;
            assert!(result.is_ok());

            // Third removal with overlapping range (125..200)
            let result = pruner.remove(125..200).await;
            assert!(result.is_ok());

            // Check the store to confirm proper range merging behavior
            let store_lock = store.read().vortex_expect("poisoned lock");
            assert!(!store_lock.contains_key(&SegmentPriority::new(
                0,
                100,
                RequiredSegmentKind::PROJECTION
            )));
            assert!(!store_lock.contains_key(&SegmentPriority::new(
                50,
                150,
                RequiredSegmentKind::PROJECTION
            )));
            assert!(store_lock.contains_key(&SegmentPriority::new(
                150,
                250,
                RequiredSegmentKind::FILTER
            ))); // Not fully encompassed
        })
    }

    #[test]
    fn test_retain_matching_with_pruning_segments() {
        block_on(async {
            // Setup
            let store = setup_store();
            let (tx, _rx) = mpsc::unbounded();
            let mut pruner = RowRangePruner {
                store: store.clone(),
                cancellations_tx: tx,
                excluded_ranges: Default::default(),
                metrics: Default::default(),
            };

            // Create a buffer with specific row indices
            let row_indices = Buffer::from_iter(vec![75, 125, 175, 225, 325, 375]);

            // Call retain_matching
            pruner.retain_matching(row_indices);

            // Check that the correct segments were retained
            let store_lock = store.read().vortex_expect("poisoned lock");

            // Segments that intersect with the row indices should be kept
            assert!(store_lock.contains_key(&SegmentPriority::new(
                0,
                100,
                RequiredSegmentKind::PROJECTION
            ))); // Contains 75
            assert!(store_lock.contains_key(&SegmentPriority::new(
                50,
                150,
                RequiredSegmentKind::PROJECTION
            ))); // Contains 75, 125
            assert!(store_lock.contains_key(&SegmentPriority::new(
                150,
                250,
                RequiredSegmentKind::FILTER
            ))); // Contains 175, 225
            assert!(store_lock.contains_key(&SegmentPriority::new(
                200,
                300,
                RequiredSegmentKind::PROJECTION
            ))); // Contains 225

            // PRUNING segments should always be kept
            assert!(store_lock.contains_key(&SegmentPriority::new(
                0,
                0,
                RequiredSegmentKind::PRUNING
            )));
        })
    }

    #[test]
    fn test_cancellation_channel_closed() {
        block_on(async {
            let store = setup_store();
            let (tx, rx) = mpsc::unbounded();
            let mut pruner = RowRangePruner {
                store: store.clone(),
                cancellations_tx: tx,
                excluded_ranges: Default::default(),
                metrics: Default::default(),
            };

            // Drop the receiver to close the channel
            drop(rx);

            // Attempt to remove segments
            let result = pruner.remove(0..100).await;

            // Should fail with channel closed error
            assert!(
                result.is_err(),
                "Removal should fail when channel is closed"
            );
        })
    }

    #[test]
    fn test_segments_of_different_kinds() {
        block_on(async {
            // Setup
            let store = setup_store();
            let (tx, mut rx) = mpsc::unbounded();
            let mut pruner = RowRangePruner {
                store: store.clone(),
                cancellations_tx: tx,
                excluded_ranges: Default::default(),
                metrics: Default::default(),
            };

            // Test removing segments that cover the entire range
            let result = pruner.remove(0..400).await;
            assert!(result.is_ok(), "Removal operation should succeed");

            // Check that segments of all kinds that are fully encompassed were removed
            let store_lock = store.read().vortex_expect("poisoned lock");
            assert_eq!(store_lock.len(), 1);

            // Verify the cancellations
            let mut received_cancellations = HashSet::new();
            while let Ok(Some(id)) = rx.try_next() {
                received_cancellations.insert(id);
            }

            assert!(received_cancellations.contains(&SegmentId(1))); // PROJECTION
            assert!(received_cancellations.contains(&SegmentId(2))); // PROJECTION
            assert!(received_cancellations.contains(&SegmentId(3))); // FILTER
            assert!(received_cancellations.contains(&SegmentId(4))); // PROJECTION
        })
    }
}
