// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Keeps recently read segments so the chunks of a scan that share a segment read it once.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use futures::FutureExt;
use futures::TryFutureExt;
use parking_lot::RwLock;
use rustc_hash::FxHashMap;
use vortex_array::buffer::BufferHandle;

use crate::segments::SegmentFuture;
use crate::segments::SegmentId;
use crate::segments::SegmentSource;

/// The host bytes of recently read segments, evicting segments that have gone unused the longest
/// once past a budget.
///
/// A V2 scan cuts its rows into chunks that each read through a session of their own, and a
/// stored segment that spans a chunk boundary is read by every chunk it overlaps. The file's segment
/// source shares a read only while it is in flight, so a chunk that starts after its neighbour
/// finished would read the segment again. Neighbouring chunks run close together, so a small
/// budget is enough to read each such segment once.
///
/// Every segment request of every chunk looks a segment up here, from many threads at once.
/// Segments are spread over shards, each behind its own lock, and a hit takes only a shared lock
/// on its shard: it marks the segment used rather than reordering anything. Eviction gives a
/// segment that was used since it was queued a second chance at the back of its shard's queue,
/// which approximates least-recently-used order without a hit ever waiting for another. The
/// budget is shared: an insert evicts from its own shard until the retained bytes of all shards
/// fit, and a shard left holding only the inserted segment leaves the rest to later inserts.
pub(super) struct RetainedSegments {
    budget: u64,
    /// The retained bytes of every shard.
    bytes: AtomicU64,
    shards: Box<[RwLock<State>]>,
}

/// The shards a file's retained segments are spread over.
const SHARDS: usize = 16;

/// The bytes a file's scans retain by default.
const DEFAULT_RETAINED_BYTES: u64 = 64 << 20;

impl Default for RetainedSegments {
    fn default() -> Self {
        // EXPERIMENT (not for merge): XP_MODE=*_nr retains nothing.
        #[allow(clippy::disallowed_methods)]
        if std::env::var("XP_MODE").is_ok_and(|m| m.ends_with("_nr")) {
            return Self::new(0);
        }
        Self::new(DEFAULT_RETAINED_BYTES)
    }
}

/// EXPERIMENT (not for merge): lock-wait and request timings for the CI A/B.
#[doc(hidden)]
#[allow(missing_docs)]
pub mod xp_layout {
    use std::sync::atomic::AtomicU64;
    use std::sync::atomic::Ordering::Relaxed;
    use std::time::Instant;

    pub static RETAIN_READ_WAIT_NS: AtomicU64 = AtomicU64::new(0);
    pub static RETAIN_WRITE_WAIT_NS: AtomicU64 = AtomicU64::new(0);
    pub static REQUEST_NS: AtomicU64 = AtomicU64::new(0);
    pub static REQUESTS: AtomicU64 = AtomicU64::new(0);

    pub fn add(c: &AtomicU64, since: Instant) {
        c.fetch_add(u64::try_from(since.elapsed().as_nanos()).unwrap_or(u64::MAX), Relaxed);
    }

    pub fn reset() {
        for c in [&RETAIN_READ_WAIT_NS, &RETAIN_WRITE_WAIT_NS, &REQUEST_NS, &REQUESTS] {
            c.store(0, Relaxed);
        }
    }
}

#[derive(Default)]
struct State {
    segments: FxHashMap<SegmentId, Retained>,
    /// Retained segments in the order they were queued, oldest first.
    queue: VecDeque<SegmentId>,
}

struct Retained {
    buffer: BufferHandle,
    /// Whether the segment was used since it was last queued.
    used: AtomicBool,
}

impl RetainedSegments {
    pub(super) fn new(budget: u64) -> Self {
        Self::with_shards(budget, SHARDS)
    }

    fn with_shards(budget: u64, shards: usize) -> Self {
        Self {
            budget,
            bytes: AtomicU64::new(0),
            shards: (0..shards.max(1)).map(|_| RwLock::default()).collect(),
        }
    }

    fn shard(&self, id: SegmentId) -> &RwLock<State> {
        &self.shards[*id as usize % self.shards.len()]
    }

    fn get(&self, id: SegmentId) -> Option<BufferHandle> {
        let t = std::time::Instant::now();
        let state = self.shard(id).read();
        xp_layout::add(&xp_layout::RETAIN_READ_WAIT_NS, t);
        let retained = state.segments.get(&id)?;
        // Skip the store when already set, so hits on a shared segment do not keep writing to it.
        if !retained.used.load(Ordering::Relaxed) {
            retained.used.store(true, Ordering::Relaxed);
        }
        Some(retained.buffer.clone())
    }

    fn insert(&self, id: SegmentId, buffer: &BufferHandle) {
        let len = buffer.len() as u64;
        if !buffer.is_on_host() || len > self.budget / 4 {
            return;
        }
        let t = std::time::Instant::now();
        let mut state = self.shard(id).write();
        xp_layout::add(&xp_layout::RETAIN_WRITE_WAIT_NS, t);
        if state.segments.contains_key(&id) {
            return;
        }
        state.segments.insert(
            id,
            Retained {
                buffer: buffer.clone(),
                used: AtomicBool::new(false),
            },
        );
        state.queue.push_back(id);
        self.bytes.fetch_add(len, Ordering::Relaxed);
        // Two passes over the shard's queue clear every mark once, so the loop ends even when
        // the shard holds only the inserted segment.
        let mut visits = 2 * state.queue.len();
        while visits > 0 && self.bytes.load(Ordering::Relaxed) > self.budget {
            visits -= 1;
            let Some(oldest) = state.queue.pop_front() else {
                break;
            };
            let Some(retained) = state.segments.get(&oldest) else {
                continue;
            };
            // The segment being inserted is about to be used, and a segment used since it was
            // queued goes to the back once, unmarked.
            if oldest == id || retained.used.swap(false, Ordering::Relaxed) {
                state.queue.push_back(oldest);
                continue;
            }
            if let Some(evicted) = state.segments.remove(&oldest) {
                self.bytes
                    .fetch_sub(evicted.buffer.len() as u64, Ordering::Relaxed);
            }
        }
    }
}

/// Serves segments from [`RetainedSegments`] before reading them from `inner`, and retains
/// what it reads.
pub(super) struct RetainingSegmentSource {
    pub(super) inner: Arc<dyn SegmentSource>,
    pub(super) retained: Arc<RetainedSegments>,
}

impl SegmentSource for RetainingSegmentSource {
    fn request(&self, id: SegmentId) -> SegmentFuture {
        let t = std::time::Instant::now();
        xp_layout::REQUESTS.fetch_add(1, Ordering::Relaxed);
        if let Some(buffer) = self.retained.get(id) {
            xp_layout::add(&xp_layout::REQUEST_NS, t);
            return futures::future::ready(Ok(buffer)).boxed();
        }
        let retained = Arc::clone(&self.retained);
        let fut = self
            .inner
            .request(id)
            .inspect_ok(move |buffer| retained.insert(id, buffer))
            .boxed();
        xp_layout::add(&xp_layout::REQUEST_NS, t);
        fut
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::buffer::BufferHandle;
    use vortex_buffer::ByteBuffer;

    use super::RetainedSegments;
    use crate::segments::SegmentId;

    fn buffer(len: usize) -> BufferHandle {
        BufferHandle::new_host(ByteBuffer::from(vec![0u8; len]))
    }

    #[test]
    fn evicts_least_recently_used_past_budget() {
        let retained = RetainedSegments::with_shards(400, 1);
        retained.insert(SegmentId::from(0), &buffer(100));
        retained.insert(SegmentId::from(1), &buffer(100));
        retained.insert(SegmentId::from(2), &buffer(100));
        // Using segment 0 makes segment 1 the least recently used.
        assert!(retained.get(SegmentId::from(0)).is_some());
        retained.insert(SegmentId::from(3), &buffer(100));
        retained.insert(SegmentId::from(4), &buffer(100));

        assert!(retained.get(SegmentId::from(1)).is_none());
        for id in [0, 2, 3, 4] {
            assert!(retained.get(SegmentId::from(id)).is_some(), "segment {id}");
        }
    }

    #[test]
    fn evicts_within_budget_when_every_segment_was_used() {
        let retained = RetainedSegments::with_shards(400, 1);
        for id in 0..4 {
            retained.insert(SegmentId::from(id), &buffer(100));
        }
        for id in 0..4 {
            assert!(retained.get(SegmentId::from(id)).is_some(), "segment {id}");
        }
        // Every queued segment was used, so each gets its second chance, and the oldest goes.
        retained.insert(SegmentId::from(4), &buffer(100));

        assert!(retained.get(SegmentId::from(0)).is_none());
        for id in 1..5 {
            assert!(retained.get(SegmentId::from(id)).is_some(), "segment {id}");
        }
    }

    #[test]
    fn shards_share_one_budget() {
        let retained = RetainedSegments::with_shards(400, 4);
        for id in 0..8 {
            retained.insert(SegmentId::from(id), &buffer(100));
        }
        let kept = (0..8)
            .filter(|id| retained.get(SegmentId::from(*id)).is_some())
            .count();
        assert_eq!(kept, 4);
        assert!(retained.get(SegmentId::from(7)).is_some());
    }

    #[test]
    fn skips_segments_over_a_quarter_of_the_budget() {
        let retained = RetainedSegments::with_shards(400, 1);
        retained.insert(SegmentId::from(0), &buffer(101));
        assert!(retained.get(SegmentId::from(0)).is_none());
    }
}
