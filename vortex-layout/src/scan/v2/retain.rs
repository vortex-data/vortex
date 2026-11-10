// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Keeps recently read segments so the chunks of a scan that share a segment read it once.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
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
/// Every segment request of every chunk looks a segment up here, so a hit takes only a shared
/// lock: it marks the segment used rather than reordering anything. Eviction gives a segment that
/// was used since it was queued a second chance at the back of the queue, which approximates
/// least-recently-used order without a hit ever waiting for another.
pub(super) struct RetainedSegments {
    budget: u64,
    state: RwLock<State>,
}

/// The bytes a file's scans retain by default.
const DEFAULT_RETAINED_BYTES: u64 = 64 << 20;

impl Default for RetainedSegments {
    fn default() -> Self {
        Self::new(DEFAULT_RETAINED_BYTES)
    }
}

#[derive(Default)]
struct State {
    bytes: u64,
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
        Self {
            budget,
            state: RwLock::default(),
        }
    }

    fn get(&self, id: SegmentId) -> Option<BufferHandle> {
        let state = self.state.read();
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
        let mut state = self.state.write();
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
        state.bytes += len;
        while state.bytes > self.budget {
            let Some(oldest) = state.queue.pop_front() else {
                break;
            };
            let Some(retained) = state.segments.get(&oldest) else {
                continue;
            };
            // The segment being inserted is about to be used, and a segment used since it was
            // queued goes to the back once, unmarked. Neither loops forever: the inserted segment
            // fits the budget alone, and every other pass clears a mark.
            if oldest == id || retained.used.swap(false, Ordering::Relaxed) {
                state.queue.push_back(oldest);
                continue;
            }
            if let Some(evicted) = state.segments.remove(&oldest) {
                state.bytes -= evicted.buffer.len() as u64;
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
        if let Some(buffer) = self.retained.get(id) {
            return futures::future::ready(Ok(buffer)).boxed();
        }
        let retained = Arc::clone(&self.retained);
        self.inner
            .request(id)
            .inspect_ok(move |buffer| retained.insert(id, buffer))
            .boxed()
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
        let retained = RetainedSegments::new(400);
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
        let retained = RetainedSegments::new(400);
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
    fn skips_segments_over_a_quarter_of_the_budget() {
        let retained = RetainedSegments::new(400);
        retained.insert(SegmentId::from(0), &buffer(101));
        assert!(retained.get(SegmentId::from(0)).is_none());
    }
}
