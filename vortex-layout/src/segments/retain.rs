// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Keeps recently read segments, so later requests for them read nothing.

use std::collections::BTreeMap;
use std::sync::Arc;

use futures::FutureExt;
use futures::TryFutureExt;
use parking_lot::Mutex;
use rustc_hash::FxHashMap;
use vortex_array::buffer::BufferHandle;

use crate::segments::SegmentBytes;
use crate::segments::SegmentFuture;
use crate::segments::SegmentId;
use crate::segments::SegmentSource;

/// The host bytes of recently read segments, evicting the least recently used past a byte
/// budget.
///
/// One store can serve many [`RetainingSegmentSource`]s, such as the sources of the scans of one
/// file, so a segment that one of them read is served to the others. A segment larger than a
/// quarter of the budget is not kept, so one large segment does not evict many small ones.
/// Segments that do not live in host memory are not kept.
pub struct RetainedSegments {
    budget: u64,
    state: Mutex<State>,
}

/// The bytes a store keeps by default.
const DEFAULT_RETAINED_BYTES: u64 = 64 << 20;

impl Default for RetainedSegments {
    fn default() -> Self {
        Self::new(DEFAULT_RETAINED_BYTES)
    }
}

#[derive(Default)]
struct State {
    bytes: u64,
    tick: u64,
    /// Each retained segment and the tick of its last use.
    segments: FxHashMap<SegmentId, (BufferHandle, u64)>,
    /// Retained segments by the tick of their last use, oldest first.
    by_use: BTreeMap<u64, SegmentId>,
}

impl RetainedSegments {
    /// A store that keeps at most `budget` bytes. A budget of zero keeps nothing.
    pub fn new(budget: u64) -> Self {
        Self {
            budget,
            state: Mutex::default(),
        }
    }

    /// A store that keeps every segment it is given, until it is dropped.
    pub fn unbounded() -> Self {
        Self::new(u64::MAX)
    }

    fn get(&self, id: SegmentId) -> Option<BufferHandle> {
        let mut state = self.state.lock();
        state.tick += 1;
        let tick = state.tick;
        let (buffer, used) = state.segments.get_mut(&id)?;
        let buffer = buffer.clone();
        let previous = std::mem::replace(used, tick);
        state.by_use.remove(&previous);
        state.by_use.insert(tick, id);
        Some(buffer)
    }

    fn insert(&self, id: SegmentId, buffer: &BufferHandle) {
        let len = buffer.len() as u64;
        if !buffer.is_on_host() || len > self.budget / 4 {
            return;
        }
        let mut state = self.state.lock();
        if state.segments.contains_key(&id) {
            return;
        }
        state.tick += 1;
        let tick = state.tick;
        state.segments.insert(id, (buffer.clone(), tick));
        state.by_use.insert(tick, id);
        state.bytes += len;
        while state.bytes > self.budget {
            let Some((_, oldest)) = state.by_use.pop_first() else {
                break;
            };
            if let Some((evicted, _)) = state.segments.remove(&oldest) {
                state.bytes -= evicted.len() as u64;
            }
        }
    }
}

/// Serves segments from [`RetainedSegments`] before reading them from an inner source, and
/// retains what it reads.
///
/// It does not share requests that are still in flight. Wrap a source that does, such as a
/// [`SharedSegmentSource`](crate::segments::SharedSegmentSource), when concurrent requests for one
/// segment must read it once.
pub struct RetainingSegmentSource {
    inner: Arc<dyn SegmentSource>,
    retained: Arc<RetainedSegments>,
}

impl RetainingSegmentSource {
    /// Serves the segments of `inner` through `retained`.
    pub fn new(inner: Arc<dyn SegmentSource>, retained: Arc<RetainedSegments>) -> Self {
        Self { inner, retained }
    }
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

    fn reads_ranges(&self) -> bool {
        self.inner.reads_ranges()
    }

    /// Serves part of a retained segment from it. It does not retain a part that it reads.
    fn request_bytes(&self, id: SegmentId, bytes: SegmentBytes) -> SegmentFuture {
        if bytes == SegmentBytes::All {
            return self.request(id);
        }

        if let Some(buffer) = self.retained.get(id) {
            return futures::future::ready(bytes.slice(buffer)).boxed();
        }
        self.inner.request_bytes(id, bytes)
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
    fn unbounded_keeps_every_segment() {
        let retained = RetainedSegments::unbounded();
        for id in 0..10 {
            retained.insert(SegmentId::from(id), &buffer(1 << 20));
        }
        for id in 0..10 {
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
