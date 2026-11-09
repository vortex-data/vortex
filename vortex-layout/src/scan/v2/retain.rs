// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Keeps recently read segments so the chunks of a scan that share a segment read it once.

use std::collections::BTreeMap;
use std::sync::Arc;

use futures::FutureExt;
use futures::TryFutureExt;
use parking_lot::Mutex;
use rustc_hash::FxHashMap;
use vortex_array::buffer::BufferHandle;

use crate::segments::SegmentFuture;
use crate::segments::SegmentId;
use crate::segments::SegmentSource;

/// The host bytes of recently read segments, evicting the least recently used past a budget.
///
/// A V2 scan cuts its rows into chunks that each read through a session of their own, and a
/// stored segment that spans a chunk boundary is read by every chunk it overlaps. The file's segment
/// source shares a read only while it is in flight, so a chunk that starts after its neighbour
/// finished would read the segment again. Neighbouring chunks run close together, so a small
/// budget is enough to read each such segment once.
pub(super) struct RetainedSegments {
    budget: u64,
    state: Mutex<State>,
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
    tick: u64,
    /// Each retained segment and the tick of its last use.
    segments: FxHashMap<SegmentId, (BufferHandle, u64)>,
    /// Retained segments by the tick of their last use, oldest first.
    by_use: BTreeMap<u64, SegmentId>,
}

impl RetainedSegments {
    pub(super) fn new(budget: u64) -> Self {
        Self {
            budget,
            state: Mutex::default(),
        }
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
    fn skips_segments_over_a_quarter_of_the_budget() {
        let retained = RetainedSegments::new(400);
        retained.insert(SegmentId::from(0), &buffer(101));
        assert!(retained.get(SegmentId::from(0)).is_none());
    }
}
