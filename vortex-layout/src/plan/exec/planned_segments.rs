// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Decoded segments shared between the splits of every scan over one file, kept only while a split
//! that has not finished is planned to read them.

use std::env;
use std::ptr;
use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::Weak;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use parking_lot::Mutex;
use vortex_array::ArrayRef;
use vortex_utils::aliases::hash_map::HashMap;

use crate::LayoutRef;
use crate::layout::DynLayout;
use crate::segments::SegmentId;

/// Shards of the segment map; segment ids spread over them, so splits touching different segments
/// rarely take the same lock.
const SHARDS: usize = 16;

/// Decoded segments of one file, shared by every split of every scan over it.
///
/// Each split registers the segments its plans read when it announces them, and releases them
/// when it finishes or is dropped. A decoded segment is kept only while another split that has
/// registered it is still live, and is dropped as soon as the last one releases it, so the cache
/// holds nothing no planned split will read and nothing outlives the scans of a query. Every cache
/// also shares one byte budget; a segment that would exceed it is not kept.
pub struct PlannedSegments {
    shards: [Mutex<HashMap<SegmentId, Slot>>; SHARDS],
    hits: AtomicU64,
    stored: AtomicU64,
}

#[derive(Default)]
struct Slot {
    /// Live splits that registered the segment.
    demand: u32,
    /// The decoded segment and the bytes it counts against the budget.
    array: Option<(ArrayRef, u64)>,
}

/// Bytes of decoded segments held by every cache together.
static HELD: AtomicU64 = AtomicU64::new(0);

/// The most bytes every cache together holds, from `VORTEX_SCAN_SHARED_SEGMENTS_MB` (default 256).
static BUDGET: LazyLock<u64> = LazyLock::new(|| {
    env::var("VORTEX_SCAN_SHARED_SEGMENTS_MB")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(256)
        .saturating_mul(1 << 20)
});

/// Whether scans share decoded segments; `VORTEX_SCAN_SHARED_SEGMENTS=0` turns it off.
static ENABLED: LazyLock<bool> =
    LazyLock::new(|| !env::var("VORTEX_SCAN_SHARED_SEGMENTS").is_ok_and(|value| value == "0"));

/// The cache of every live file, keyed by its root layout. Engines that cache a file's footer
/// open every scan of the file over the same layout.
type Entry = (Weak<dyn DynLayout>, Weak<PlannedSegments>);
static FILES: LazyLock<Mutex<Vec<Entry>>> = LazyLock::new(Default::default);

impl PlannedSegments {
    fn new() -> Self {
        Self {
            shards: Default::default(),
            hits: AtomicU64::new(0),
            stored: AtomicU64::new(0),
        }
    }

    /// The cache shared by the scans over `layout`, or `None` when sharing is turned off.
    pub(crate) fn for_layout(layout: &LayoutRef) -> Option<Arc<Self>> {
        if !*ENABLED {
            return None;
        }
        let mut files = FILES.lock();
        files.retain(|(layout, cache)| layout.strong_count() > 0 && cache.strong_count() > 0);
        if let Some(cache) = files
            .iter()
            .find(|(known, _)| ptr::addr_eq(known.as_ptr(), Arc::as_ptr(layout)))
            .and_then(|(_, cache)| cache.upgrade())
        {
            return Some(cache);
        }
        let cache = Arc::new(Self::new());
        files.push((Arc::downgrade(layout), Arc::downgrade(&cache)));
        Some(cache)
    }

    fn shard(&self, id: SegmentId) -> &Mutex<HashMap<SegmentId, Slot>> {
        &self.shards[*id as usize % SHARDS]
    }

    /// Registers a split's demand for `ids`, released when the returned guard drops.
    pub(crate) fn register(self: &Arc<Self>, ids: Vec<SegmentId>) -> SplitSegments {
        for &id in &ids {
            self.shard(id).lock().entry(id).or_default().demand += 1;
        }
        SplitSegments {
            cache: Arc::clone(self),
            ids,
        }
    }

    fn release(&self, ids: &[SegmentId]) {
        for &id in ids {
            let mut shard = self.shard(id).lock();
            let Some(slot) = shard.get_mut(&id) else {
                continue;
            };
            slot.demand = slot.demand.saturating_sub(1);
            if slot.demand == 0
                && let Some(slot) = shard.remove(&id)
                && let Some((_, bytes)) = slot.array
            {
                HELD.fetch_sub(bytes, Ordering::Relaxed);
            }
        }
    }

    /// The decoded segment `id`, if a split decoded it while another still planned to read it.
    pub(crate) fn get(&self, id: SegmentId) -> Option<ArrayRef> {
        let array = self
            .shard(id)
            .lock()
            .get(&id)
            .and_then(|slot| slot.array.as_ref().map(|(array, _)| array.clone()));
        if array.is_some() {
            self.hits.fetch_add(1, Ordering::Relaxed);
        }
        array
    }

    /// Whether the decoded segment `id` is held.
    pub(crate) fn contains(&self, id: SegmentId) -> bool {
        self.shard(id)
            .lock()
            .get(&id)
            .is_some_and(|slot| slot.array.is_some())
    }

    /// Keeps the segment `id` a split decoded, if another live split registered it and the budget
    /// has room.
    fn offer(&self, id: SegmentId, array: &ArrayRef) {
        let mut shard = self.shard(id).lock();
        let Some(slot) = shard.get_mut(&id) else {
            return;
        };
        if slot.demand < 2 || slot.array.is_some() {
            return;
        }
        let bytes = array.nbytes();
        if HELD
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |held| {
                held.checked_add(bytes).filter(|&total| total <= *BUDGET)
            })
            .is_err()
        {
            return;
        }
        slot.array = Some((array.clone(), bytes));
        self.stored.fetch_add(1, Ordering::Relaxed);
    }
}

impl Drop for PlannedSegments {
    fn drop(&mut self) {
        // Splits release their demand before the cache drops, so nothing is held by now; any
        // segment still counted is returned to the budget.
        for shard in &self.shards {
            for (_, slot) in shard.lock().drain() {
                if let Some((_, bytes)) = slot.array {
                    HELD.fetch_sub(bytes, Ordering::Relaxed);
                }
            }
        }
        tracing::debug!(
            target: "vortex_layout::scan::v2::segments",
            hits = self.hits.load(Ordering::Relaxed),
            stored = self.stored.load(Ordering::Relaxed),
            "shared segments dropped"
        );
    }
}

/// A split's registered demand for the segments its plans read, released on drop.
pub struct SplitSegments {
    cache: Arc<PlannedSegments>,
    ids: Vec<SegmentId>,
}

impl SplitSegments {
    pub(crate) fn get(&self, id: SegmentId) -> Option<ArrayRef> {
        self.cache.get(id)
    }

    pub(crate) fn contains(&self, id: SegmentId) -> bool {
        self.cache.contains(id)
    }

    pub(crate) fn offer(&self, id: SegmentId, array: &ArrayRef) {
        self.cache.offer(id, array);
    }
}

impl Drop for SplitSegments {
    fn drop(&mut self) {
        self.cache.release(&self.ids);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_array::IntoArray;
    use vortex_array::arrays::PrimitiveArray;

    use super::PlannedSegments;
    use crate::segments::SegmentId;

    #[test]
    fn keeps_a_segment_only_while_another_split_plans_to_read_it() {
        let cache = Arc::new(PlannedSegments::new());
        let id = SegmentId::from(3);
        let array = PrimitiveArray::from_iter(0..1024i32).into_array();

        let first = cache.register(vec![id]);
        first.offer(id, &array);
        assert!(!first.contains(id), "no other split planned to read it");

        let second = cache.register(vec![id]);
        first.offer(id, &array);
        assert!(second.get(id).is_some());

        drop(first);
        assert!(
            second.contains(id),
            "the second split still plans to read it"
        );
        drop(second);
        assert!(!cache.contains(id), "released with the last split");
    }

    #[test]
    fn ignores_segments_no_split_registered() {
        let cache = Arc::new(PlannedSegments::new());
        let split = cache.register(vec![SegmentId::from(1)]);
        let array = PrimitiveArray::from_iter(0..8i32).into_array();
        split.offer(SegmentId::from(2), &array);
        assert!(!split.contains(SegmentId::from(2)));
    }
}
