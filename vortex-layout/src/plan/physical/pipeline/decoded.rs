// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Decoded segments shared by every scan of a file while live work still covers their rows.

use std::ops::Range;
use std::sync::Arc;
use std::sync::Weak;

use parking_lot::Mutex;
use rustc_hash::FxHashMap;
use vortex_array::ArrayRef;
use vortex_scan::planning::driver::Lifetime;
use vortex_scan::planning::driver::Liveness;

use crate::segments::SegmentId;

/// The root rows each segment of a file serves, by segment id.
pub type SegmentRows = Arc<FxHashMap<SegmentId, Range<u64>>>;

/// Whole decoded segments, shared by the filter, pruning, and projection pipelines of every
/// chunk of a file, each kept while the driver reports live work over the rows it serves.
///
/// A segment that spans chunks, or that a chunk's filter and projection both read, is otherwise
/// decoded once per reader. Here the first reader's decode is kept, pinned in the file's
/// [`Liveness`] to the segment's root rows, and dropped as soon as no planner or morsel claims
/// any of them, so the cache holds no more than the work in flight can still use.
///
/// The default cache keeps nothing.
#[derive(Clone, Default)]
pub struct DecodedSegments(Option<Arc<Inner>>);

struct Inner {
    liveness: Liveness,
    rows: SegmentRows,
    entries: Mutex<Entries>,
}

#[derive(Default)]
struct Entries {
    arrays: FxHashMap<SegmentId, (ArrayRef, u64)>,
    next_generation: u64,
}

/// Removes its segment's entry when liveness drops it, unless a later decode replaced it.
struct Evict {
    inner: Weak<Inner>,
    segment: SegmentId,
    generation: u64,
}

impl Drop for Evict {
    fn drop(&mut self) {
        let Some(inner) = self.inner.upgrade() else {
            return;
        };
        let evicted = {
            let mut entries = inner.entries.lock();
            let generation = entries
                .arrays
                .get(&self.segment)
                .map(|(_, generation)| *generation);
            if generation == Some(self.generation) {
                entries.arrays.remove(&self.segment)
            } else {
                None
            }
        };
        drop(evicted);
    }
}

impl DecodedSegments {
    /// A cache pinning each segment to its rows in `rows`, or to the whole file when `rows` does
    /// not list it.
    pub fn new(liveness: Liveness, rows: SegmentRows) -> Self {
        Self(Some(Arc::new(Inner {
            liveness,
            rows,
            entries: Mutex::default(),
        })))
    }

    /// The decoded segment an earlier reader kept.
    pub(crate) fn get(&self, segment: SegmentId) -> Option<ArrayRef> {
        let inner = self.0.as_ref()?;
        inner
            .entries
            .lock()
            .arrays
            .get(&segment)
            .map(|(array, _)| array.clone())
    }

    /// Whether a decode of `segment` is kept.
    pub(crate) fn contains(&self, segment: SegmentId) -> bool {
        self.0
            .as_ref()
            .is_some_and(|inner| inner.entries.lock().arrays.contains_key(&segment))
    }

    /// Keeps `array`, the decode of `segment`, while live work claims the segment's rows, and
    /// returns the decode readers should use: an earlier reader's when two race.
    pub(crate) fn insert(&self, segment: SegmentId, array: ArrayRef) -> ArrayRef {
        let Some(inner) = self.0.as_ref() else {
            return array;
        };
        let generation = {
            let mut entries = inner.entries.lock();
            if let Some((kept, _)) = entries.arrays.get(&segment) {
                return kept.clone();
            }
            let generation = entries.next_generation;
            entries.next_generation += 1;
            entries.arrays.insert(segment, (array.clone(), generation));
            generation
        };
        let lifetime = inner
            .rows
            .get(&segment)
            .map_or(Lifetime::File, |rows| Lifetime::Rows(rows.clone()));
        // Pinning unclaimed rows drops the guard at once, which removes the entry again. The
        // entries lock is released first, as the guard takes it.
        inner.liveness.pin(
            lifetime,
            Evict {
                inner: Arc::downgrade(inner),
                segment,
                generation,
            },
        );
        array
    }
}
