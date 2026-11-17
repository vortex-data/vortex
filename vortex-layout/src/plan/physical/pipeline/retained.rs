// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Decoded segments kept while live work still claims the file rows they cover.

use std::ops::Range;
use std::sync::Arc;
use std::sync::Weak;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use parking_lot::RwLock;
use rustc_hash::FxHashMap;
use vortex_array::ArrayRef;
use vortex_scan::planning::driver::Lifetime;
use vortex_scan::planning::driver::Liveness;

use crate::segments::SegmentId;

const SHARDS: usize = 32;

/// The arrays decoded from a file's segments, kept while live work in a [`Liveness`] still
/// claims a row the segment covers.
///
/// The neighbouring chunks of a scan each read a segment that spans the boundary between them,
/// and every chunk reads a zone table whole. Each of those reads finds the decoded array here
/// instead of reading and decoding the segment again, and the array is dropped once no live work
/// covers its rows, so the arrays kept are only those of rows a scan has yet to finish.
///
/// Lookups take a read lock on one of several shards, so readers on many threads do not contend
/// on one lock. The byte budget only bounds memory if work is claimed far ahead of its reads; the
/// claims keep the arrays to the rows in flight.
#[derive(Clone)]
pub struct RetainedArrays(Arc<Inner>);

struct Inner {
    liveness: Liveness,
    budget: u64,
    bytes: AtomicU64,
    next_generation: AtomicU64,
    shards: [RwLock<FxHashMap<SegmentId, Entry>>; SHARDS],
}

struct Entry {
    array: ArrayRef,
    len: u64,
    generation: u64,
}

/// The pin holding one retained array. Dropping it removes the array, unless it was retained
/// again since.
struct Pin {
    inner: Weak<Inner>,
    segment: SegmentId,
    generation: u64,
}

impl Drop for Pin {
    fn drop(&mut self) {
        let Some(inner) = self.inner.upgrade() else {
            return;
        };
        let removed = {
            let mut shard = inner.shard(self.segment).write();
            match shard.get(&self.segment) {
                Some(entry) if entry.generation == self.generation => shard.remove(&self.segment),
                _ => None,
            }
        };
        if let Some(entry) = removed {
            inner.bytes.fetch_sub(entry.len, Ordering::Relaxed);
        }
    }
}

impl Inner {
    fn shard(&self, segment: SegmentId) -> &RwLock<FxHashMap<SegmentId, Entry>> {
        &self.shards[*segment as usize % SHARDS]
    }
}

impl RetainedArrays {
    /// Arrays kept while work in `liveness` claims their rows, up to `budget` bytes of segments.
    pub fn new(liveness: Liveness, budget: u64) -> Self {
        Self(Arc::new(Inner {
            liveness,
            budget,
            bytes: AtomicU64::new(0),
            next_generation: AtomicU64::new(0),
            shards: std::array::from_fn(|_| RwLock::default()),
        }))
    }

    /// The array decoded from `segment`, if one is retained.
    pub(crate) fn get(&self, segment: SegmentId) -> Option<ArrayRef> {
        self.0
            .shard(segment)
            .read()
            .get(&segment)
            .map(|entry| entry.array.clone())
    }

    /// Retains `array`, decoded from the `len` bytes of `segment`, while work claims `rows`.
    pub(crate) fn insert(&self, segment: SegmentId, array: &ArrayRef, len: u64, rows: Range<u64>) {
        if self.0.bytes.load(Ordering::Relaxed).saturating_add(len) > self.0.budget {
            return;
        }
        let generation = self.0.next_generation.fetch_add(1, Ordering::Relaxed);
        {
            let mut shard = self.0.shard(segment).write();
            if shard.contains_key(&segment) {
                return;
            }
            shard.insert(
                segment,
                Entry {
                    array: array.clone(),
                    len,
                    generation,
                },
            );
        }
        self.0.bytes.fetch_add(len, Ordering::Relaxed);
        // No shard lock may be held here: a pin on rows no work claims is dropped at once.
        self.0.liveness.pin(
            Lifetime::Rows(rows),
            Pin {
                inner: Arc::downgrade(&self.0),
                segment,
                generation,
            },
        );
    }

    /// The bytes of the segments retained now.
    #[cfg(test)]
    pub(crate) fn retained_bytes(&self) -> u64 {
        self.0.bytes.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::IntoArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_scan::planning::driver::Liveness;

    use super::RetainedArrays;
    use crate::segments::SegmentId;

    fn array() -> vortex_array::ArrayRef {
        PrimitiveArray::from_iter([1i32, 2, 3]).into_array()
    }

    #[test]
    fn an_array_lives_while_its_rows_are_claimed() {
        let liveness = Liveness::new();
        let retained = RetainedArrays::new(liveness.clone(), 1 << 20);
        let claim = liveness.claim(0..100);
        retained.insert(SegmentId::from(0), &array(), 12, 10..20);
        assert!(retained.get(SegmentId::from(0)).is_some());
        assert_eq!(retained.retained_bytes(), 12);

        drop(claim);
        assert!(retained.get(SegmentId::from(0)).is_none());
        assert_eq!(retained.retained_bytes(), 0);
    }

    #[test]
    fn an_array_of_unclaimed_rows_is_not_kept() {
        let liveness = Liveness::new();
        let retained = RetainedArrays::new(liveness.clone(), 1 << 20);
        let _claim = liveness.claim(0..10);
        retained.insert(SegmentId::from(0), &array(), 12, 10..20);
        assert!(retained.get(SegmentId::from(0)).is_none());
        assert_eq!(retained.retained_bytes(), 0);
    }

    #[test]
    fn the_budget_bounds_the_bytes_kept() {
        let liveness = Liveness::new();
        let retained = RetainedArrays::new(liveness.clone(), 20);
        let _claim = liveness.claim(0..100);
        retained.insert(SegmentId::from(0), &array(), 12, 0..10);
        retained.insert(SegmentId::from(1), &array(), 12, 10..20);
        assert!(retained.get(SegmentId::from(0)).is_some());
        assert!(retained.get(SegmentId::from(1)).is_none());
        assert_eq!(retained.retained_bytes(), 12);
    }
}
