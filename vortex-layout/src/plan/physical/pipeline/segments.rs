// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Whole encoded arrays retained until their row consumers finish.

use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::Arc;

use once_cell::sync::OnceCell;
use parking_lot::Mutex;
use rustc_hash::FxHashMap;
use vortex_array::ArrayRef;
use vortex_error::VortexResult;

use crate::plan::physical::PlanByPtr;
use crate::plan::physical::PlanRef;

#[derive(Clone)]
pub(crate) struct SegmentCache {
    shared: Arc<SharedCache>,
}

struct SharedCache {
    state: Mutex<Cache>,
    reuse_whole: bool,
}

struct Cache {
    /// Disjoint unfinished intervals, including chunks that have not been planned yet.
    pending: BTreeMap<u64, u64>,
    arrays: FxHashMap<(PlanByPtr, u64), Entry>,
}

struct Entry {
    rows: Range<u64>,
    array: Arc<OnceCell<ArrayRef>>,
}

fn overlaps(pending: &BTreeMap<u64, u64>, rows: &Range<u64>) -> bool {
    pending
        .range(..rows.end)
        .next_back()
        .is_some_and(|(_, end)| *end > rows.start)
}

impl SegmentCache {
    pub(crate) fn new(rows: Range<u64>, reuse_whole: bool) -> Self {
        let pending = if rows.is_empty() {
            BTreeMap::new()
        } else {
            BTreeMap::from([(rows.start, rows.end)])
        };
        Self {
            shared: Arc::new(SharedCache {
                state: Mutex::new(Cache {
                    pending,
                    arrays: FxHashMap::default(),
                }),
                reuse_whole,
            }),
        }
    }

    pub(crate) fn reuse_whole(&self) -> bool {
        self.shared.reuse_whole
    }

    pub(crate) fn lease(&self, rows: Range<u64>) -> Arc<SegmentLease> {
        Arc::new(SegmentLease {
            cache: self.clone(),
            rows,
        })
    }

    pub(crate) fn reader(&self, plan: PlanRef, rows: Range<u64>) -> CachedSegment {
        CachedSegment {
            cache: self.clone(),
            key: (PlanByPtr(plan), rows.start),
            rows,
        }
    }

    /// Retires completed, pruned, or cancelled rows, even when chunks finish out of order.
    pub(crate) fn complete(&self, rows: Range<u64>) {
        if rows.is_empty() {
            return;
        }
        let mut cache = self.shared.state.lock();
        while let Some((&start, &end)) = cache.pending.range(..rows.end).next_back() {
            if end <= rows.start {
                break;
            }
            cache.pending.remove(&start);
            if end > rows.end {
                cache.pending.insert(rows.end, end);
            }
            if start < rows.start {
                cache.pending.insert(start, rows.start);
                break;
            }
        }
        let Cache { pending, arrays } = &mut *cache;
        arrays.retain(|_, entry| overlaps(pending, &entry.rows));
    }
}

/// Shared by a chunk's filters and projection morsels. Dropping the last consumer retires it.
pub(crate) struct SegmentLease {
    pub(crate) cache: SegmentCache,
    rows: Range<u64>,
}

impl Drop for SegmentLease {
    fn drop(&mut self) {
        self.cache.complete(self.rows.clone());
    }
}

pub(crate) struct CachedSegment {
    cache: SegmentCache,
    key: (PlanByPtr, u64),
    rows: Range<u64>,
}

impl CachedSegment {
    pub(crate) fn get(&self) -> Option<ArrayRef> {
        self.cache
            .shared
            .state
            .lock()
            .arrays
            .get(&self.key)
            .and_then(|entry| entry.array.get())
            .cloned()
    }

    /// Concurrent readers share initialization of this segment. Decoding holds no global cache
    /// lock, and a failed initialization leaves the entry available for another reader to retry.
    pub(crate) fn get_or_try_init(
        &self,
        initialize: impl FnOnce() -> VortexResult<ArrayRef>,
    ) -> VortexResult<ArrayRef> {
        let array = {
            let mut cache = self.cache.shared.state.lock();
            if !overlaps(&cache.pending, &self.rows) {
                drop(cache);
                return initialize();
            }
            Arc::clone(
                &cache
                    .arrays
                    .entry(self.key.clone())
                    .or_insert_with(|| Entry {
                        rows: self.rows.clone(),
                        array: Arc::new(OnceCell::new()),
                    })
                    .array,
            )
        };
        array.get_or_try_init(initialize).cloned()
    }
}

#[cfg(test)]
mod tests;
