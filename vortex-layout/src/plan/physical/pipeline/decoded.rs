// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Decoded segments shared by every scan of a file while live work still covers their readers'
//! rows.

use std::ops::Range;
use std::sync::Arc;
use std::sync::Weak;

use parking_lot::Mutex;
use rustc_hash::FxHashMap;
use vortex_array::ArrayRef;
use vortex_scan::planning::driver::Lifetime;
use vortex_scan::planning::driver::Liveness;

use crate::segments::SegmentId;

/// Whole decoded segments, shared by the filter, pruning, and projection pipelines of every
/// chunk of a file while the driver reports live work over the rows of a split that read them.
///
/// A segment that a chunk's filter and projection both read, or that neighbouring chunks
/// overlap, is otherwise decoded once per reader. Here the first reader's decode is kept, pinned
/// in the file's [`Liveness`] to the root rows of the split that read it. Every later reader that
/// takes the decode pins it to its own split's rows as well, and the decode is dropped once none
/// of those rows is claimed by live work, so the cache holds no more than the work in flight can
/// still use.
///
/// The default cache keeps nothing.
#[derive(Clone, Default)]
pub struct DecodedSegments(Option<Arc<Inner>>);

struct Inner {
    liveness: Liveness,
    entries: Mutex<Entries>,
}

#[derive(Default)]
struct Entries {
    decodes: FxHashMap<SegmentId, Entry>,
    next_generation: u64,
}

struct Entry {
    array: ArrayRef,
    generation: u64,
    /// The live pins holding the decode.
    pins: usize,
}

/// One reader's pin on a decode. Dropping the last pin of a decode removes it.
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
        let evicted = {
            let mut entries = inner.entries.lock();
            match entries.decodes.get_mut(&self.segment) {
                Some(entry) if entry.generation == self.generation => {
                    entry.pins -= 1;
                    if entry.pins == 0 {
                        entries.decodes.remove(&self.segment)
                    } else {
                        None
                    }
                }
                _ => None,
            }
        };
        drop(evicted);
    }
}

impl DecodedSegments {
    /// A cache whose decodes are kept while `liveness` reports live work over their readers'
    /// rows.
    pub fn new(liveness: Liveness) -> Self {
        Self(Some(Arc::new(Inner {
            liveness,
            entries: Mutex::default(),
        })))
    }

    /// The decoded segment an earlier reader kept, now also kept for a reader of `rows`.
    pub(crate) fn get(&self, segment: SegmentId, rows: &Range<u64>) -> Option<ArrayRef> {
        let inner = self.0.as_ref()?;
        let (array, generation) = {
            let mut entries = inner.entries.lock();
            let entry = entries.decodes.get_mut(&segment)?;
            entry.pins += 1;
            (entry.array.clone(), entry.generation)
        };
        inner.pin(segment, generation, rows);
        Some(array)
    }

    /// Keeps `array`, the decode of `segment` by a reader of `rows`, while live work claims those
    /// rows, and returns the decode readers should use: an earlier reader's when two race.
    pub(crate) fn insert(
        &self,
        segment: SegmentId,
        array: ArrayRef,
        rows: &Range<u64>,
    ) -> ArrayRef {
        let Some(inner) = self.0.as_ref() else {
            return array;
        };
        let (array, generation) = {
            let mut entries = inner.entries.lock();
            let generation = entries.next_generation;
            match entries.decodes.get_mut(&segment) {
                Some(entry) => {
                    entry.pins += 1;
                    (entry.array.clone(), entry.generation)
                }
                None => {
                    entries.next_generation += 1;
                    entries.decodes.insert(
                        segment,
                        Entry {
                            array: array.clone(),
                            generation,
                            pins: 1,
                        },
                    );
                    (array, generation)
                }
            }
        };
        inner.pin(segment, generation, rows);
        array
    }
}

impl Inner {
    /// Pins a counted reader of `segment`'s decode to `rows`. Pinning unclaimed rows drops the
    /// pin at once, which releases the count again; the entries lock must not be held, as the
    /// pin takes it.
    fn pin(self: &Arc<Self>, segment: SegmentId, generation: u64, rows: &Range<u64>) {
        self.liveness.pin(
            Lifetime::Rows(rows.clone()),
            Pin {
                inner: Arc::downgrade(self),
                segment,
                generation,
            },
        );
    }
}
