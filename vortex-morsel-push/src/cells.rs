// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Shared decoded cells with demand-derived retention — the P1 slice of P2's keyed cells.
//!
//! This is deliberately **not a cache**. A cache decides what to keep with a budget and an
//! eviction heuristic, and holds data on the chance it is wanted again. A cell here is kept by
//! *leases*: before the scan starts, the driver counts, from the morsel cut alone, exactly how
//! many morsels will touch each stored unit. Each retiring morsel releases its lease whether it
//! used the cell or not, and the moment the count reaches zero the decoded array is dropped.
//! Nothing is retained speculatively, nothing survives the scan, and there is no budget because
//! there is nothing discretionary to budget: the set of live cells is a function of scan
//! progress, not of policy.
//!
//! The lease arithmetic mirrors planning exactly: a flat node registers a use for a morsel iff
//! the morsel's range overlaps its chunk, and the precomputed lease count for a unit is the
//! number of (node, morsel) pairs with that overlap. Every planned use is released at retire, so
//! the counts drain to zero by construction — an imbalance is a bug, not a leak policy.
//!
//! Because morsels are contiguous ranges taken off a monotone cursor, the morsels overlapping
//! one unit are consecutive indices; a cell is born at the first of them and dies at the last,
//! so the set of live cells tracks the active window of the scan.

use std::hash::BuildHasher;
use std::hash::RandomState;

use parking_lot::Mutex;
use vortex_array::ArrayRef;
use vortex_utils::aliases::hash_map::HashMap;

use crate::io::IoKey;

/// Shard count for the cell map. Lease traffic is one lookup per (node, morsel) use, which on a
/// wide table with per-split morsels is thousands of touches per scan; one lock measurably
/// serialises 4 threads, sixteen shards make collisions rare.
const SHARDS: usize = 16;

type Shard = Mutex<HashMap<IoKey, CellEntry>>;

struct CellEntry {
    /// Outstanding (node, morsel) uses that have not yet retired.
    leases: usize,
    /// The decoded array, published by the first morsel to decode this unit.
    decoded: Option<ArrayRef>,
}

/// Keyed decoded-value cells shared by every driving thread of one scan.
pub struct SharedCells {
    shards: Option<Box<[Shard]>>,
    hasher: RandomState,
}

impl SharedCells {
    /// A disabled cell layer: every lookup misses, publishes and releases are no-ops.
    ///
    /// This disables decoded-array reuse between morsels. The scan-wide raw IO service remains
    /// enabled and is tested independently.
    pub fn disabled() -> Self {
        Self {
            shards: None,
            hasher: RandomState::new(),
        }
    }

    /// Build the cell layer from precomputed lease counts.
    ///
    /// A unit touched by exactly one (node, morsel) pair can never be reused, so it is not
    /// registered at all: no lookup, no publish, no release. On a scan with no straddling and no
    /// column shared between filter and projection this leaves the map empty and the mechanism
    /// costs nothing, which measurably matters — the first version registered every unit and
    /// paid ~20% on a pure six-column scan for bookkeeping that could never pay off.
    pub fn with_leases(counts: HashMap<IoKey, usize>) -> Self {
        let hasher = RandomState::new();
        let mut shards: Vec<HashMap<IoKey, CellEntry>> =
            (0..SHARDS).map(|_| HashMap::default()).collect();
        for (key, count) in counts {
            if count > 1 {
                shards[usize::try_from(hasher.hash_one(key)).unwrap_or(0) % SHARDS].insert(
                    key,
                    CellEntry {
                        leases: count,
                        decoded: None,
                    },
                );
            }
        }
        let shards = if shards.iter().all(HashMap::is_empty) {
            None
        } else {
            Some(shards.into_iter().map(Mutex::new).collect())
        };
        Self { shards, hasher }
    }

    /// Whether the cell layer is enabled.
    pub fn is_enabled(&self) -> bool {
        self.shards.is_some()
    }

    fn shard(&self, key: IoKey) -> Option<&Shard> {
        let shards = self.shards.as_ref()?;
        Some(&shards[usize::try_from(self.hasher.hash_one(key)).unwrap_or(0) % SHARDS])
    }

    /// The decoded array for a unit, if some morsel has already published it.
    ///
    /// A hit is stable for the caller's whole morsel: the caller's own unreleased lease keeps
    /// the count positive until its retire, so the cell cannot be dropped underneath it.
    pub fn decoded(&self, key: IoKey) -> Option<ArrayRef> {
        self.shard(key)?
            .lock()
            .get(&key)
            .and_then(|entry| entry.decoded.clone())
    }

    /// Publish a decoded array for a unit. First writer wins; a publish for a unit with no
    /// outstanding leases (or an unknown unit) is dropped.
    pub fn publish(&self, key: IoKey, array: &ArrayRef) {
        let Some(shard) = self.shard(key) else {
            return;
        };
        let mut cells = shard.lock();
        if let Some(entry) = cells.get_mut(&key)
            && entry.leases > 0
            && entry.decoded.is_none()
        {
            entry.decoded = Some(array.clone());
        }
    }

    /// Release one lease on a unit, dropping the cell when the last lease goes.
    ///
    /// A key with no entry is a single-lease unit that was never registered, which is the common
    /// case on a well-aligned file; releasing it is a no-op rather than an error.
    pub fn release(&self, key: IoKey) {
        let Some(shard) = self.shard(key) else {
            return;
        };
        let mut cells = shard.lock();
        let Some(entry) = cells.get_mut(&key) else {
            return;
        };
        debug_assert!(entry.leases > 0, "released a lease past zero on {key:?}");
        entry.leases = entry.leases.saturating_sub(1);
        if entry.leases == 0 {
            cells.remove(&key);
        }
    }

    /// The number of live cells, for tests and diagnostics.
    pub fn live(&self) -> usize {
        self.shards
            .as_ref()
            .map(|shards| shards.iter().map(|shard| shard.lock().len()).sum())
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use vortex_layout::segments::SegmentId;

    use super::*;

    #[test]
    fn unique_leases_disable_cell_layer() {
        let mut counts = HashMap::default();
        counts.insert(IoKey::Segment(SegmentId::from(0)), 1);

        let cells = SharedCells::with_leases(counts);

        assert!(!cells.is_enabled());
    }

    #[test]
    fn shared_leases_enable_cell_layer() {
        let mut counts = HashMap::default();
        counts.insert(IoKey::Segment(SegmentId::from(0)), 2);

        let cells = SharedCells::with_leases(counts);

        assert!(cells.is_enabled());
    }
}
