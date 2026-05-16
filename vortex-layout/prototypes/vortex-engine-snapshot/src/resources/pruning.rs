//! Zone-map side-channel between three operators of one Zoned
//! binding:
//!
//! - **`ZoneMapSink`** drains the zones subgraph's batches, builds
//!   a Vortex [`ZoneMap`], evaluates the lowered pruning predicate
//!   against it, and publishes the resulting per-zone mask.
//! - **`ZoneMapOperator`** sits on the data side: input = data
//!   batches, output = data batches with a refined `demand`
//!   reflecting the latest pruning. Holds a clone of the
//!   `Arc<ZoneMapResource>`.
//! - **`ZoneOperator`** (future, for the "exact" case): reads the
//!   resource and short-circuits when zone stats fully answer the
//!   expression — emits a derived batch and propagates `NotNeeded`
//!   for the entire data subgraph.
//!
//! The resource is a private `Arc`-shared handle between exactly
//! these operators of a single Zoned binding. It does NOT live in
//! the engine's `Resource` enum — that surface stays for shared
//! cross-graph state (dynamic-filter results, etc.). When the
//! resource model needs unification, this would migrate to a
//! `dyn Resource` trait — until then it's a focused side-channel.

use std::ops::Range;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use parking_lot::RwLock;
use vortex_array::expr::Expression;
use vortex_buffer::BitBufferMut;
use vortex_layout::layouts::zoned::zone_map::ZoneMap;
use vortex_mask::Mask;

/// A zone-map cache and pruning result, shared across the
/// operators of one Zoned binding.
pub struct ZoneMapResource {
    /// Bumped on every state change (zone-map built, mask
    /// refreshed). Readers compare against `last_seen_version` to
    /// short-circuit unchanged reads.
    version: AtomicU64,
    /// Lowered pruning expression, computed once at bind time via
    /// `checked_pruning_expr`. Evaluated against the zone map's
    /// stat-table struct array; `true` per zone means *prune*.
    pruning_predicate: Expression,
    /// Cumulative row offset per zone — `zone_row_offsets[i]` is
    /// the row at which zone `i` starts. Length == zone_count + 1.
    /// Set at construction; never changes.
    zone_row_offsets: Vec<u64>,
    state: RwLock<ZoneMapState>,
}

#[derive(Default)]
struct ZoneMapState {
    /// Vortex zone map. `None` until the sink has finished
    /// draining + building. Once `Some`, can be re-evaluated
    /// against new dynamic versions of the predicate.
    zone_map: Option<ZoneMap>,
    /// Latest per-zone prune mask. Length == zone_count.
    /// `true` at index `i` means zone `i` cannot match (skip).
    /// `None` until first evaluation.
    pruning_mask: Option<Mask>,
}

impl ZoneMapResource {
    pub fn new(pruning_predicate: Expression, zone_row_offsets: Vec<u64>) -> Self {
        debug_assert!(
            zone_row_offsets.len() >= 2,
            "ZoneMapResource needs at least one zone (offsets length >= 2)"
        );
        Self {
            version: AtomicU64::new(0),
            pruning_predicate,
            zone_row_offsets,
            state: RwLock::new(ZoneMapState::default()),
        }
    }

    pub fn zone_count(&self) -> usize {
        self.zone_row_offsets.len() - 1
    }

    pub fn total_rows(&self) -> u64 {
        *self.zone_row_offsets.last().unwrap_or(&0)
    }

    pub fn version(&self) -> u64 {
        self.version.load(Ordering::Acquire)
    }

    pub fn pruning_predicate(&self) -> &Expression {
        &self.pruning_predicate
    }

    /// Borrow the cumulative zone-row-offsets list. Useful for
    /// downstream operators that need to translate row positions
    /// into zone indices for fine-grained per-zone reasoning.
    pub fn zone_row_offsets_for_lookup(&self) -> &[u64] {
        &self.zone_row_offsets
    }

    /// Sink-side: install the freshly built `ZoneMap`. Bumps the
    /// version so readers know to re-pull.
    pub fn install_zone_map(&self, zone_map: ZoneMap) {
        {
            let mut state = self.state.write();
            state.zone_map = Some(zone_map);
            // Mask is invalidated until the sink calls
            // `refresh_mask`.
            state.pruning_mask = None;
        }
        self.version.fetch_add(1, Ordering::Release);
    }

    /// Sink-side: store the freshly evaluated prune mask. Length
    /// must equal `zone_count()`.
    pub fn install_mask(&self, mask: Mask) {
        debug_assert_eq!(
            mask.len(),
            self.zone_count(),
            "install_mask: mask length must equal zone count"
        );
        {
            let mut state = self.state.write();
            state.pruning_mask = Some(mask);
        }
        self.version.fetch_add(1, Ordering::Release);
    }

    /// Reader-side: borrow the current zone map for evaluation.
    /// Returns `None` if the sink hasn't finished building yet.
    pub fn with_zone_map<R>(&self, f: impl FnOnce(&ZoneMap) -> R) -> Option<R> {
        let state = self.state.read();
        state.zone_map.as_ref().map(f)
    }

    /// Reader-side: per-row demand mask (`true` = keep,
    /// `false` = pruned) for `row_range`, returned only if the
    /// resource's version is `> since_version` (i.e. has refined
    /// since the caller last looked).
    ///
    /// Fast paths:
    ///   * empty range → `Mask::new_true(0)`.
    ///   * pruning mask all-false (nothing pruned globally) →
    ///     `Mask::new_true(len)`.
    ///   * pruning mask all-true (everything pruned globally) →
    ///     `Mask::new_false(len)`.
    ///   * single uniform overlap (every zone touching `row_range`
    ///     is kept, or every zone is pruned) → AllTrue / AllFalse
    ///     without ever allocating a bit buffer.
    ///
    /// Mixed kept/pruned: builds the bit buffer directly with
    /// `BitBufferMut::append_n`, which writes whole words at once
    /// — strictly faster than going through `Mask::from_slices`
    /// (which uses per-bit `set` calls) for any non-trivial slice.
    pub fn demand_for_range(
        &self,
        row_range: Range<u64>,
        since_version: u64,
    ) -> Option<(u64, Mask)> {
        let v = self.version();
        if v <= since_version {
            return None;
        }
        let state = self.state.read();
        let pruning = state.pruning_mask.as_ref()?;
        let len = usize::try_from(row_range.end.saturating_sub(row_range.start)).ok()?;
        if len == 0 {
            return Some((v, Mask::new_true(0)));
        }
        if pruning.all_false() {
            return Some((v, Mask::new_true(len)));
        }
        if pruning.all_true() {
            return Some((v, Mask::new_false(len)));
        }

        // First sweep: detect the uniform-overlap case (every zone
        // intersecting `row_range` is kept, or every one is pruned)
        // before allocating anything. q20-shaped batches typically
        // land entirely inside one zone, so this is the hot path.
        let zone_count = self.zone_count();
        let zi_first = self
            .zone_row_offsets
            .partition_point(|&o| o <= row_range.start)
            .saturating_sub(1);
        let mut any_kept = false;
        let mut any_pruned = false;
        {
            let mut zi = zi_first;
            while zi < zone_count {
                let z_start = self.zone_row_offsets[zi];
                if z_start >= row_range.end {
                    break;
                }
                if pruning.value(zi) {
                    any_pruned = true;
                } else {
                    any_kept = true;
                }
                if any_kept && any_pruned {
                    break;
                }
                zi += 1;
            }
        }
        if !any_pruned {
            return Some((v, Mask::new_true(len)));
        }
        if !any_kept {
            return Some((v, Mask::new_false(len)));
        }

        // Mixed: build the bit buffer directly with `append_n`.
        // Each call writes a contiguous run of N identical bits
        // word-at-a-time, strictly faster than per-bit `set`.
        let mut buf = BitBufferMut::with_capacity(len);
        let mut cursor: usize = 0;
        let mut zi = zi_first;
        while zi < zone_count {
            let z_start = self.zone_row_offsets[zi];
            let z_end = self.zone_row_offsets[zi + 1];
            if z_start >= row_range.end {
                break;
            }
            let lo = z_start.max(row_range.start);
            let hi = z_end.min(row_range.end);
            let lo_u = (lo - row_range.start) as usize;
            let hi_u = (hi - row_range.start) as usize;
            // Gap before this zone (rows before zone 0 or between
            // non-contiguous zones). No zone evidence ⇒ keep.
            if lo_u > cursor {
                buf.append_n(true, lo_u - cursor);
            }
            // `pruning.value(zi) == true` means zone is pruned.
            // Output mask bit is the inverse: true = kept.
            buf.append_n(!pruning.value(zi), hi_u - lo_u);
            cursor = hi_u;
            zi += 1;
        }
        // Tail past the last covering zone — keep by default.
        if cursor < len {
            buf.append_n(true, len - cursor);
        }
        Some((v, Mask::from_buffer(buf.freeze())))
    }

    /// Reader-side: are *all* zones overlapping `row_range`
    /// pruned? Used by `propagate_requirements` to mark whole
    /// chunks `NotNeeded` upstream.
    pub fn is_range_fully_pruned(&self, row_range: Range<u64>) -> bool {
        if self.version() == 0 {
            return false;
        }
        let state = self.state.read();
        let Some(pruning) = state.pruning_mask.as_ref() else {
            return false;
        };
        if pruning.all_false() {
            return false;
        }
        let zone_count = self.zone_count();
        let mut zi = self
            .zone_row_offsets
            .partition_point(|&o| o <= row_range.start)
            .saturating_sub(1);
        let mut any = false;
        while zi < zone_count {
            let z_start = self.zone_row_offsets[zi];
            if z_start >= row_range.end {
                break;
            }
            if !pruning.value(zi) {
                return false;
            }
            any = true;
            zi += 1;
        }
        any
    }
}
