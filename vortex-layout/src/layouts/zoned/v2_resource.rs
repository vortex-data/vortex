// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`ZoneMapResource`] — pull-based [`Resource`] backing zone-level
//! pruning for the LayoutPlan v2 path.
//!
//! ## Lifecycle
//!
//! - Construction: [`ZonedLayout::plan`](super::ZonedLayout) builds
//!   one of these per filter conjunct that has a falsifiable
//!   pruning predicate. The resource is registered on the plan-time
//!   [`PlanCtx::resources`](crate::v2::plan::PlanCtx) collector and
//!   also held as an `Arc` by the
//!   [`ZonedPruningPlan`](super::v2_plan::ZonedPruningPlan) that
//!   consumes it.
//! - `ensure_ready`: reads the zones table once via the inner
//!   `zones_plan`, evaluates the falsified predicate against those
//!   stats, and caches both the per-zone prune mask (for the
//!   conjunct's bool-stream consumer) and an expanded per-row
//!   demand mask (for the [`DemandSource`] interface).
//! - Pulls: synchronous after `ensure_ready`. The cached masks are
//!   sliced for the requested range.
//!
//! ## What's *not* here yet
//!
//! Dynamic re-evaluation. The current impl computes once and never
//! advances `version()`. When dynamic-input watching lands, the
//! resource will re-falsify against fresh inputs and bump version on
//! change. Pull-based `RowDemand` will pick that up automatically.

use std::ops::Range;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use futures::FutureExt;
use futures::future::BoxFuture;
use once_cell::sync::OnceCell;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::StructArray;
use vortex_array::expr::Expression;
use vortex_array::stream::ArrayStreamExt;
use vortex_buffer::BitBufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use crate::layouts::zoned::zone_map::ZoneMap;
use crate::v2::dataflow::OutputFrontier;
use crate::v2::demand::DemandSource;
use crate::v2::demand::Resource;
use crate::v2::demand::RowDemand;
use crate::v2::plan::LayoutPlanRef;
use crate::v2::scan_ctx::ScanCtx;

/// Zone-map-backed [`Resource`] + [`DemandSource`].
///
/// Holds the inputs needed to compute the per-zone prune mask
/// (zones plan, falsified predicate, zone geometry, session) and a
/// cache of the computed per-zone + per-row masks.
pub struct ZoneMapResource {
    /// Sub-plan that materialises the zones stats table. Run once
    /// at `ensure_ready`.
    zones_plan: LayoutPlanRef,
    /// Falsified pruning expression (evaluated against the zones
    /// table). True for a zone means "this zone is prunable" —
    /// no row in it can match the original filter.
    pruning_predicate: Expression,
    zone_len: u64,
    row_count: u64,
    session: VortexSession,

    /// Populated by `ensure_ready`. Subsequent pulls slice into this.
    computed: OnceCell<Computed>,
    /// Bumped each time `computed` advances (today: 0 → 1 once).
    version: AtomicU64,
}

struct Computed {
    /// Per-zone bool, length `nzones`. `true` ⇒ zone is prunable.
    /// Used by the conjunct's bool-stream emitter.
    zone_prune_mask: Mask,
    /// Per-row "still demanded" bool, length `row_count`. Bit is
    /// `false` for any row inside a prunable zone, `true` otherwise.
    /// This is what `DemandSource::mask_for` returns.
    row_demand_mask: Mask,
}

impl ZoneMapResource {
    /// Construct a resource. State is not populated until
    /// `ensure_ready` is awaited.
    pub fn new(
        zones_plan: LayoutPlanRef,
        pruning_predicate: Expression,
        zone_len: u64,
        row_count: u64,
        session: VortexSession,
    ) -> Self {
        debug_assert!(zone_len > 0, "ZoneMapResource requires zone_len > 0");
        Self {
            zones_plan,
            pruning_predicate,
            zone_len,
            row_count,
            session,
            computed: OnceCell::new(),
            version: AtomicU64::new(0),
        }
    }

    pub fn zone_len(&self) -> u64 {
        self.zone_len
    }

    pub fn row_count(&self) -> u64 {
        self.row_count
    }

    /// Returns the per-zone prune mask. Errors if `ensure_ready` has
    /// not yet completed; callers in the body subtree are guaranteed
    /// it has completed because `ScanPlan` awaits it before invoking
    /// the body.
    pub fn zone_prune_mask(&self) -> VortexResult<Mask> {
        Ok(self
            .computed
            .get()
            .ok_or_else(|| vortex_err!("ZoneMapResource not ready"))?
            .zone_prune_mask
            .clone())
    }
}

impl Resource for ZoneMapResource {
    fn version(&self) -> u64 {
        self.version.load(Ordering::Acquire)
    }

    fn ensure_ready(&self) -> BoxFuture<'_, VortexResult<()>> {
        async move {
            // Fast path: already populated.
            if self.computed.get().is_some() {
                return Ok(());
            }

            // Read the zones table. The zones plan is in zone-coords
            // (one row per zone), unrelated to the partition row
            // space — pass an empty demand. The scratch `ScanCtx` we
            // construct here is sufficient for the typical Auxiliary
            // zones-leaf reads; if a zones plan ever needed shared
            // per-scan state (LetRegistry, etc.) this would need to
            // be plumbed differently.
            let zones_row_count = self
                .zones_plan
                .partition_stats(0)
                .map(|s| s.row_count())
                .unwrap_or(0);
            let zones_demand = RowDemand::empty(zones_row_count);
            let zones_frontier = OutputFrontier::unbounded(zones_row_count);
            let scratch_ctx = ScanCtx::new(self.session.clone());
            let zones_stream = self.zones_plan.execute(
                0..zones_row_count,
                &zones_demand,
                &zones_frontier,
                &scratch_ctx,
            )?;
            let zones_array = zones_stream.read_all().await?;
            let mut ctx_exec = self.session.create_execution_ctx();
            let zones_struct = zones_array.execute::<StructArray>(&mut ctx_exec)?;

            // Evaluate the falsified predicate per zone. SAFETY: the
            // ZonedLayout writer guarantees the zones struct's schema
            // matches the stats it claims to carry.
            let zone_map =
                unsafe { ZoneMap::new_unchecked(zones_struct, self.zone_len, self.row_count) };
            let zone_prune_mask = zone_map.prune(&self.pruning_predicate, &self.session)?;

            // Build the per-row demand mask (true = still demanded,
            // i.e. NOT inside a prunable zone).
            let row_demand_mask =
                expand_zones_to_rows(&zone_prune_mask, self.zone_len, self.row_count)?;

            // We don't serialise concurrent `ensure_ready` callers —
            // the work is idempotent. Race losers compute the same
            // answer; only the first `set` wins. Crucially, only the
            // winner bumps `version` — race losers must not, or
            // downstream `RowDemand` caches keyed by version would
            // invalidate on every ensure_ready call.
            if self
                .computed
                .set(Computed {
                    zone_prune_mask,
                    row_demand_mask,
                })
                .is_ok()
            {
                self.version.fetch_add(1, Ordering::AcqRel);
            }
            Ok(())
        }
        .boxed()
    }
}

impl DemandSource for ZoneMapResource {
    fn mask_for(&self, range: Range<u64>) -> BoxFuture<'_, VortexResult<Mask>> {
        async move {
            // Lazy-init: if no one has triggered the zones read yet,
            // do it now.
            self.ensure_ready().await?;
            let computed = self
                .computed
                .get()
                .ok_or_else(|| vortex_err!("ZoneMapResource ensure_ready did not populate"))?;
            let start = usize::try_from(range.start)?;
            let end = usize::try_from(range.end)?;
            Ok(computed.row_demand_mask.slice(start..end))
        }
        .boxed()
    }
}

/// Expand a per-zone prune mask into a per-row demand mask.
/// Per-row bit is `true` (demanded) when the zone's prune bit is
/// `false`, and vice versa.
fn expand_zones_to_rows(zone_prune: &Mask, zone_len: u64, row_count: u64) -> VortexResult<Mask> {
    let row_count_usize = usize::try_from(row_count)?;
    // Optimistic fast path: if no zones are prunable, every row is
    // still demanded.
    if zone_prune.true_count() == 0 {
        return Ok(Mask::new_true(row_count_usize));
    }
    let mut bits = BitBufferMut::new_set(row_count_usize);
    let nzones = zone_prune.len();
    for z in 0..nzones {
        if zone_prune.value(z) {
            let zr_start = (z as u64) * zone_len;
            let zr_end = (zr_start + zone_len).min(row_count);
            let s = usize::try_from(zr_start)?;
            let e = usize::try_from(zr_end)?;
            for i in s..e {
                bits.set_to(i, false);
            }
        }
    }
    Ok(Mask::from_buffer(bits.freeze()))
}
