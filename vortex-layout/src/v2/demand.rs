// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`RowDemand`] — a backwards-flowing SIP from `FilterPlan` to source
//! plans. Tells producers which rows have demand zero downstream, so
//! they can skip work for those rows.
//!
//! See `LAYOUT_PLAN.md` § SIPs / RowDemand.
//!
//! Today only the placeholder `RowDemand::empty()` is implemented —
//! `PlanContext` constructs one as a passthrough so the surface is
//! threaded through every layout. The producer/consumer side
//! (monotone-cell + watch-channel publish/subscribe) lands on top of
//! the stack along with its first real consumer.

use std::ops::Range;

/// A row range within a partition's row space. Half-open: `[start, end)`.
pub type RowRange = Range<u64>;

/// `RowDemand` is the partition-local resource that tracks which rows
/// still have demand. Producers will publish "these rows are no longer
/// needed"; consumers will query "is this range still needed?".
///
/// Monotone: bits only go `1 → 0` (rows newly known not to be needed).
/// Dropping the resource is always safe — producers just spend more
/// effort than necessary.
///
/// Today the publish/subscribe surface is unimplemented; this struct
/// is a stable handle so `PlanContext` can thread one resource per
/// scan through the plan tree without each layout having to know
/// whether the SIP is wired up yet.
pub struct RowDemand {}

impl RowDemand {
    /// Construct an empty `RowDemand`. The publish/subscribe surface
    /// is not yet wired up; this is the only constructor today.
    pub fn empty() -> Self {
        Self {}
    }
}
