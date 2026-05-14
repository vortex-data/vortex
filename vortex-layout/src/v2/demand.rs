// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`RowDemand`] — a backwards-flowing SIP from `FilterPlan` to source
//! plans. Tells producers which rows have demand zero downstream, so
//! they can skip work for those rows.
//!
//! See `LAYOUT_PLAN.md` § SIPs / RowDemand.
//!
//! This is the trait-level surface only. The monotone-cell /
//! watch-channel implementation lands in a later PR.

use std::ops::Range;
use std::sync::Arc;

use vortex_error::VortexResult;
use vortex_mask::Mask;

/// A row range within a partition's row space. Half-open: `[start, end)`.
pub type RowRange = Range<u64>;

/// `RowDemand` is the partition-local resource that tracks which rows
/// still have demand. Producers publish "these rows are no longer
/// needed"; consumers query "is this range still needed?".
///
/// Monotone: bits only go `1 → 0` (rows newly known not to be needed).
/// Dropping the resource is always safe — producers just spend more
/// effort than necessary.
pub struct RowDemand {
    // One window cell per fixed-size slice of the partition.
    // Implementation lands in a later PR.
    _spans: Vec<MonotoneMaskCell>,
}

impl RowDemand {
    /// Construct an empty `RowDemand` (no spans). Methods that take a
    /// `RowRange` will report "fully demanded" for any input.
    pub fn empty() -> Self {
        Self { _spans: vec![] }
    }

    /// Make a producer handle. Producers meet new masks into the
    /// cells covering a [`RowRange`].
    pub fn producer(self: &Arc<Self>) -> RowDemandProducer {
        RowDemandProducer {
            _demand: Arc::clone(self),
        }
    }

    /// Make a consumer handle. Consumers query the current demand
    /// state for arbitrary [`RowRange`]s.
    pub fn consumer(self: &Arc<Self>) -> RowDemandConsumer {
        RowDemandConsumer {
            _demand: Arc::clone(self),
        }
    }
}

/// One window cell. Stub for PR 1.
struct MonotoneMaskCell {}

/// Publishes demand reductions. Each call meets the published mask
/// into the relevant window cells.
pub struct RowDemandProducer {
    _demand: Arc<RowDemand>,
}

impl RowDemandProducer {
    /// Meet `mask` into the cells covering `range`. Monotone — bits
    /// only go `1 → 0`.
    pub fn publish(&self, _range: RowRange, _mask: Mask) -> VortexResult<()> {
        todo!("RowDemandProducer::publish — implemented in a later PR")
    }
}

/// Reads the current demand state. Queries are over arbitrary
/// [`RowRange`]s; the implementation gathers overlapping window cells.
///
/// The three reads (`is_empty`, `cardinality`, `snapshot`) form a
/// staircase of cost: cheap → medium → full. Sources should call them
/// in that order and bail out as early as the answer allows.
pub struct RowDemandConsumer {
    _demand: Arc<RowDemand>,
}

impl RowDemandConsumer {
    /// True iff no row in `range` has any remaining demand. Cheapest
    /// check; sources should call this first before doing any per-row
    /// work.
    pub fn is_empty(&self, _range: RowRange) -> bool {
        todo!("RowDemandConsumer::is_empty — implemented in a later PR")
    }

    /// Popcount of demand over `range`. Lets sources branch between
    /// dense and sparse evaluation paths.
    pub fn cardinality(&self, _range: RowRange) -> u64 {
        todo!("RowDemandConsumer::cardinality — implemented in a later PR")
    }

    /// The current demand mask sliced to `range`, along with a
    /// monotonically-increasing version number. Cheap when the
    /// underlying windows are unchanged since the last snapshot.
    pub fn snapshot(&self, _range: RowRange) -> (u64, Arc<Mask>) {
        todo!("RowDemandConsumer::snapshot — implemented in a later PR")
    }

    /// Wait until any window overlapping `range` is tighter than
    /// `version`. Watch-channel semantics.
    pub async fn wait_for_newer(&self, _range: RowRange, _version: u64) {
        todo!("RowDemandConsumer::wait_for_newer — implemented in a later PR")
    }
}
