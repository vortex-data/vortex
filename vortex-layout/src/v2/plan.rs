// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The [`LayoutPlan`] trait and its supporting types.
//!
//! A `LayoutPlan` is the unit of recursive plan-tree construction
//! returned by [`crate::Layout::plan`]. See `LAYOUT_PLAN.md` § Model.

use std::sync::Arc;

use vortex_array::dtype::DType;
use vortex_array::expr::Expression;
use vortex_array::stream::SendableArrayStream;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_scan::selection::Selection;
use vortex_session::VortexSession;

use crate::segments::SegmentSource;
use crate::v2::demand::RowDemand;

pub type LayoutPlanRef = Arc<dyn LayoutPlan>;

/// A node in a layout plan tree. Each node produces output in one
/// row domain, partitioned into `partition_count()` independent units
/// of execution.
///
/// See `LAYOUT_PLAN.md` § Model.
pub trait LayoutPlan: 'static + Send + Sync {
    /// The output schema of this plan node.
    fn schema(&self) -> &DType;

    /// Number of independent partitions this plan exposes. A partition
    /// is the unit of parallel execution; what it spans (rows, list
    /// elements, arbitrary blocks) is layout-defined.
    fn partition_count(&self) -> usize;

    /// Stats this plan can vouch for, per partition. Fields are
    /// `Option<_>` because a plan may not know its row count or byte
    /// size until execution. Returning a fully-unknown
    /// [`PartitionStats`] is always valid.
    fn partition_stats(&self, partition: usize) -> VortexResult<PartitionStats>;

    /// True iff this plan emits rows in the layout's natural row order:
    /// within a partition, rows in row-id order; across partitions,
    /// partitions in partition-id order.
    fn output_ordered(&self) -> bool;

    /// For each child, true iff this plan needs that child row-ordered.
    fn required_input_ordered(&self) -> Vec<bool>;

    /// For each child, true iff this plan's output preserves the
    /// child's ordering (i.e., reading our output gives back its rows
    /// in the same row order as the child produced them).
    fn maintains_input_order(&self) -> Vec<bool>;

    /// Coalesce or split partitions to match a target count. Default
    /// is `Err` unless `n == partition_count()`; layouts that can
    /// rebalance (typically `Chunked`) override.
    fn repartition(self: Arc<Self>, _n: usize) -> VortexResult<LayoutPlanRef> {
        Err(vortex_err!("repartition not supported by this layout plan"))
    }

    /// Children of this plan. Used by pushdown rules to walk the tree.
    fn children(&self) -> &[LayoutPlanRef];

    /// Rebuild this node with new children. Used by pushdown rules
    /// to produce a rewritten subtree without each node having to
    /// know how to reconstruct itself.
    fn with_new_children(
        self: Arc<Self>,
        children: Vec<LayoutPlanRef>,
    ) -> VortexResult<LayoutPlanRef>;

    /// Execute one partition. Returns a stream of arrays in this
    /// plan's output schema.
    fn execute(
        &self,
        partition: usize,
        session: &VortexSession,
    ) -> VortexResult<SendableArrayStream>;
}

/// Arguments passed to [`crate::Layout::plan`]. Carries the consumer's
/// row selection, the expression to evaluate against the layout, and
/// a [`PlanContext`] with cross-cutting handles.
#[derive(Clone)]
pub struct PlanArguments {
    pub selection: Selection,
    pub expr: Expression,
    pub ctx: PlanContext,
}

impl PlanArguments {
    /// Replace the expression while keeping selection and context.
    /// Used by layouts that rewrite the expression on the way down
    /// (e.g., `Struct` field routing, `Dict` predicate rewrite).
    pub fn with_expr(self, expr: Expression) -> Self {
        Self { expr, ..self }
    }
}

/// Cross-cutting context threaded through [`crate::Layout::plan`].
///
/// Carries:
/// - the [`RowDemand`] SIP resource for layout-emitted publishers and
///   downstream consumers,
/// - the [`SegmentSource`] used to fetch on-disk bytes at execute time,
/// - the [`VortexSession`] used for plan-time setup that touches the
///   array context (e.g. constructing `LayoutReader`s while we still
///   bridge to the v1 read path).
#[derive(Clone)]
pub struct PlanContext {
    pub demand: Arc<RowDemand>,
    pub segment_source: Arc<dyn SegmentSource>,
    pub session: VortexSession,
}

impl PlanContext {
    /// Construct a context with the given segment source and session, and
    /// an empty (no-op) [`RowDemand`]. Useful as a default for layouts
    /// that don't need to participate in demand publication.
    pub fn new(segment_source: Arc<dyn SegmentSource>, session: VortexSession) -> Self {
        Self {
            demand: Arc::new(RowDemand::empty()),
            segment_source,
            session,
        }
    }

    /// Replace this context's demand with a freshly-allocated, empty
    /// one. Useful for layouts that recurse into a sub-plan that
    /// shouldn't share demand with the parent (e.g., evaluating a
    /// pruning expression over a zone-map child).
    pub fn without_demand(&self) -> Self {
        Self {
            demand: Arc::new(RowDemand::empty()),
            segment_source: Arc::clone(&self.segment_source),
            session: self.session.clone(),
        }
    }
}

/// Stats a partition can vouch for. All fields are `Option<_>`; an
/// empty `PartitionStats` is always valid.
///
/// Per-column stats (min/max/null counts) will be added in a later
/// PR when there's a real consumer; for now the type carries row
/// count and byte-size estimates only.
#[derive(Default, Clone, Debug)]
pub struct PartitionStats {
    pub row_count: Option<u64>,
    pub byte_size_estimate: Option<u64>,
}

impl PartitionStats {
    pub fn unknown() -> Self {
        Self::default()
    }

    pub fn with_row_count(mut self, n: u64) -> Self {
        self.row_count = Some(n);
        self
    }

    pub fn with_byte_size_estimate(mut self, n: u64) -> Self {
        self.byte_size_estimate = Some(n);
        self
    }
}
