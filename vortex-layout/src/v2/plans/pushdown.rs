// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Pushdown rules over the layout plan tree.
//!
//! See `LAYOUT_PLAN.md` § PlanPushdownRule.
//!
//! Parking-lot module: defines the trait surface that the design doc
//! describes, but no rules implement it yet. The current pushdown
//! work is per-node `LayoutPlan::try_pushdown_mask`. This module
//! lives here so future rule-based pushdown can grow without
//! re-introducing the surface.

use vortex_error::VortexResult;

use crate::v2::plans::LayoutPlanRef;
use crate::v2::plans::matcher::Matcher;

/// A rule that pattern-matches some parent shape in the plan tree and
/// produces a rewritten subtree. Rules are stateless; they cannot
/// inspect or mutate `RowDemand` (that's runtime state) — they only
/// rewrite plan structure.
pub trait PlanPushdownRule {
    /// The parent shape this rule applies to.
    type Parent: Matcher;

    /// Try to rewrite `parent`. Returns [`RewriteResult::Unchanged`]
    /// if the rule does not apply.
    fn rewrite(&self, parent: LayoutPlanRef) -> VortexResult<RewriteResult>;
}

/// Outcome of one [`PlanPushdownRule::rewrite`] call.
pub enum RewriteResult {
    /// The rule did not apply. The plan is unchanged.
    Unchanged,
    /// The rule rewrote the subtree.
    Rewritten(LayoutPlanRef),
}
