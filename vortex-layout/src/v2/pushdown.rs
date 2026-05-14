// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Pushdown rules over the layout plan tree.
//!
//! See `LAYOUT_PLAN.md` § PlanPushdownRule.

use vortex_error::VortexResult;

use crate::v2::matcher::Matcher;
use crate::v2::plan::LayoutPlanRef;

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

pub enum RewriteResult {
    /// The rule did not apply. The plan is unchanged.
    Unchanged,
    /// The rule rewrote the subtree.
    Rewritten(LayoutPlanRef),
}
