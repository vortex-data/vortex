// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Static parent-child rewrite rules for physical plans.

mod rules;

pub use rules::DynPlanParentReduceRule;
pub use rules::PlanParentReduceRule;
pub use rules::PlanParentReduceRuleAdapter;
pub use rules::PlanParentRuleSet;
use vortex_error::VortexResult;

use super::Concat;
use super::Pack;
use super::PlanRef;
use super::RowIdx;
use super::Take;
use super::Zoned;
use super::plans::ExpressionConcatRule;
use super::plans::ExpressionPackRule;
use super::plans::ExpressionRowIdxRule;
use super::plans::ExpressionTakeRule;
use super::plans::ExpressionZonedRule;

static EXPRESSION_CONCAT_RULE: PlanParentReduceRuleAdapter<Concat, ExpressionConcatRule> =
    PlanParentReduceRuleAdapter::new(ExpressionConcatRule);
static EXPRESSION_TAKE_RULE: PlanParentReduceRuleAdapter<Take, ExpressionTakeRule> =
    PlanParentReduceRuleAdapter::new(ExpressionTakeRule);
static EXPRESSION_ROW_IDX_RULE: PlanParentReduceRuleAdapter<RowIdx, ExpressionRowIdxRule> =
    PlanParentReduceRuleAdapter::new(ExpressionRowIdxRule);
static EXPRESSION_PACK_RULE: PlanParentReduceRuleAdapter<Pack, ExpressionPackRule> =
    PlanParentReduceRuleAdapter::new(ExpressionPackRule);
static EXPRESSION_ZONED_RULE: PlanParentReduceRuleAdapter<Zoned, ExpressionZonedRule> =
    PlanParentReduceRuleAdapter::new(ExpressionZonedRule);

static PARENT_RULES: PlanParentRuleSet = PlanParentRuleSet::new(&[
    &EXPRESSION_CONCAT_RULE,
    &EXPRESSION_TAKE_RULE,
    &EXPRESSION_ROW_IDX_RULE,
    &EXPRESSION_PACK_RULE,
    &EXPRESSION_ZONED_RULE,
]);

/// Attempts a static rewrite for `parent` and its child at `child_idx`.
pub(crate) fn reduce_parent(parent: &PlanRef, child_idx: usize) -> VortexResult<Option<PlanRef>> {
    let Some(child) = parent.child(child_idx)? else {
        return Ok(None);
    };
    PARENT_RULES.evaluate(&child, parent, child_idx)
}
