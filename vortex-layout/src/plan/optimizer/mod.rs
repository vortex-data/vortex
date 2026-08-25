// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Static rewrite rules for physical plans.

mod rules;

pub use rules::DynPlanParentReduceRule;
pub use rules::DynPlanReduceRule;
pub use rules::PlanParentReduceRule;
pub use rules::PlanParentReduceRuleAdapter;
pub use rules::PlanParentRuleSet;
pub use rules::PlanReduceRule;
pub use rules::PlanReduceRuleAdapter;
pub use rules::PlanRuleSet;
use vortex_error::VortexResult;

use super::Concat;
use super::Eval;
use super::Pack;
use super::PlanRef;
use super::Take;
use super::plans::EvalIdentityRule;
use super::plans::ExpressionConcatRule;
use super::plans::ExpressionPackRule;
use super::plans::ExpressionTakeRule;

static EVAL_IDENTITY_RULE: PlanReduceRuleAdapter<Eval, EvalIdentityRule> =
    PlanReduceRuleAdapter::new(EvalIdentityRule);

static PLAN_RULES: PlanRuleSet = PlanRuleSet::new(&[&EVAL_IDENTITY_RULE]);

static EXPRESSION_CONCAT_RULE: PlanParentReduceRuleAdapter<Concat, ExpressionConcatRule> =
    PlanParentReduceRuleAdapter::new(ExpressionConcatRule);
static EXPRESSION_TAKE_RULE: PlanParentReduceRuleAdapter<Take, ExpressionTakeRule> =
    PlanParentReduceRuleAdapter::new(ExpressionTakeRule);
static EXPRESSION_PACK_RULE: PlanParentReduceRuleAdapter<Pack, ExpressionPackRule> =
    PlanParentReduceRuleAdapter::new(ExpressionPackRule);

static PARENT_RULES: PlanParentRuleSet = PlanParentRuleSet::new(&[
    &EXPRESSION_CONCAT_RULE,
    &EXPRESSION_TAKE_RULE,
    &EXPRESSION_PACK_RULE,
]);

/// Attempts a static rewrite for `plan`.
pub(crate) fn reduce_plan(plan: &PlanRef) -> VortexResult<Option<PlanRef>> {
    PLAN_RULES.evaluate(plan)
}

/// Attempts a static rewrite for `parent` and its child at `child_idx`.
pub(crate) fn reduce_parent(parent: &PlanRef, child_idx: usize) -> VortexResult<Option<PlanRef>> {
    let Some(child) = parent.child(child_idx)? else {
        return Ok(None);
    };
    PARENT_RULES.evaluate(&child, parent, child_idx)
}
