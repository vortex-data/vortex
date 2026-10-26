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
use vortex_session::VortexSession;

use super::Concat;
use super::Eval;
use super::LegacyStats;
use super::Pack;
use super::PlanRef;
use super::Take;
use super::Zoned;
use super::plans::EvalIdentityRule;
use super::plans::ExpressionConcatRule;
use super::plans::ExpressionPackRule;
use super::plans::ExpressionTakeRule;
use super::plans::ExpressionZonedRule;
use super::plans::LegacyStatsRule;
use super::plans::ZonedLegacyStatsRule;
use crate::plan::physical::plans::EvalEvalRule;
use crate::plan::physical::plans::EvalPackRule;

static EVAL_IDENTITY_RULE: PlanReduceRuleAdapter<Eval, EvalIdentityRule> =
    PlanReduceRuleAdapter::new(EvalIdentityRule);

static LEGACY_STATS_RULE: PlanReduceRuleAdapter<LegacyStats, LegacyStatsRule> =
    PlanReduceRuleAdapter::new(LegacyStatsRule);

static PLAN_RULES: PlanRuleSet = PlanRuleSet::new(&[
    &EVAL_IDENTITY_RULE,
    &PlanReduceRuleAdapter::new(EvalPackRule),
    &LEGACY_STATS_RULE,
]);

static EXPRESSION_CONCAT_RULE: PlanParentReduceRuleAdapter<Concat, ExpressionConcatRule> =
    PlanParentReduceRuleAdapter::new(ExpressionConcatRule);
static EXPRESSION_TAKE_RULE: PlanParentReduceRuleAdapter<Take, ExpressionTakeRule> =
    PlanParentReduceRuleAdapter::new(ExpressionTakeRule);
static EXPRESSION_PACK_RULE: PlanParentReduceRuleAdapter<Pack, ExpressionPackRule> =
    PlanParentReduceRuleAdapter::new(ExpressionPackRule);
static EXPRESSION_ZONED_RULE: PlanParentReduceRuleAdapter<Zoned, ExpressionZonedRule> =
    PlanParentReduceRuleAdapter::new(ExpressionZonedRule);

static ZONED_LEGACY_STATS_RULE: PlanParentReduceRuleAdapter<LegacyStats, ZonedLegacyStatsRule> =
    PlanParentReduceRuleAdapter::new(ZonedLegacyStatsRule);

static PARENT_RULES: PlanParentRuleSet = PlanParentRuleSet::new(&[
    &PlanParentReduceRuleAdapter::new(EvalEvalRule),
    &EXPRESSION_CONCAT_RULE,
    &EXPRESSION_TAKE_RULE,
    &EXPRESSION_PACK_RULE,
    &EXPRESSION_ZONED_RULE,
    &ZONED_LEGACY_STATS_RULE,
]);

/// Attempts a static rewrite for `plan`.
pub(crate) fn reduce_plan(
    plan: &PlanRef,
    session: &VortexSession,
) -> VortexResult<Option<PlanRef>> {
    PLAN_RULES.evaluate(plan, session)
}

/// Attempts a static rewrite for `parent` and its child at `child_idx`.
pub(crate) fn reduce_parent(
    parent: &PlanRef,
    child_idx: usize,
    session: &VortexSession,
) -> VortexResult<Option<PlanRef>> {
    let Some(child) = parent.children().get_ref(child_idx)? else {
        return Ok(None);
    };
    PARENT_RULES.evaluate(child, parent, child_idx, session)
}
