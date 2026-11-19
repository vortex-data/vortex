// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Static rewrite rules for physical plans.

use vortex_error::VortexResult;
use vortex_session::VortexSession;

use super::rules::*;
use crate::plan::Concat;
use crate::plan::Eval;
use crate::plan::Pack;
use crate::plan::PlanRef;
use crate::plan::Take;
use crate::plan::Zoned;
use crate::plan::builtins::EvalEvalRule;
use crate::plan::builtins::EvalIdentityRule;
use crate::plan::builtins::EvalPackRule;
use crate::plan::builtins::ExpressionConcatRule;
use crate::plan::builtins::ExpressionPackRule;
use crate::plan::builtins::ExpressionTakeRule;
use crate::plan::builtins::ExpressionZonedRule;
use crate::plan::builtins::ZonedDataRule;

static EVAL_IDENTITY_RULE: PlanReduceRuleAdapter<Eval, EvalIdentityRule> =
    PlanReduceRuleAdapter::new(EvalIdentityRule);

static ZONED_DATA_RULE: PlanReduceRuleAdapter<Zoned, ZonedDataRule> =
    PlanReduceRuleAdapter::new(ZonedDataRule);

static PLAN_RULES: PlanRuleSet = PlanRuleSet::new(&[
    &EVAL_IDENTITY_RULE,
    &PlanReduceRuleAdapter::new(EvalPackRule),
    &ZONED_DATA_RULE,
]);

static EXPRESSION_CONCAT_RULE: PlanParentReduceRuleAdapter<Concat, ExpressionConcatRule> =
    PlanParentReduceRuleAdapter::new(ExpressionConcatRule);
static EXPRESSION_TAKE_RULE: PlanParentReduceRuleAdapter<Take, ExpressionTakeRule> =
    PlanParentReduceRuleAdapter::new(ExpressionTakeRule);
static EXPRESSION_PACK_RULE: PlanParentReduceRuleAdapter<Pack, ExpressionPackRule> =
    PlanParentReduceRuleAdapter::new(ExpressionPackRule);
static EXPRESSION_ZONED_RULE: PlanParentReduceRuleAdapter<Zoned, ExpressionZonedRule> =
    PlanParentReduceRuleAdapter::new(ExpressionZonedRule);
static PARENT_RULES: PlanParentRuleSet = PlanParentRuleSet::new(&[
    &PlanParentReduceRuleAdapter::new(EvalEvalRule),
    &EXPRESSION_CONCAT_RULE,
    &EXPRESSION_TAKE_RULE,
    &EXPRESSION_PACK_RULE,
    &EXPRESSION_ZONED_RULE,
]);

/// Attempts a static rewrite for `plan`.
pub(crate) fn reduce_plan(
    plan: &PlanRef,
    session: &VortexSession,
) -> VortexResult<Option<PlanRef>> {
    match PLAN_RULES.evaluate(plan, session)? {
        Some(plan) => Ok(Some(plan)),
        None => plan.reduce(session),
    }
}

/// Attempts a static rewrite for `parent` and `child`, its child in `slot`.
pub(crate) fn reduce_parent(
    child: &PlanRef,
    parent: &PlanRef,
    slot: usize,
    session: &VortexSession,
) -> VortexResult<Option<PlanRef>> {
    match PARENT_RULES.evaluate(child, parent, slot, session)? {
        Some(plan) => Ok(Some(plan)),
        None => child.reduce_parent(parent, slot, session),
    }
}
