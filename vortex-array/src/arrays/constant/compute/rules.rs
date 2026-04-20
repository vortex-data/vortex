// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_session::registry::CachedId;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Constant;
use crate::arrays::ConstantArray;
use crate::arrays::Filter;
use crate::arrays::dict::TakeReduceAdaptor;
use crate::arrays::filter::FilterReduceAdaptor;
use crate::arrays::slice::SliceReduceAdaptor;
use crate::optimizer::rules::ArrayParentReduceRule;
use crate::optimizer::rules::ParentRuleDense;
use crate::optimizer::rules::ParentRuleEntry;
use crate::optimizer::rules::ParentRuleSet;
use crate::scalar_fn::fns::between::BetweenReduceAdaptor;
use crate::scalar_fn::fns::cast::CastReduceAdaptor;
use crate::scalar_fn::fns::fill_null::FillNullReduceAdaptor;
use crate::scalar_fn::fns::not::NotReduceAdaptor;

static KEYED_PARENT_RULES: [ParentRuleEntry<Constant>; 6] = [
    ParentRuleSet::lift_id(CachedId::new("vortex.cast"), &CastReduceAdaptor(Constant)),
    ParentRuleSet::lift_id(CachedId::new("vortex.filter"), &ConstantFilterRule),
    ParentRuleSet::lift_id(
        CachedId::new("vortex.fill_null"),
        &FillNullReduceAdaptor(Constant),
    ),
    ParentRuleSet::lift_id(
        CachedId::new("vortex.filter"),
        &FilterReduceAdaptor(Constant),
    ),
    ParentRuleSet::lift_id(CachedId::new("vortex.slice"), &SliceReduceAdaptor(Constant)),
    ParentRuleSet::lift_id(CachedId::new("vortex.dict"), &TakeReduceAdaptor(Constant)),
];

static KEYED_PARENT_RULES_DENSE: ParentRuleDense<Constant> = ParentRuleDense::new();

pub(crate) static PARENT_RULES: ParentRuleSet<Constant> = ParentRuleSet::new_indexed(
    &KEYED_PARENT_RULES,
    &KEYED_PARENT_RULES_DENSE,
    &[
        ParentRuleSet::lift(&BetweenReduceAdaptor(Constant)),
        ParentRuleSet::lift(&NotReduceAdaptor(Constant)),
    ],
);

#[derive(Debug)]
struct ConstantFilterRule;

impl ArrayParentReduceRule<Constant> for ConstantFilterRule {
    type Parent = Filter;

    fn reduce_parent(
        &self,
        child: ArrayView<'_, Constant>,
        parent: ArrayView<'_, Filter>,
        _child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        Ok(Some(
            ConstantArray::new(child.scalar.clone(), parent.len()).into_array(),
        ))
    }
}
