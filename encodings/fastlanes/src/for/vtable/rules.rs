// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::arrays::Filter;
use vortex_array::arrays::filter::FilterReduceAdaptor;
use vortex_array::arrays::slice::SliceReduceAdaptor;
use vortex_array::optimizer::rules::ArrayParentReduceRule;
use vortex_array::optimizer::rules::ParentRuleDense;
use vortex_array::optimizer::rules::ParentRuleEntry;
use vortex_array::optimizer::rules::ParentRuleSet;
use vortex_array::scalar_fn::fns::cast::CastReduceAdaptor;
use vortex_error::VortexResult;
use vortex_session::registry::CachedId;

use crate::FoR;
use crate::r#for::array::FoRArrayExt;

static KEYED_PARENT_RULES: [ParentRuleEntry<FoR>; 3] = [
    ParentRuleSet::lift_id(CachedId::new("vortex.filter"), &FilterReduceAdaptor(FoR)),
    ParentRuleSet::lift_id(CachedId::new("vortex.slice"), &SliceReduceAdaptor(FoR)),
    ParentRuleSet::lift_id(CachedId::new("vortex.cast"), &CastReduceAdaptor(FoR)),
];

static KEYED_PARENT_RULES_DENSE: ParentRuleDense<FoR> = ParentRuleDense::new();

pub(super) static PARENT_RULES: ParentRuleSet<FoR> = ParentRuleSet::new_indexed(
    &KEYED_PARENT_RULES,
    &KEYED_PARENT_RULES_DENSE,
    &[
        // TODO: add BetweenReduceAdaptor(FoR)
        ParentRuleSet::lift(&FoRFilterPushDownRule),
    ],
);

#[derive(Debug)]
struct FoRFilterPushDownRule;

impl ArrayParentReduceRule<FoR> for FoRFilterPushDownRule {
    type Parent = Filter;

    fn reduce_parent(
        &self,
        child: ArrayView<'_, FoR>,
        parent: ArrayView<'_, Filter>,
        _child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        Ok(Some(
            FoR::try_new(
                child.encoded().filter(parent.filter_mask().clone())?,
                child.reference_scalar().clone(),
            )?
            .into_array(),
        ))
    }
}
