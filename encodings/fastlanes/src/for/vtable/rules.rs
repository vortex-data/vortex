// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::arrays::Filter;
use vortex_array::arrays::filter::FilterReduceAdaptor;
use vortex_array::arrays::slice::SliceReduceAdaptor;
use vortex_array::optimizer::rules::ArrayParentReduceRule;
use vortex_array::optimizer::rules::ParentRuleSet;
use vortex_array::scalar_fn::fns::cast::CastReduceAdaptor;
use vortex_error::VortexResult;

use crate::FoR;
use crate::FoRData;

pub(super) const PARENT_RULES: ParentRuleSet<FoR> = ParentRuleSet::new(&[
    // TODO: add BetweenReduceAdaptor(FoR)
    ParentRuleSet::lift(&FoRFilterPushDownRule),
    ParentRuleSet::lift(&FilterReduceAdaptor(FoR)),
    ParentRuleSet::lift(&SliceReduceAdaptor(FoR)),
    ParentRuleSet::lift(&CastReduceAdaptor(FoR)),
]);

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
        let new_array = unsafe {
            FoRData::new_unchecked(
                child.encoded().filter(parent.filter_mask().clone())?,
                child.reference.clone(),
            )
        };
        Ok(Some(new_array.into_array()))
    }
}
