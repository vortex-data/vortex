// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::DynArray;
use vortex_array::IntoArray;
use vortex_array::arrays::FilterArray;
use vortex_array::arrays::FilterReduceAdaptor;
use vortex_array::arrays::FilterVTable;
use vortex_array::arrays::SliceReduceAdaptor;
use vortex_array::optimizer::rules::ArrayParentReduceRule;
use vortex_array::optimizer::rules::ParentRuleSet;
use vortex_array::scalar_fn::fns::cast::CastReduceAdaptor;
use vortex_error::VortexResult;

use crate::FoRArray;
use crate::FoRVTable;

pub(super) const PARENT_RULES: ParentRuleSet<FoRVTable> = ParentRuleSet::new(&[
    // TODO: add BetweenReduceAdaptor(FoRVTable)
    ParentRuleSet::lift(&FoRFilterPushDownRule),
    ParentRuleSet::lift(&FilterReduceAdaptor(FoRVTable)),
    ParentRuleSet::lift(&SliceReduceAdaptor(FoRVTable)),
    ParentRuleSet::lift(&CastReduceAdaptor(FoRVTable)),
]);

#[derive(Debug)]
struct FoRFilterPushDownRule;

impl ArrayParentReduceRule<FoRVTable> for FoRFilterPushDownRule {
    type Parent = FilterVTable;

    fn reduce_parent(
        &self,
        child: &FoRArray,
        parent: &FilterArray,
        _child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        let new_array = unsafe {
            FoRArray::new_unchecked(
                child.encoded.filter(parent.filter_mask().clone())?,
                child.reference.clone(),
            )
        };
        Ok(Some(new_array.into_array()))
    }
}
