// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::Array;
use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::FilterArray;
use crate::arrays::FilterVTable;
use crate::arrays::ListViewArray;
use crate::arrays::ListViewVTable;
use crate::arrays::SliceReduceAdaptor;
use crate::arrays::TakeReduceAdaptor;
use crate::optimizer::rules::ArrayParentReduceRule;
use crate::optimizer::rules::ParentRuleSet;
use crate::scalar_fn::fns::cast::CastReduceAdaptor;
use crate::scalar_fn::fns::mask::MaskReduceAdaptor;
use crate::vtable::ValidityHelper;

pub(crate) const PARENT_RULES: ParentRuleSet<ListViewVTable> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&ListViewFilterPushDown),
    ParentRuleSet::lift(&CastReduceAdaptor(ListViewVTable)),
    ParentRuleSet::lift(&MaskReduceAdaptor(ListViewVTable)),
    ParentRuleSet::lift(&SliceReduceAdaptor(ListViewVTable)),
    ParentRuleSet::lift(&TakeReduceAdaptor(ListViewVTable)),
]);

#[derive(Debug)]
struct ListViewFilterPushDown;

impl ArrayParentReduceRule<ListViewVTable> for ListViewFilterPushDown {
    type Parent = FilterVTable;

    fn reduce_parent(
        &self,
        array: &ListViewArray,
        parent: &FilterArray,
        _child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        // NOTE(ngates): if the filter is super selective, we maybe ought to consider masking
        //  the elements array too. We can create a new Vortex array that represents the explosion
        //  of the parent mask using the offsets/sizes arrays, and then that will be part of the
        //  filter plan.
        Ok(Some(
            unsafe {
                ListViewArray::new_unchecked(
                    array.elements().clone(),
                    array.offsets().filter(parent.filter_mask().clone())?,
                    array.sizes().filter(parent.filter_mask().clone())?,
                    array.validity().filter(parent.filter_mask())?,
                )
            }
            .into_array(),
        ))
    }
}
