// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_session::registry::CachedId;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Filter;
use crate::arrays::ListView;
use crate::arrays::ListViewArray;
use crate::arrays::dict::TakeReduceAdaptor;
use crate::arrays::listview::ListViewArrayExt;
use crate::arrays::slice::SliceReduceAdaptor;
use crate::optimizer::rules::ArrayParentReduceRule;
use crate::optimizer::rules::ParentRuleDense;
use crate::optimizer::rules::ParentRuleEntry;
use crate::optimizer::rules::ParentRuleSet;
use crate::scalar_fn::fns::cast::CastReduceAdaptor;
use crate::scalar_fn::fns::mask::MaskReduceAdaptor;

static KEYED_PARENT_RULES: [ParentRuleEntry<ListView>; 5] = [
    ParentRuleSet::lift_id(CachedId::new("vortex.filter"), &ListViewFilterPushDown),
    ParentRuleSet::lift_id(CachedId::new("vortex.cast"), &CastReduceAdaptor(ListView)),
    ParentRuleSet::lift_id(CachedId::new("vortex.mask"), &MaskReduceAdaptor(ListView)),
    ParentRuleSet::lift_id(CachedId::new("vortex.slice"), &SliceReduceAdaptor(ListView)),
    ParentRuleSet::lift_id(CachedId::new("vortex.dict"), &TakeReduceAdaptor(ListView)),
];

static KEYED_PARENT_RULES_DENSE: ParentRuleDense<ListView> = ParentRuleDense::new();

pub(crate) static PARENT_RULES: ParentRuleSet<ListView> =
    ParentRuleSet::new_indexed(&KEYED_PARENT_RULES, &KEYED_PARENT_RULES_DENSE, &[]);

#[derive(Debug)]
struct ListViewFilterPushDown;

impl ArrayParentReduceRule<ListView> for ListViewFilterPushDown {
    type Parent = Filter;

    fn reduce_parent(
        &self,
        array: ArrayView<'_, ListView>,
        parent: ArrayView<'_, Filter>,
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
                    array.validity()?.filter(parent.filter_mask())?,
                )
            }
            .into_array(),
        ))
    }
}
