// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::Array;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::FilterArray;
use vortex_array::arrays::FilterVTable;
use vortex_array::arrays::SliceReduceAdaptor;
use vortex_array::optimizer::rules::ArrayParentReduceRule;
use vortex_array::optimizer::rules::ParentRuleSet;
use vortex_error::VortexResult;

use crate::DecimalBytePartsArray;
use crate::DecimalBytePartsVTable;

pub(super) const PARENT_RULES: ParentRuleSet<DecimalBytePartsVTable> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&DecimalBytePartsFilterPushDownRule),
    ParentRuleSet::lift(&SliceReduceAdaptor(DecimalBytePartsVTable)),
]);

#[derive(Debug)]
struct DecimalBytePartsFilterPushDownRule;

impl ArrayParentReduceRule<DecimalBytePartsVTable> for DecimalBytePartsFilterPushDownRule {
    type Parent = FilterVTable;

    fn reduce_parent(
        &self,
        child: &DecimalBytePartsArray,
        parent: &FilterArray,
        _child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        // TODO(ngates): we should benchmark whether to push-down filters with "lower parts".
        //  For now, we only push down if there are no lower parts.
        if !child._lower_parts.is_empty() {
            return Ok(None);
        }

        let new_msp = child.msp.filter(parent.filter_mask().clone())?;
        let new_child =
            DecimalBytePartsArray::try_new(new_msp, *child.decimal_dtype())?.into_array();
        Ok(Some(new_child))
    }
}
