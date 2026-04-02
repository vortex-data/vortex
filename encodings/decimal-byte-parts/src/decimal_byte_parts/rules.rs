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
use vortex_array::scalar_fn::fns::mask::MaskReduceAdaptor;
use vortex_error::VortexResult;

use super::DecimalBytePartsData;
use crate::DecimalByteParts;

pub(super) const PARENT_RULES: ParentRuleSet<DecimalByteParts> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&DecimalBytePartsFilterPushDownRule),
    ParentRuleSet::lift(&CastReduceAdaptor(DecimalByteParts)),
    ParentRuleSet::lift(&FilterReduceAdaptor(DecimalByteParts)),
    ParentRuleSet::lift(&MaskReduceAdaptor(DecimalByteParts)),
    ParentRuleSet::lift(&SliceReduceAdaptor(DecimalByteParts)),
]);

#[derive(Debug)]
struct DecimalBytePartsFilterPushDownRule;

impl ArrayParentReduceRule<DecimalByteParts> for DecimalBytePartsFilterPushDownRule {
    type Parent = Filter;

    fn reduce_parent(
        &self,
        child: ArrayView<'_, DecimalByteParts>,
        parent: ArrayView<'_, Filter>,
        _child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        // TODO(ngates): we should benchmark whether to push-down filters with "lower parts".
        //  For now, we only push down if there are no lower parts.
        if !child._lower_parts.is_empty() {
            return Ok(None);
        }

        let new_msp = child.msp().filter(parent.filter_mask().clone())?;
        let new_child =
            DecimalBytePartsData::try_new(new_msp, *child.decimal_dtype())?.into_array();
        Ok(Some(new_child))
    }
}
