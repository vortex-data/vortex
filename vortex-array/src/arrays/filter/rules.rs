// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::FilterArray;
use crate::arrays::FilterVTable;
use crate::optimizer::rules::ArrayParentReduceRule;
use crate::optimizer::rules::Exact;
use crate::optimizer::rules::ParentRuleSet;

pub(super) const PARENT_RULES: ParentRuleSet<FilterVTable> =
    ParentRuleSet::new(&[ParentRuleSet::lift(&FilterFilterRule)]);

/// Reduce rule that simplifies a Filter array whose child is also a Filter array
#[derive(Debug)]
struct FilterFilterRule;

impl ArrayParentReduceRule<FilterVTable> for FilterFilterRule {
    type Parent = Exact<FilterVTable>;

    fn parent(&self) -> Self::Parent {
        Exact::from(&FilterVTable)
    }

    fn reduce_parent(
        &self,
        child: &FilterArray,
        parent: &FilterArray,
        _child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        let combined_mask = child.mask.intersect_by_rank(&parent.mask);
        let new_array = FilterArray::new(child.child.clone(), combined_mask);
        Ok(Some(new_array.into_array()))
    }
}
