// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::Array;
use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::FilterArray;
use crate::arrays::FilterVTable;
use crate::matchers::Exact;
use crate::optimizer::rules::ArrayParentReduceRule;
use crate::optimizer::rules::ParentRuleSet;

pub(super) const PARENT_RULES: ParentRuleSet<FilterVTable> =
    ParentRuleSet::new(&[ParentRuleSet::lift(&FilterFilterRule)]);

/// A simple redecution rule that simplifies a [`FilterArray`] whose child is also a
/// [`FilterArray`].
#[derive(Debug)]
struct FilterFilterRule;

impl ArrayParentReduceRule<FilterVTable> for FilterFilterRule {
    type Parent = Exact<FilterVTable>;

    fn parent(&self) -> Self::Parent {
        Exact::new()
    }

    fn reduce_parent(
        &self,
        child: &FilterArray,
        parent: &FilterArray,
        _child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        let combined_mask = child.mask.intersect_by_rank(&parent.mask);
        let new_array = child.child.filter(combined_mask)?;

        Ok(Some(new_array.into_array()))
    }
}
