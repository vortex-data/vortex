// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::ConstantVTable;
use vortex_array::arrays::FilterArray;
use vortex_array::arrays::FilterVTable;
use vortex_array::matchers::Exact;
use vortex_array::optimizer::ArrayOptimizer;
use vortex_array::optimizer::rules::ArrayParentReduceRule;
use vortex_array::optimizer::rules::ParentRuleSet;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::DateTimePartsArray;
use crate::DateTimePartsVTable;

pub(crate) const PARENT_RULES: ParentRuleSet<DateTimePartsVTable> =
    ParentRuleSet::new(&[ParentRuleSet::lift(&DTPFilterPushDownRule)]);

/// Push the filter into the days column of a date time parts, we could extend this to other fields
/// but its less clear if that is beneficial.
#[derive(Debug)]
struct DTPFilterPushDownRule;

impl ArrayParentReduceRule<DateTimePartsVTable> for DTPFilterPushDownRule {
    type Parent = Exact<FilterVTable>;

    fn parent(&self) -> Self::Parent {
        Exact::from(&FilterVTable)
    }

    fn reduce_parent(
        &self,
        child: &DateTimePartsArray,
        parent: &FilterArray,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        if child_idx != 0 {
            return Ok(None);
        }

        if !child.seconds().is::<ConstantVTable>() || !child.subseconds().is::<ConstantVTable>() {
            return Ok(None);
        }

        DateTimePartsArray::try_new(
            child.dtype().clone(),
            FilterArray::new(child.days().clone(), parent.filter_mask().clone())
                .into_array()
                .optimize()?,
            ConstantArray::new(
                child.seconds().as_constant().vortex_expect("constant"),
                parent.filter_mask().true_count(),
            )
            .into_array(),
            ConstantArray::new(
                child.subseconds().as_constant().vortex_expect("constant"),
                parent.filter_mask().true_count(),
            )
            .into_array(),
        )
        .map(|x| Some(x.into_array()))
    }
}
