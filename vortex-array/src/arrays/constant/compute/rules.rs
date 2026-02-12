// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::ConstantVTable;
use crate::arrays::FilterArray;
use crate::arrays::FilterReduceAdaptor;
use crate::arrays::FilterVTable;
use crate::arrays::SliceReduceAdaptor;
use crate::arrays::TakeReduceAdaptor;
use crate::compute::FillNullReduceAdaptor;
use crate::compute::NotReduceAdaptor;
use crate::optimizer::rules::ArrayParentReduceRule;
use crate::optimizer::rules::ParentRuleSet;

pub(crate) const PARENT_RULES: ParentRuleSet<ConstantVTable> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&ConstantFilterRule),
    ParentRuleSet::lift(&NotReduceAdaptor(ConstantVTable)),
    ParentRuleSet::lift(&FillNullReduceAdaptor(ConstantVTable)),
    ParentRuleSet::lift(&FilterReduceAdaptor(ConstantVTable)),
    ParentRuleSet::lift(&SliceReduceAdaptor(ConstantVTable)),
    ParentRuleSet::lift(&TakeReduceAdaptor(ConstantVTable)),
]);

#[derive(Debug)]
struct ConstantFilterRule;

impl ArrayParentReduceRule<ConstantVTable> for ConstantFilterRule {
    type Parent = FilterVTable;

    fn reduce_parent(
        &self,
        child: &ConstantArray,
        parent: &FilterArray,
        _child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        Ok(Some(
            ConstantArray::new(child.scalar.clone(), parent.len()).into_array(),
        ))
    }
}
