// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::ConstantVTable;
use crate::arrays::FilterArray;
use crate::arrays::FilterVTable;
use crate::optimizer::rules::ArrayParentReduceRule;
use crate::optimizer::rules::ParentRuleSet;

pub(super) const PARENT_RULES: ParentRuleSet<ConstantVTable> =
    ParentRuleSet::new(&[ParentRuleSet::lift(&ConstantFilterRule)]);

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
