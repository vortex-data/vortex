// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::constant::ConstantArray;
use crate::arrays::constant::ConstantVTable;
use crate::arrays::dict::TakeReduceAdaptor;
use crate::arrays::filter::FilterArray;
use crate::arrays::filter::FilterReduceAdaptor;
use crate::arrays::filter::FilterVTable;
use crate::arrays::slice::SliceReduceAdaptor;
use crate::optimizer::rules::ArrayParentReduceRule;
use crate::optimizer::rules::ParentRuleSet;
use crate::scalar_fn::fns::between::BetweenReduceAdaptor;
use crate::scalar_fn::fns::cast::CastReduceAdaptor;
use crate::scalar_fn::fns::fill_null::FillNullReduceAdaptor;
use crate::scalar_fn::fns::not::NotReduceAdaptor;

pub(crate) const PARENT_RULES: ParentRuleSet<ConstantVTable> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&BetweenReduceAdaptor(ConstantVTable)),
    ParentRuleSet::lift(&CastReduceAdaptor(ConstantVTable)),
    ParentRuleSet::lift(&ConstantFilterRule),
    ParentRuleSet::lift(&FillNullReduceAdaptor(ConstantVTable)),
    ParentRuleSet::lift(&FilterReduceAdaptor(ConstantVTable)),
    ParentRuleSet::lift(&NotReduceAdaptor(ConstantVTable)),
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
