// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Constant;
use crate::arrays::ConstantArray;
use crate::arrays::Filter;
use crate::arrays::dict::TakeReduceAdaptor;
use crate::arrays::filter::FilterReduceAdaptor;
use crate::arrays::slice::SliceReduceAdaptor;
use crate::optimizer::rules::ArrayParentReduceRule;
use crate::optimizer::rules::ParentRuleSet;
use crate::scalar_fn::fns::between::BetweenReduceAdaptor;
use crate::scalar_fn::fns::cast::CastReduceAdaptor;
use crate::scalar_fn::fns::fill_null::FillNullReduceAdaptor;
use crate::scalar_fn::fns::not::NotReduceAdaptor;

pub(crate) const PARENT_RULES: ParentRuleSet<Constant> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&BetweenReduceAdaptor(Constant)),
    ParentRuleSet::lift(&CastReduceAdaptor(Constant)),
    ParentRuleSet::lift(&ConstantFilterRule),
    ParentRuleSet::lift(&FillNullReduceAdaptor(Constant)),
    ParentRuleSet::lift(&FilterReduceAdaptor(Constant)),
    ParentRuleSet::lift(&NotReduceAdaptor(Constant)),
    ParentRuleSet::lift(&SliceReduceAdaptor(Constant)),
    ParentRuleSet::lift(&TakeReduceAdaptor(Constant)),
]);

#[derive(Debug)]
struct ConstantFilterRule;

impl ArrayParentReduceRule<Constant> for ConstantFilterRule {
    type Parent = Filter;

    fn reduce_parent(
        &self,
        child: ArrayView<'_, Constant>,
        parent: ArrayView<'_, Filter>,
        _child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        Ok(Some(
            ConstantArray::new(child.scalar.clone(), parent.len()).into_array(),
        ))
    }
}
