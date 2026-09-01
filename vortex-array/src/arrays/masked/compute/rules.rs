// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::array::ArrayView;
use crate::arrays::Masked;
use crate::arrays::dict::TakeReduceAdaptor;
use crate::arrays::filter::FilterReduceAdaptor;
use crate::arrays::masked::MaskedArrayExt;
use crate::arrays::masked::MaskedArraySlotsExt;
use crate::arrays::scalar_fn::ExactScalarFn;
use crate::arrays::scalar_fn::ScalarFnArrayView;
use crate::arrays::slice::SliceReduceAdaptor;
use crate::builtins::ArrayBuiltins;
use crate::optimizer::rules::ArrayParentReduceRule;
use crate::optimizer::rules::ParentRuleSet;
use crate::scalar_fn::fns::get_item::GetItem;
use crate::scalar_fn::fns::mask::MaskReduceAdaptor;

pub(crate) const PARENT_RULES: ParentRuleSet<Masked> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&FilterReduceAdaptor(Masked)),
    ParentRuleSet::lift(&MaskedGetItemRule),
    ParentRuleSet::lift(&MaskReduceAdaptor(Masked)),
    ParentRuleSet::lift(&SliceReduceAdaptor(Masked)),
    ParentRuleSet::lift(&TakeReduceAdaptor(Masked)),
]);

#[derive(Debug)]
struct MaskedGetItemRule;

impl ArrayParentReduceRule<Masked> for MaskedGetItemRule {
    type Parent = ExactScalarFn<GetItem>;

    fn reduce_parent(
        &self,
        array: ArrayView<'_, Masked>,
        parent: ScalarFnArrayView<'_, GetItem>,
        _child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        let field = array.child().get_item(parent.options.clone())?;
        Ok(Some(
            field.mask(array.masked_validity().to_array(array.len()))?,
        ))
    }
}
