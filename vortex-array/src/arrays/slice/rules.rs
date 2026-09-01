// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::array::ArrayView;
use crate::arrays::Slice;
use crate::arrays::scalar_fn::ExactScalarFn;
use crate::arrays::scalar_fn::ScalarFnArrayView;
use crate::arrays::slice::SliceArraySlotsExt;
use crate::arrays::slice::SliceReduceAdaptor;
use crate::builtins::ArrayBuiltins;
use crate::optimizer::rules::ArrayParentReduceRule;
use crate::optimizer::rules::ParentRuleSet;
use crate::scalar_fn::fns::get_item::GetItem;

pub(super) const PARENT_RULES: ParentRuleSet<Slice> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&SliceReduceAdaptor(Slice)),
    ParentRuleSet::lift(&SliceGetItemRule),
]);

#[derive(Debug)]
struct SliceGetItemRule;

impl ArrayParentReduceRule<Slice> for SliceGetItemRule {
    type Parent = ExactScalarFn<GetItem>;

    fn reduce_parent(
        &self,
        array: ArrayView<'_, Slice>,
        parent: ScalarFnArrayView<'_, GetItem>,
        _child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        let field = array.child().get_item(parent.options.clone())?;
        Ok(Some(field.slice(array.slice_range().clone())?))
    }
}
