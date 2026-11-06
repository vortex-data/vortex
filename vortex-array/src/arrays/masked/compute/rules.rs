// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::array::ArrayView;
use crate::arrays::Dict;
use crate::arrays::Filter;
use crate::arrays::Masked;
use crate::arrays::Slice;
use crate::arrays::dict::TakeReduceAdaptor;
use crate::arrays::filter::FilterReduceAdaptor;
use crate::arrays::masked::MaskedArrayExt;
use crate::arrays::masked::MaskedArraySlotsExt;
use crate::arrays::scalar_fn::ExactScalarFn;
use crate::arrays::scalar_fn::ScalarFnArrayView;
use crate::arrays::slice::SliceReduceAdaptor;
use crate::builtins::ArrayBuiltins;
use crate::optimizer::kernels::ArrayKernelsExt;
use crate::optimizer::rules::ArrayParentReduceRule;
use crate::scalar_fn::fns::get_item::GetItem;
use crate::scalar_fn::fns::mask::Mask as MaskFn;
use crate::scalar_fn::fns::mask::MaskReduceAdaptor;

pub(crate) fn register_parent_rules(session: &VortexSession) {
    let kernels = session.kernels();
    kernels.register_reduce_parent_rule(Filter, Masked, const { &FilterReduceAdaptor(Masked) });
    kernels.register_reduce_parent_rule(GetItem, Masked, const { &MaskedGetItemRule });
    kernels.register_reduce_parent_rule(MaskFn, Masked, const { &MaskReduceAdaptor(Masked) });
    kernels.register_reduce_parent_rule(Slice, Masked, const { &SliceReduceAdaptor(Masked) });
    kernels.register_reduce_parent_rule(Dict, Masked, const { &TakeReduceAdaptor(Masked) });
}

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
