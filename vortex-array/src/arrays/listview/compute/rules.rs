// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::ArrayVTable;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Dict;
use crate::arrays::Filter;
use crate::arrays::ListView;
use crate::arrays::ListViewArray;
use crate::arrays::Slice;
use crate::arrays::dict::TakeReduceAdaptor;
use crate::arrays::listview::ListViewArraySlotsExt;
use crate::arrays::slice::SliceReduceAdaptor;
use crate::optimizer::kernels::ArrayKernelsExt;
use crate::optimizer::rules::ArrayParentReduceRule;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::fns::cast::Cast;
use crate::scalar_fn::fns::cast::CastReduceAdaptor;
use crate::scalar_fn::fns::mask::Mask as MaskFn;
use crate::scalar_fn::fns::mask::MaskReduceAdaptor;

/// Registers the parent-reduce rules for [`ListView`] children in the session kernels.
pub(crate) fn register_parent_rules(session: &VortexSession) {
    let kernels = session.kernels();
    kernels.register_reduce_parent_rule(Filter.id(), ListView, ListViewFilterPushDown);
    kernels.register_reduce_parent_rule(Cast.id(), ListView, CastReduceAdaptor(ListView));
    kernels.register_reduce_parent_rule(MaskFn.id(), ListView, MaskReduceAdaptor(ListView));
    kernels.register_reduce_parent_rule(Slice.id(), ListView, SliceReduceAdaptor(ListView));
    kernels.register_reduce_parent_rule(Dict.id(), ListView, TakeReduceAdaptor(ListView));
}

#[derive(Debug)]
struct ListViewFilterPushDown;

impl ArrayParentReduceRule<ListView> for ListViewFilterPushDown {
    type Parent = Filter;

    fn reduce_parent(
        &self,
        array: ArrayView<'_, ListView>,
        parent: ArrayView<'_, Filter>,
        _child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        // NOTE(ngates): if the filter is super selective, we maybe ought to consider masking
        //  the elements array too. We can create a new Vortex array that represents the explosion
        //  of the parent mask using the offsets/sizes arrays, and then that will be part of the
        //  filter plan.
        Ok(Some(
            unsafe {
                ListViewArray::new_unchecked(
                    array.elements().clone(),
                    array.offsets().filter(parent.filter_mask().clone())?,
                    array.sizes().filter(parent.filter_mask().clone())?,
                    array.validity()?.filter(parent.filter_mask())?,
                )
            }
            .into_array(),
        ))
    }
}
