// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayVTable;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::arrays::Filter;
use vortex_array::arrays::Slice;
use vortex_array::arrays::filter::FilterReduceAdaptor;
use vortex_array::arrays::slice::SliceReduceAdaptor;
use vortex_array::optimizer::kernels::ArrayKernelsExt;
use vortex_array::optimizer::rules::ArrayParentReduceRule;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::scalar_fn::fns::cast::Cast;
use vortex_array::scalar_fn::fns::cast::CastReduceAdaptor;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::FoR;
use crate::FoRReferences;
use crate::for_::array::FoRArrayExt;
use crate::for_::array::FoRArraySlotsExt;

/// Registers the parent-reduce rules for [`FoR`] children in the session kernels.
pub(crate) fn register_parent_rules(session: &VortexSession) {
    let kernels = session.kernels();
    // TODO: add BetweenReduceAdaptor(FoR)
    kernels.register_reduce_parent_rule(Filter.id(), FoR, FoRFilterPushDownRule);
    kernels.register_reduce_parent_rule(Filter.id(), FoR, FilterReduceAdaptor(FoR));
    kernels.register_reduce_parent_rule(Slice.id(), FoR, SliceReduceAdaptor(FoR));
    kernels.register_reduce_parent_rule(Cast.id(), FoR, CastReduceAdaptor(FoR));
}

#[derive(Debug)]
struct FoRFilterPushDownRule;

impl ArrayParentReduceRule<FoR> for FoRFilterPushDownRule {
    type Parent = Filter;

    fn reduce_parent(
        &self,
        child: ArrayView<'_, FoR>,
        parent: ArrayView<'_, Filter>,
        _child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        // TODO(mk): support many references.
        let FoRReferences::Global(reference) = child.references() else {
            return Ok(None);
        };
        Ok(Some(
            FoR::try_new(
                child.encoded().filter(parent.filter_mask().clone())?,
                reference.clone(),
            )?
            .into_array(),
        ))
    }
}
