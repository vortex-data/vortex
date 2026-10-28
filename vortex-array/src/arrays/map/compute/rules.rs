// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_session::VortexSession;

use crate::ArrayVTable;
use crate::arrays::Dict;
use crate::arrays::Filter;
use crate::arrays::Map;
use crate::arrays::Slice;
use crate::arrays::dict::TakeReduceAdaptor;
use crate::arrays::filter::FilterReduceAdaptor;
use crate::arrays::slice::SliceReduceAdaptor;
use crate::optimizer::kernels::ArrayKernelsExt;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::fns::cast::Cast;
use crate::scalar_fn::fns::cast::CastReduceAdaptor;
use crate::scalar_fn::fns::mask::Mask as MaskFn;
use crate::scalar_fn::fns::mask::MaskReduceAdaptor;

/// Registers the parent-reduce rules for [`Map`] children in the session kernels.
pub(crate) fn register_parent_rules(session: &VortexSession) {
    let kernels = session.kernels();
    kernels.register_reduce_parent_rule(Filter.id(), Map.id(), FilterReduceAdaptor(Map));
    kernels.register_reduce_parent_rule(Cast.id(), Map.id(), CastReduceAdaptor(Map));
    kernels.register_reduce_parent_rule(MaskFn.id(), Map.id(), MaskReduceAdaptor(Map));
    kernels.register_reduce_parent_rule(Slice.id(), Map.id(), SliceReduceAdaptor(Map));
    kernels.register_reduce_parent_rule(Dict.id(), Map.id(), TakeReduceAdaptor(Map));
}
