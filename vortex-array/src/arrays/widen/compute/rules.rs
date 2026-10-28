// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_session::VortexSession;

use crate::ArrayVTable;
use crate::arrays::Filter;
use crate::arrays::Slice;
use crate::arrays::Widen;
use crate::arrays::filter::FilterReduceAdaptor;
use crate::arrays::slice::SliceReduceAdaptor;
use crate::optimizer::kernels::ArrayKernelsExt;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::fns::between::Between;
use crate::scalar_fn::fns::between::BetweenReduceAdaptor;
use crate::scalar_fn::fns::cast::Cast;
use crate::scalar_fn::fns::cast::CastReduceAdaptor;
use crate::scalar_fn::fns::fill_null::FillNull;
use crate::scalar_fn::fns::fill_null::FillNullReduceAdaptor;
use crate::scalar_fn::fns::mask::Mask as MaskFn;
use crate::scalar_fn::fns::mask::MaskReduceAdaptor;

/// Registers the parent-reduce rules for [`Widen`] children in the session kernels.
pub(crate) fn register_parent_rules(session: &VortexSession) {
    let kernels = session.kernels();
    kernels.register_reduce_parent_rule(Between.id(), Widen.id(), BetweenReduceAdaptor(Widen));
    kernels.register_reduce_parent_rule(Cast.id(), Widen.id(), CastReduceAdaptor(Widen));
    kernels.register_reduce_parent_rule(FillNull.id(), Widen.id(), FillNullReduceAdaptor(Widen));
    kernels.register_reduce_parent_rule(Filter.id(), Widen.id(), FilterReduceAdaptor(Widen));
    kernels.register_reduce_parent_rule(MaskFn.id(), Widen.id(), MaskReduceAdaptor(Widen));
    kernels.register_reduce_parent_rule(Slice.id(), Widen.id(), SliceReduceAdaptor(Widen));
}
