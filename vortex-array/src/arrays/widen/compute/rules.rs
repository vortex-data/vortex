// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_session::VortexSession;

use crate::arrays::Filter;
use crate::arrays::Slice;
use crate::arrays::Widen;
use crate::arrays::filter::FilterReduceAdaptor;
use crate::arrays::slice::SliceReduceAdaptor;
use crate::optimizer::kernels::ArrayKernelsExt;
use crate::scalar_fn::fns::between::Between;
use crate::scalar_fn::fns::between::BetweenReduceAdaptor;
use crate::scalar_fn::fns::cast::Cast;
use crate::scalar_fn::fns::cast::CastReduceAdaptor;
use crate::scalar_fn::fns::fill_null::FillNull;
use crate::scalar_fn::fns::fill_null::FillNullReduceAdaptor;
use crate::scalar_fn::fns::mask::Mask as MaskFn;
use crate::scalar_fn::fns::mask::MaskReduceAdaptor;

pub(crate) fn register_parent_rules(session: &VortexSession) {
    let kernels = session.kernels();
    kernels.register_reduce_parent_rule(Between, Widen, const { &BetweenReduceAdaptor(Widen) });
    kernels.register_reduce_parent_rule(Cast, Widen, const { &CastReduceAdaptor(Widen) });
    kernels.register_reduce_parent_rule(FillNull, Widen, const { &FillNullReduceAdaptor(Widen) });
    kernels.register_reduce_parent_rule(Filter, Widen, const { &FilterReduceAdaptor(Widen) });
    kernels.register_reduce_parent_rule(MaskFn, Widen, const { &MaskReduceAdaptor(Widen) });
    kernels.register_reduce_parent_rule(Slice, Widen, const { &SliceReduceAdaptor(Widen) });
}
