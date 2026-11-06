// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::Filter;
use vortex_array::arrays::Slice;
use vortex_array::arrays::filter::FilterReduceAdaptor;
use vortex_array::arrays::slice::SliceReduceAdaptor;
use vortex_array::optimizer::kernels::ArrayKernelsExt;
use vortex_array::scalar_fn::fns::cast::Cast;
use vortex_array::scalar_fn::fns::cast::CastReduceAdaptor;
use vortex_array::scalar_fn::fns::mask::Mask as MaskFn;
use vortex_array::scalar_fn::fns::mask::MaskReduceAdaptor;
use vortex_session::VortexSession;

use crate::ZigZag;

pub(crate) fn register_parent_rules(session: &VortexSession) {
    let kernels = session.kernels();
    kernels.register_reduce_parent_rule(Cast, ZigZag, const { &CastReduceAdaptor(ZigZag) });
    kernels.register_reduce_parent_rule(Filter, ZigZag, const { &FilterReduceAdaptor(ZigZag) });
    kernels.register_reduce_parent_rule(MaskFn, ZigZag, const { &MaskReduceAdaptor(ZigZag) });
    kernels.register_reduce_parent_rule(Slice, ZigZag, const { &SliceReduceAdaptor(ZigZag) });
}
