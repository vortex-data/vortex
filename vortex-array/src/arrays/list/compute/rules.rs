// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_session::VortexSession;

use crate::arrays::List;
use crate::arrays::Slice;
use crate::arrays::slice::SliceReduceAdaptor;
use crate::optimizer::kernels::ArrayKernelsExt;
use crate::scalar_fn::fns::cast::Cast;
use crate::scalar_fn::fns::cast::CastReduceAdaptor;
use crate::scalar_fn::fns::mask::Mask as MaskFn;
use crate::scalar_fn::fns::mask::MaskReduceAdaptor;

pub(crate) fn register_parent_rules(session: &VortexSession) {
    let kernels = session.kernels();
    kernels.register_reduce_parent_rule(Cast, List, const { &CastReduceAdaptor(List) });
    kernels.register_reduce_parent_rule(MaskFn, List, const { &MaskReduceAdaptor(List) });
    kernels.register_reduce_parent_rule(Slice, List, const { &SliceReduceAdaptor(List) });
}
