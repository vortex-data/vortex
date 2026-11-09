// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_session::VortexSession;

use crate::arrays::Filter;
use crate::arrays::Patched;
use crate::arrays::Slice;
use crate::arrays::filter::FilterReduceAdaptor;
use crate::arrays::slice::SliceReduceAdaptor;
use crate::optimizer::kernels::ArrayKernelsExt;

pub(crate) fn register_parent_rules(session: &VortexSession) {
    let kernels = session.kernels();
    kernels.register_reduce_parent_rule(Filter, Patched, const { &FilterReduceAdaptor(Patched) });
    kernels.register_reduce_parent_rule(Slice, Patched, const { &SliceReduceAdaptor(Patched) });
}
