// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_session::VortexSession;

use crate::ArrayVTable;
use crate::arrays::Filter;
use crate::arrays::Patched;
use crate::arrays::Slice;
use crate::arrays::filter::FilterReduceAdaptor;
use crate::arrays::slice::SliceReduceAdaptor;
use crate::optimizer::kernels::ArrayKernelsExt;

/// Registers the parent-reduce rules for [`Patched`] children in the session kernels.
pub(crate) fn register_parent_rules(session: &VortexSession) {
    let kernels = session.kernels();
    kernels.register_reduce_parent_rule(Filter.id(), Patched.id(), FilterReduceAdaptor(Patched));
    kernels.register_reduce_parent_rule(Slice.id(), Patched.id(), SliceReduceAdaptor(Patched));
}
