// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_session::VortexSession;

use crate::ArrayVTable;
use crate::arrays::Dict;
use crate::arrays::Filter;
use crate::arrays::Slice;
use crate::arrays::Variant;
use crate::arrays::dict::TakeReduceAdaptor;
use crate::arrays::filter::FilterReduceAdaptor;
use crate::arrays::slice::SliceReduceAdaptor;
use crate::optimizer::kernels::ArrayKernelsExt;

/// Registers the parent-reduce rules for [`Variant`] children in the session kernels.
pub(crate) fn register_parent_rules(session: &VortexSession) {
    let kernels = session.kernels();
    kernels.register_reduce_parent_rule(Slice.id(), Variant, SliceReduceAdaptor(Variant));
    kernels.register_reduce_parent_rule(Filter.id(), Variant, FilterReduceAdaptor(Variant));
    kernels.register_reduce_parent_rule(Dict.id(), Variant, TakeReduceAdaptor(Variant));
}
