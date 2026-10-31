// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_session::VortexSession;

use crate::ArrayVTable;
use crate::arrays::Dict;
use crate::arrays::Slice;
use crate::arrays::Union;
use crate::arrays::dict::TakeReduceAdaptor;
use crate::arrays::slice::SliceReduceAdaptor;
use crate::optimizer::kernels::ArrayKernelsExt;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::fns::mask::Mask as MaskFn;
use crate::scalar_fn::fns::mask::MaskReduceAdaptor;

/// Registers the parent-reduce rules for [`Union`] children in the session kernels.
pub(crate) fn register_parent_rules(session: &VortexSession) {
    let kernels = session.kernels();
    kernels.register_reduce_parent_rule(MaskFn.id(), Union, MaskReduceAdaptor(Union));
    kernels.register_reduce_parent_rule(Slice.id(), Union, SliceReduceAdaptor(Union));
    kernels.register_reduce_parent_rule(Dict.id(), Union, TakeReduceAdaptor(Union));
}
