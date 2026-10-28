// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_session::VortexSession;

use crate::ArrayVTable;
use crate::arrays::List;
use crate::arrays::Slice;
use crate::arrays::slice::SliceReduceAdaptor;
use crate::optimizer::kernels::ArrayKernelsExt;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::fns::cast::Cast;
use crate::scalar_fn::fns::cast::CastReduceAdaptor;
use crate::scalar_fn::fns::mask::Mask as MaskFn;
use crate::scalar_fn::fns::mask::MaskReduceAdaptor;

/// Registers the parent-reduce rules for [`List`] children in the session kernels.
pub(crate) fn register_parent_rules(session: &VortexSession) {
    let kernels = session.kernels();
    kernels.register_reduce_parent_rule(Cast.id(), List.id(), CastReduceAdaptor(List));
    kernels.register_reduce_parent_rule(MaskFn.id(), List.id(), MaskReduceAdaptor(List));
    kernels.register_reduce_parent_rule(Slice.id(), List.id(), SliceReduceAdaptor(List));
}
