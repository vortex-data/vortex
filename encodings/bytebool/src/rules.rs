// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayVTable;
use vortex_array::arrays::Slice;
use vortex_array::arrays::slice::SliceReduceAdaptor;
use vortex_array::optimizer::kernels::ArrayKernelsExt;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::scalar_fn::fns::cast::Cast;
use vortex_array::scalar_fn::fns::cast::CastReduceAdaptor;
use vortex_array::scalar_fn::fns::mask::Mask as MaskFn;
use vortex_array::scalar_fn::fns::mask::MaskReduceAdaptor;
use vortex_session::VortexSession;

use crate::ByteBool;

/// Registers the parent-reduce rules for [`ByteBool`] children in the session kernels.
pub(crate) fn register_parent_rules(session: &VortexSession) {
    let kernels = session.kernels();
    kernels.register_reduce_parent_rule(Cast.id(), ByteBool, CastReduceAdaptor(ByteBool));
    kernels.register_reduce_parent_rule(MaskFn.id(), ByteBool, MaskReduceAdaptor(ByteBool));
    kernels.register_reduce_parent_rule(Slice.id(), ByteBool, SliceReduceAdaptor(ByteBool));
}
