// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
use vortex_session::VortexSession;

use crate::ArrayVTable;
use crate::arrays::Slice;
use crate::arrays::VarBinView;
use crate::arrays::slice::SliceReduceAdaptor;
use crate::optimizer::kernels::ArrayKernelsExt;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::fns::cast::Cast;
use crate::scalar_fn::fns::cast::CastReduceAdaptor;
use crate::scalar_fn::fns::mask::Mask as MaskFn;
use crate::scalar_fn::fns::mask::MaskReduceAdaptor;

/// Registers the parent-reduce rules for [`VarBinView`] children in the session kernels.
pub(crate) fn register_parent_rules(session: &VortexSession) {
    let kernels = session.kernels();
    kernels.register_reduce_parent_rule(Cast.id(), VarBinView, CastReduceAdaptor(VarBinView));
    kernels.register_reduce_parent_rule(MaskFn.id(), VarBinView, MaskReduceAdaptor(VarBinView));
    kernels.register_reduce_parent_rule(Slice.id(), VarBinView, SliceReduceAdaptor(VarBinView));
}
