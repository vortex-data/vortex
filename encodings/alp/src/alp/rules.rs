// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayVTable;
use vortex_array::arrays::Dict;
use vortex_array::arrays::Filter;
use vortex_array::arrays::Slice;
use vortex_array::arrays::dict::TakeExecuteAdaptor;
use vortex_array::arrays::filter::FilterExecuteAdaptor;
use vortex_array::arrays::slice::SliceExecuteAdaptor;
use vortex_array::optimizer::kernels::ArrayKernelsExt;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::scalar_fn::fns::between::Between;
use vortex_array::scalar_fn::fns::between::BetweenReduceAdaptor;
use vortex_array::scalar_fn::fns::binary::Binary;
use vortex_array::scalar_fn::fns::binary::CompareExecuteAdaptor;
use vortex_array::scalar_fn::fns::cast::Cast;
use vortex_array::scalar_fn::fns::cast::CastReduceAdaptor;
use vortex_array::scalar_fn::fns::mask::Mask as MaskFn;
use vortex_array::scalar_fn::fns::mask::Mask;
use vortex_array::scalar_fn::fns::mask::MaskExecuteAdaptor;
use vortex_array::scalar_fn::fns::mask::MaskReduceAdaptor;
use vortex_session::VortexSession;

use crate::ALP;

pub(super) fn initialize(session: &VortexSession) {
    let kernels = session.kernels();
    kernels.register_execute_parent_kernel(Binary.id(), ALP, CompareExecuteAdaptor(ALP));
    kernels.register_execute_parent_kernel(Filter.id(), ALP, FilterExecuteAdaptor(ALP));
    kernels.register_execute_parent_kernel(Mask.id(), ALP, MaskExecuteAdaptor(ALP));
    kernels.register_execute_parent_kernel(Slice.id(), ALP, SliceExecuteAdaptor(ALP));
    kernels.register_execute_parent_kernel(Dict.id(), ALP, TakeExecuteAdaptor(ALP));
}

/// Registers the parent-reduce rules for [`ALP`] children in the session kernels.
pub(crate) fn register_parent_rules(session: &VortexSession) {
    let kernels = session.kernels();
    kernels.register_reduce_parent_rule(Between.id(), ALP, BetweenReduceAdaptor(ALP));
    kernels.register_reduce_parent_rule(Cast.id(), ALP, CastReduceAdaptor(ALP));
    kernels.register_reduce_parent_rule(MaskFn.id(), ALP, MaskReduceAdaptor(ALP));
}
