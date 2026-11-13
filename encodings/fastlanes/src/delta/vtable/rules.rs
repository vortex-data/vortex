// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayVTable;
use vortex_array::arrays::Dict;
use vortex_array::arrays::Filter;
use vortex_array::arrays::Slice;
use vortex_array::arrays::dict::TakeExecuteAdaptor;
use vortex_array::arrays::filter::FilterExecuteAdaptor;
use vortex_array::arrays::slice::SliceReduceAdaptor;
use vortex_array::optimizer::kernels::ArrayKernelsExt;
use vortex_array::scalar_fn::builtin::cast::Cast;
use vortex_array::scalar_fn::builtin::cast::CastReduceAdaptor;
use vortex_session::VortexSession;

use crate::delta::vtable::Delta;

pub(crate) fn register_parent_rules(session: &VortexSession) {
    let kernels = session.kernels();
    kernels.register_reduce_parent_rule(Slice, Delta, const { &SliceReduceAdaptor(Delta) });
    kernels.register_reduce_parent_rule(Cast, Delta, const { &CastReduceAdaptor(Delta) });
    kernels.register_execute_parent_kernel(Filter.id(), Delta, FilterExecuteAdaptor(Delta));
    kernels.register_execute_parent_kernel(Dict.id(), Delta, TakeExecuteAdaptor(Delta));
}
