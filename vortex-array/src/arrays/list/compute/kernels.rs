// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_session::VortexSession;

use crate::ArrayVTable;
use crate::arrays::Dict;
use crate::arrays::Filter;
use crate::arrays::List;
use crate::arrays::dict::TakeExecuteAdaptor;
use crate::arrays::filter::FilterExecuteAdaptor;
use crate::optimizer::kernels::ArrayKernelsExt;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::fns::cast::Cast;
use crate::scalar_fn::fns::cast::CastExecuteAdaptor;

pub(crate) fn initialize(session: &VortexSession) {
    let kernels = session.kernels();
    kernels.register_execute_parent_kernel(Cast.id(), List, const { &CastExecuteAdaptor(List) });
    kernels.register_execute_parent_kernel(Filter.id(), List, const { &FilterExecuteAdaptor(List) });
    kernels.register_execute_parent_kernel(Dict.id(), List, const { &TakeExecuteAdaptor(List) });
}
