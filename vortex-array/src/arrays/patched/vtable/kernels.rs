// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_session::VortexSession;

use crate::ArrayVTable;
use crate::arrays::Dict;
use crate::arrays::Filter;
use crate::arrays::Patched;
use crate::arrays::dict::TakeExecuteAdaptor;
use crate::arrays::filter::FilterExecuteAdaptor;
use crate::optimizer::kernels::ArrayKernelsExt;

pub(crate) fn initialize(session: &VortexSession) {
    let kernels = session.kernels();
    kernels.register_execute_parent_kernel(Dict.id(), Patched, TakeExecuteAdaptor(Patched));
    kernels.register_execute_parent_kernel(Filter.id(), Patched, FilterExecuteAdaptor(Patched));
}
