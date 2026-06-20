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
use vortex_session::VortexSession;

use crate::alp_rd::ALPRD;

pub(crate) fn initialize(session: &VortexSession) {
    let kernels = session.kernels();
    kernels.register_execute_parent_kernel(Slice.id(), ALPRD, SliceExecuteAdaptor(ALPRD));
    kernels.register_execute_parent_kernel(Filter.id(), ALPRD, FilterExecuteAdaptor(ALPRD));
    kernels.register_execute_parent_kernel(Dict.id(), ALPRD, TakeExecuteAdaptor(ALPRD));
}
