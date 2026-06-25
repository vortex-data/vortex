// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayVTable;
use vortex_array::arrays::Dict;
use vortex_array::arrays::Filter;
use vortex_array::arrays::Slice;
use vortex_array::arrays::dict::TakeExecuteAdaptor;
use vortex_array::arrays::filter::FilterExecuteAdaptor;
use vortex_array::arrays::slice::SliceExecuteAdaptor;
use vortex_array::optimizer::kernels::builder_kernels;
use vortex_session::VortexSessionBuilder;

use crate::alp_rd::ALPRD;

pub(crate) fn initialize(session: &mut VortexSessionBuilder) {
    let kernels = builder_kernels(session);
    kernels.register_execute_parent_kernel(Slice.id(), ALPRD, SliceExecuteAdaptor(ALPRD));
    kernels.register_execute_parent_kernel(Filter.id(), ALPRD, FilterExecuteAdaptor(ALPRD));
    kernels.register_execute_parent_kernel(Dict.id(), ALPRD, TakeExecuteAdaptor(ALPRD));
}
