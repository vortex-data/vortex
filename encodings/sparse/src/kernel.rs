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
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::scalar_fn::fns::between::Between;
use vortex_array::scalar_fn::fns::between::BetweenExecuteAdaptor;
use vortex_array::scalar_fn::fns::binary::Binary;
use vortex_array::scalar_fn::fns::binary::CompareExecuteAdaptor;
use vortex_array::scalar_fn::fns::fill_null::FillNull;
use vortex_array::scalar_fn::fns::fill_null::FillNullExecuteAdaptor;
use vortex_session::VortexSessionBuilder;

use crate::Sparse;

pub(crate) fn initialize(session: &mut VortexSessionBuilder) {
    let kernels = builder_kernels(session);
    kernels.register_execute_parent_kernel(Between.id(), Sparse, BetweenExecuteAdaptor(Sparse));
    kernels.register_execute_parent_kernel(Binary.id(), Sparse, CompareExecuteAdaptor(Sparse));
    kernels.register_execute_parent_kernel(FillNull.id(), Sparse, FillNullExecuteAdaptor(Sparse));
    kernels.register_execute_parent_kernel(Filter.id(), Sparse, FilterExecuteAdaptor(Sparse));
    kernels.register_execute_parent_kernel(Slice.id(), Sparse, SliceExecuteAdaptor(Sparse));
    kernels.register_execute_parent_kernel(Dict.id(), Sparse, TakeExecuteAdaptor(Sparse));
}
