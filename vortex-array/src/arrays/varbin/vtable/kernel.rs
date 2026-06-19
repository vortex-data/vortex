// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_session::VortexSessionBuilder;

use crate::ArrayVTable;
use crate::arrays::Dict;
use crate::arrays::Filter;
use crate::arrays::VarBin;
use crate::arrays::dict::TakeExecuteAdaptor;
use crate::arrays::filter::FilterExecuteAdaptor;
use crate::optimizer::kernels::builder_kernels;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::fns::binary::Binary;
use crate::scalar_fn::fns::binary::CompareExecuteAdaptor;
use crate::scalar_fn::fns::cast::Cast;
use crate::scalar_fn::fns::cast::CastExecuteAdaptor;

pub(crate) fn initialize(session: &mut VortexSessionBuilder) {
    let kernels = builder_kernels(session);
    kernels.register_execute_parent_kernel(Cast.id(), VarBin, CastExecuteAdaptor(VarBin));
    kernels.register_execute_parent_kernel(Binary.id(), VarBin, CompareExecuteAdaptor(VarBin));
    kernels.register_execute_parent_kernel(Filter.id(), VarBin, FilterExecuteAdaptor(VarBin));
    kernels.register_execute_parent_kernel(Dict.id(), VarBin, TakeExecuteAdaptor(VarBin));
}
