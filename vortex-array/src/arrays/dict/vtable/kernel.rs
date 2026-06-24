// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_session::VortexSession;

use crate::ArrayVTable;
use crate::arrays::Dict;
use crate::arrays::dict::TakeExecuteAdaptor;
use crate::optimizer::kernels::ArrayKernelsExt;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::fns::binary::Binary;
use crate::scalar_fn::fns::binary::CompareExecuteAdaptor;
use crate::scalar_fn::fns::fill_null::FillNull;
use crate::scalar_fn::fns::fill_null::FillNullExecuteAdaptor;

pub(crate) fn initialize(session: &VortexSession) {
    let kernels = session.kernels();
    kernels.register_execute_parent_kernel(Binary.id(), Dict, CompareExecuteAdaptor(Dict));
    kernels.register_execute_parent_kernel(Dict.id(), Dict, TakeExecuteAdaptor(Dict));
    kernels.register_execute_parent_kernel(FillNull.id(), Dict, FillNullExecuteAdaptor(Dict));
}
