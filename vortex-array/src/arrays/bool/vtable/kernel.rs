// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_session::VortexSession;

use crate::ArrayVTable;
use crate::arrays::Bool;
use crate::arrays::Dict;
use crate::arrays::dict::TakeExecuteAdaptor;
use crate::optimizer::kernels::ArrayKernelsExt;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::builtin::binary::Binary;
use crate::scalar_fn::builtin::binary::BooleanExecuteAdaptor;
use crate::scalar_fn::builtin::cast::Cast;
use crate::scalar_fn::builtin::cast::CastExecuteAdaptor;
use crate::scalar_fn::builtin::fill_null::FillNull;
use crate::scalar_fn::builtin::fill_null::FillNullExecuteAdaptor;

pub(crate) fn initialize(session: &VortexSession) {
    let kernels = session.kernels();
    kernels.register_execute_parent_kernel(Binary.id(), Bool, BooleanExecuteAdaptor(Bool));
    kernels.register_execute_parent_kernel(Cast.id(), Bool, CastExecuteAdaptor(Bool));
    kernels.register_execute_parent_kernel(FillNull.id(), Bool, FillNullExecuteAdaptor(Bool));
    kernels.register_execute_parent_kernel(Dict.id(), Bool, TakeExecuteAdaptor(Bool));
}
