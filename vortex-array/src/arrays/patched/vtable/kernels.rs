// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_session::VortexSession;

use crate::ArrayVTable;
use crate::arrays::Dict;
use crate::arrays::Patched;
use crate::arrays::dict::TakeExecuteAdaptor;
use crate::optimizer::kernels::ArrayKernelsExt;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::builtin::compare::Compare;
use crate::scalar_fn::builtin::compare::CompareExecuteAdaptor;

pub(crate) fn initialize(session: &VortexSession) {
    let kernels = session.kernels();
    kernels.register_execute_parent_kernel(Compare.id(), Patched, CompareExecuteAdaptor(Patched));
    kernels.register_execute_parent_kernel(Dict.id(), Patched, TakeExecuteAdaptor(Patched));
}
