// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_session::VortexSession;

use crate::ArrayVTable;
use crate::arrays::Dict;
use crate::arrays::Extension;
use crate::arrays::dict::TakeExecuteAdaptor;
use crate::optimizer::kernels::ArrayKernelsExt;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::fns::binary::Binary;
use crate::scalar_fn::fns::binary::CompareExecuteAdaptor;

pub(crate) fn initialize(session: &VortexSession) {
    let Some(kernels) = session.kernels() else {
        return;
    };
    kernels.register_execute_parent_kernel(
        Binary.id(),
        Extension,
        CompareExecuteAdaptor(Extension),
    );
    kernels.register_execute_parent_kernel(Dict.id(), Extension, TakeExecuteAdaptor(Extension));
}
