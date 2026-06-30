// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayVTable;
use vortex_array::arrays::Dict;
use vortex_array::arrays::dict::TakeExecuteAdaptor;
use vortex_array::optimizer::kernels::ArrayKernelsExt;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::scalar_fn::fns::binary::Binary;
use vortex_array::scalar_fn::fns::binary::CompareExecuteAdaptor;
use vortex_session::VortexSession;

use crate::DateTimeParts;

pub(crate) fn initialize(session: &VortexSession) {
    let kernels = session.kernels();
    kernels.register_execute_parent_kernel(
        Binary.id(),
        DateTimeParts,
        CompareExecuteAdaptor(DateTimeParts),
    );
    kernels.register_execute_parent_kernel(
        Dict.id(),
        DateTimeParts,
        TakeExecuteAdaptor(DateTimeParts),
    );
}
