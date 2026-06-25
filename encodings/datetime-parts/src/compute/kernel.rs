// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayVTable;
use vortex_array::arrays::Dict;
use vortex_array::arrays::dict::TakeExecuteAdaptor;
use vortex_array::optimizer::kernels::builder_kernels;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::scalar_fn::fns::binary::Binary;
use vortex_array::scalar_fn::fns::binary::CompareExecuteAdaptor;
use vortex_session::VortexSessionBuilder;

use crate::DateTimeParts;

pub(crate) fn initialize(session: &mut VortexSessionBuilder) {
    let kernels = builder_kernels(session);
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
