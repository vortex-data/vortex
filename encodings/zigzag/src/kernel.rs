// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayVTable;
use vortex_array::arrays::Dict;
use vortex_array::arrays::dict::TakeExecuteAdaptor;
use vortex_array::optimizer::kernels::builder_kernels;
use vortex_session::VortexSessionBuilder;

use crate::ZigZag;

pub(crate) fn initialize(session: &mut VortexSessionBuilder) {
    builder_kernels(session).register_execute_parent_kernel(
        Dict.id(),
        ZigZag,
        TakeExecuteAdaptor(ZigZag),
    );
}
