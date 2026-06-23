// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayVTable;
use vortex_array::arrays::Dict;
use vortex_array::arrays::dict::TakeExecuteAdaptor;
use vortex_array::optimizer::kernels::ArrayKernelsExt;
use vortex_session::VortexSession;

use crate::ZigZag;

pub(crate) fn initialize(session: &VortexSession) {
    session
        .kernels()
        .register_execute_parent_kernel(Dict.id(), ZigZag, TakeExecuteAdaptor(ZigZag));
}
