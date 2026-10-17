// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Parent kernels: operations an Elias-Fano child executes for its parent, reading its buffers.

use vortex_array::ArrayVTable;
use vortex_array::arrays::Slice;
use vortex_array::arrays::slice::SliceExecuteAdaptor;
use vortex_array::optimizer::kernels::ArrayKernelsExt;
use vortex_session::VortexSession;

use crate::EliasFano;

pub(crate) fn initialize(session: &VortexSession) {
    session.kernels().register_execute_parent_kernel(
        Slice.id(),
        EliasFano,
        SliceExecuteAdaptor(EliasFano),
    );
}
