// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_session::VortexSession;

use crate::arrays::Struct;
use crate::optimizer::kernels::ArrayKernelsExt;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::fns::zip::Zip;
use crate::scalar_fn::fns::zip::ZipExecuteAdaptor;

pub(crate) fn initialize(session: &VortexSession) {
    session
        .kernels()
        .register_execute_parent_kernel(Zip.id(), Struct, ZipExecuteAdaptor(Struct));
}
