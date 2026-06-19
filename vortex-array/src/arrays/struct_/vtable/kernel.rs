// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_session::VortexSessionBuilder;

use crate::arrays::Struct;
use crate::optimizer::kernels::builder_kernels;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::fns::zip::Zip;
use crate::scalar_fn::fns::zip::ZipExecuteAdaptor;

pub(crate) fn initialize(session: &mut VortexSessionBuilder) {
    builder_kernels(session).register_execute_parent_kernel(
        Zip.id(),
        Struct,
        ZipExecuteAdaptor(Struct),
    );
}
