// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_session::VortexSessionBuilder;

use crate::arrays::ListView;
use crate::optimizer::kernels::builder_kernels;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::fns::cast::Cast;
use crate::scalar_fn::fns::cast::CastExecuteAdaptor;
use crate::scalar_fn::fns::zip::Zip;
use crate::scalar_fn::fns::zip::ZipExecuteAdaptor;

pub(crate) fn initialize(session: &mut VortexSessionBuilder) {
    let kernels = builder_kernels(session);
    kernels.register_execute_parent_kernel(Cast.id(), ListView, CastExecuteAdaptor(ListView));
    kernels.register_execute_parent_kernel(Zip.id(), ListView, ZipExecuteAdaptor(ListView));
}
