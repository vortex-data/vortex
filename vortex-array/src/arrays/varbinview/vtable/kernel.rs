// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_session::VortexSessionBuilder;

use crate::ArrayVTable;
use crate::arrays::Dict;
use crate::arrays::VarBinView;
use crate::arrays::dict::TakeExecuteAdaptor;
use crate::optimizer::kernels::builder_kernels;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::fns::cast::Cast;
use crate::scalar_fn::fns::cast::CastExecuteAdaptor;
use crate::scalar_fn::fns::zip::Zip;
use crate::scalar_fn::fns::zip::ZipExecuteAdaptor;

pub(crate) fn initialize(session: &mut VortexSessionBuilder) {
    let kernels = builder_kernels(session);
    kernels.register_execute_parent_kernel(Cast.id(), VarBinView, CastExecuteAdaptor(VarBinView));
    kernels.register_execute_parent_kernel(Dict.id(), VarBinView, TakeExecuteAdaptor(VarBinView));
    kernels.register_execute_parent_kernel(Zip.id(), VarBinView, ZipExecuteAdaptor(VarBinView));
}
