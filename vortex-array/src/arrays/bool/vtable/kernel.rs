// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_session::VortexSession;

use crate::ArrayVTable;
use crate::arrays::Bool;
use crate::arrays::Dict;
use crate::arrays::dict::TakeExecuteAdaptor;
use crate::optimizer::kernels::ArrayKernelsExt;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::fns::binary::Binary;
use crate::scalar_fn::fns::binary::BooleanExecuteAdaptor;
use crate::scalar_fn::fns::cast::Cast;
use crate::scalar_fn::fns::cast::CastExecuteAdaptor;
use crate::scalar_fn::fns::fill_null::FillNull;
use crate::scalar_fn::fns::fill_null::FillNullExecuteAdaptor;
use crate::scalar_fn::fns::zip::Zip;
use crate::scalar_fn::fns::zip::ZipExecuteAdaptor;

pub(crate) fn initialize(session: &VortexSession) {
    let Some(kernels) = session.kernels() else {
        return;
    };
    kernels.register_execute_parent_kernel(Binary.id(), Bool, BooleanExecuteAdaptor(Bool));
    kernels.register_execute_parent_kernel(Cast.id(), Bool, CastExecuteAdaptor(Bool));
    kernels.register_execute_parent_kernel(FillNull.id(), Bool, FillNullExecuteAdaptor(Bool));
    kernels.register_execute_parent_kernel(Dict.id(), Bool, TakeExecuteAdaptor(Bool));
    kernels.register_execute_parent_kernel(Zip.id(), Bool, ZipExecuteAdaptor(Bool));
}
