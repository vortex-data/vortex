// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayVTable;
use vortex_array::arrays::Dict;
use vortex_array::arrays::dict::TakeExecuteAdaptor;
use vortex_array::optimizer::kernels::ArrayKernelsExt;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::scalar_fn::fns::binary::Binary;
use vortex_array::scalar_fn::fns::binary::BooleanExecuteAdaptor;
use vortex_array::scalar_fn::fns::cast::Cast;
use vortex_array::scalar_fn::fns::cast::CastExecuteAdaptor;
use vortex_session::VortexSession;

use crate::ByteBool;

pub(crate) fn initialize(session: &VortexSession) {
    let Some(kernels) = session.kernels() else {
        return;
    };
    kernels.register_execute_parent_kernel(Binary.id(), ByteBool, BooleanExecuteAdaptor(ByteBool));
    kernels.register_execute_parent_kernel(Cast.id(), ByteBool, CastExecuteAdaptor(ByteBool));
    kernels.register_execute_parent_kernel(Dict.id(), ByteBool, TakeExecuteAdaptor(ByteBool));
}
