// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::optimizer::kernels::ArrayKernelsExt;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::scalar_fn::fns::between::Between;
use vortex_array::scalar_fn::fns::between::BetweenExecuteAdaptor;
use vortex_array::scalar_fn::fns::binary::Binary;
use vortex_array::scalar_fn::fns::binary::CompareExecuteAdaptor;
use vortex_array::scalar_fn::fns::cast::Cast;
use vortex_array::scalar_fn::fns::cast::CastExecuteAdaptor;
use vortex_session::VortexSession;

use crate::DecimalByteParts;

pub(crate) fn initialize(session: &VortexSession) {
    let kernels = session.kernels();
    kernels.register_execute_parent_kernel(
        Binary.id(),
        DecimalByteParts,
        CompareExecuteAdaptor(DecimalByteParts),
    );
    kernels.register_execute_parent_kernel(
        Between.id(),
        DecimalByteParts,
        BetweenExecuteAdaptor(DecimalByteParts),
    );
    kernels.register_execute_parent_kernel(
        Cast.id(),
        DecimalByteParts,
        CastExecuteAdaptor(DecimalByteParts),
    );
}
