// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_session::VortexSessionBuilder;

use crate::ArrayVTable;
use crate::arrays::Dict;
use crate::arrays::FixedSizeList;
use crate::arrays::dict::TakeExecuteAdaptor;
use crate::optimizer::kernels::builder_kernels;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::fns::cast::Cast;
use crate::scalar_fn::fns::cast::CastExecuteAdaptor;

pub(crate) fn initialize(session: &mut VortexSessionBuilder) {
    let kernels = builder_kernels(session);
    kernels.register_execute_parent_kernel(
        Cast.id(),
        FixedSizeList,
        CastExecuteAdaptor(FixedSizeList),
    );
    kernels.register_execute_parent_kernel(
        Dict.id(),
        FixedSizeList,
        TakeExecuteAdaptor(FixedSizeList),
    );
}
