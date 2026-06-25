// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_session::VortexSessionBuilder;

use crate::ArrayVTable;
use crate::arrays::Dict;
use crate::arrays::Primitive;
use crate::arrays::dict::TakeExecuteAdaptor;
use crate::optimizer::kernels::builder_kernels;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::fns::between::Between;
use crate::scalar_fn::fns::between::BetweenExecuteAdaptor;
use crate::scalar_fn::fns::cast::Cast;
use crate::scalar_fn::fns::cast::CastExecuteAdaptor;
use crate::scalar_fn::fns::fill_null::FillNull;
use crate::scalar_fn::fns::fill_null::FillNullExecuteAdaptor;
use crate::scalar_fn::fns::zip::Zip;
use crate::scalar_fn::fns::zip::ZipExecuteAdaptor;

pub(crate) fn initialize(session: &mut VortexSessionBuilder) {
    let kernels = builder_kernels(session);
    kernels.register_execute_parent_kernel(
        Between.id(),
        Primitive,
        BetweenExecuteAdaptor(Primitive),
    );
    kernels.register_execute_parent_kernel(Cast.id(), Primitive, CastExecuteAdaptor(Primitive));
    kernels.register_execute_parent_kernel(
        FillNull.id(),
        Primitive,
        FillNullExecuteAdaptor(Primitive),
    );
    kernels.register_execute_parent_kernel(Dict.id(), Primitive, TakeExecuteAdaptor(Primitive));
    kernels.register_execute_parent_kernel(Zip.id(), Primitive, ZipExecuteAdaptor(Primitive));
}
