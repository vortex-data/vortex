// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_session::VortexSessionBuilder;

use crate::ArrayVTable;
use crate::arrays::Decimal;
use crate::arrays::Dict;
use crate::arrays::dict::TakeExecuteAdaptor;
use crate::optimizer::kernels::builder_kernels;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::fns::between::Between;
use crate::scalar_fn::fns::between::BetweenExecuteAdaptor;
use crate::scalar_fn::fns::cast::Cast;
use crate::scalar_fn::fns::cast::CastExecuteAdaptor;
use crate::scalar_fn::fns::fill_null::FillNull;
use crate::scalar_fn::fns::fill_null::FillNullExecuteAdaptor;

pub(crate) fn initialize(session: &mut VortexSessionBuilder) {
    let kernels = builder_kernels(session);
    kernels.register_execute_parent_kernel(Between.id(), Decimal, BetweenExecuteAdaptor(Decimal));
    kernels.register_execute_parent_kernel(Cast.id(), Decimal, CastExecuteAdaptor(Decimal));
    kernels.register_execute_parent_kernel(FillNull.id(), Decimal, FillNullExecuteAdaptor(Decimal));
    kernels.register_execute_parent_kernel(Dict.id(), Decimal, TakeExecuteAdaptor(Decimal));
}
