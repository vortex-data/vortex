// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Registering the pushdown kernels, each letting a parent operation execute against the encoded
//! array instead of canonicalising first. An unregistered kernel is still correct, just slower, so
//! the tests assert the pushdown is actually taken.

use vortex_array::ArrayVTable;
use vortex_array::arrays::Dict;
use vortex_array::arrays::Filter;
use vortex_array::arrays::dict::TakeExecuteAdaptor;
use vortex_array::arrays::filter::FilterExecuteAdaptor;
use vortex_array::optimizer::kernels::ArrayKernelsExt;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::scalar_fn::fns::binary::Binary;
use vortex_array::scalar_fn::fns::binary::CompareExecuteAdaptor;
use vortex_session::VortexSession;

use crate::EliasFano;

pub(crate) fn initialize(session: &VortexSession) {
    let kernels = session.kernels();
    kernels.register_execute_parent_kernel(Binary.id(), EliasFano, CompareExecuteAdaptor(EliasFano));
    kernels.register_execute_parent_kernel(Filter.id(), EliasFano, FilterExecuteAdaptor(EliasFano));
    kernels.register_execute_parent_kernel(Dict.id(), EliasFano, TakeExecuteAdaptor(EliasFano));
}
