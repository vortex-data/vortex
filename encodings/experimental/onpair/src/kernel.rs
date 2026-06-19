// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayVTable;
use vortex_array::arrays::Filter;
use vortex_array::arrays::filter::FilterExecuteAdaptor;
use vortex_array::optimizer::kernels::builder_kernels;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::scalar_fn::fns::binary::Binary;
use vortex_array::scalar_fn::fns::binary::CompareExecuteAdaptor;
use vortex_array::scalar_fn::fns::byte_length::ByteLength;
use vortex_array::scalar_fn::fns::byte_length::ByteLengthExecuteAdaptor;
use vortex_session::VortexSessionBuilder;

use crate::OnPair;

// TODO: implement ListExecute & TakeExecute for OnPair
pub(super) fn initialize(session: &mut VortexSessionBuilder) {
    let kernels = builder_kernels(session);
    kernels.register_execute_parent_kernel(Filter.id(), OnPair, FilterExecuteAdaptor(OnPair));
    kernels.register_execute_parent_kernel(Binary.id(), OnPair, CompareExecuteAdaptor(OnPair));
    kernels.register_execute_parent_kernel(
        ByteLength.id(),
        OnPair,
        ByteLengthExecuteAdaptor(OnPair),
    );
}
