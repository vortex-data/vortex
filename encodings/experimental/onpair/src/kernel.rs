// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayVTable;
use vortex_array::arrays::Filter;
use vortex_array::arrays::filter::FilterExecuteAdaptor;
use vortex_array::optimizer::kernels::ArrayKernelsExt;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::scalar_fn::fns::binary::Binary;
use vortex_array::scalar_fn::fns::binary::CompareExecuteAdaptor;
use vortex_array::scalar_fn::fns::byte_length::ByteLength;
use vortex_array::scalar_fn::fns::byte_length::ByteLengthExecuteAdaptor;
use vortex_array::scalar_fn::fns::like::Like;
use vortex_array::scalar_fn::fns::like::LikeExecuteAdaptor;
use vortex_session::VortexSession;

use crate::OnPair;

// TODO: implement ListExecute & TakeExecute for OnPair
pub(super) fn initialize(session: &VortexSession) {
    let kernels = session.kernels();
    kernels.register_execute_parent_kernel(Filter.id(), OnPair, FilterExecuteAdaptor(OnPair));
    kernels.register_execute_parent_kernel(Binary.id(), OnPair, CompareExecuteAdaptor(OnPair));
    kernels.register_execute_parent_kernel(
        ByteLength.id(),
        OnPair,
        ByteLengthExecuteAdaptor(OnPair),
    );
    kernels.register_execute_parent_kernel(Like.id(), OnPair, LikeExecuteAdaptor(OnPair));
}
