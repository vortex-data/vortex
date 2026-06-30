// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayVTable;
use vortex_array::arrays::Dict;
use vortex_array::arrays::Filter;
use vortex_array::arrays::Slice;
use vortex_array::arrays::dict::TakeExecuteAdaptor;
use vortex_array::arrays::filter::FilterExecuteAdaptor;
use vortex_array::arrays::slice::SliceExecuteAdaptor;
use vortex_array::optimizer::kernels::ArrayKernelsExt;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::scalar_fn::fns::between::Between;
use vortex_array::scalar_fn::fns::between::BetweenExecuteAdaptor;
use vortex_array::scalar_fn::fns::binary::Binary;
use vortex_array::scalar_fn::fns::binary::CompareExecuteAdaptor;
use vortex_array::scalar_fn::fns::cast::Cast;
use vortex_array::scalar_fn::fns::cast::CastExecuteAdaptor;
use vortex_session::VortexSession;

use crate::BitPacked;

pub(crate) fn initialize(session: &VortexSession) {
    let kernels = session.kernels();
    kernels.register_execute_parent_kernel(
        Between.id(),
        BitPacked,
        BetweenExecuteAdaptor(BitPacked),
    );
    kernels.register_execute_parent_kernel(Cast.id(), BitPacked, CastExecuteAdaptor(BitPacked));
    kernels.register_execute_parent_kernel(
        Binary.id(),
        BitPacked,
        CompareExecuteAdaptor(BitPacked),
    );
    kernels.register_execute_parent_kernel(Filter.id(), BitPacked, FilterExecuteAdaptor(BitPacked));
    kernels.register_execute_parent_kernel(Slice.id(), BitPacked, SliceExecuteAdaptor(BitPacked));
    kernels.register_execute_parent_kernel(Dict.id(), BitPacked, TakeExecuteAdaptor(BitPacked));
}
