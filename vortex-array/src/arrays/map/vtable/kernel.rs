// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_session::VortexSession;

use crate::ArrayVTable;
use crate::arrays::Dict;
use crate::arrays::Filter;
use crate::arrays::Map;
use crate::arrays::Slice;
use crate::arrays::dict::TakeExecuteAdaptor;
use crate::arrays::filter::FilterExecuteAdaptor;
use crate::arrays::slice::SliceExecuteAdaptor;
use crate::optimizer::kernels::ArrayKernelsExt;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::fns::cast::Cast;
use crate::scalar_fn::fns::cast::CastExecuteAdaptor;
use crate::scalar_fn::fns::mask::Mask;
use crate::scalar_fn::fns::mask::MaskExecuteAdaptor;

pub(crate) fn initialize(session: &VortexSession) {
    let kernels = session.kernels();
    kernels.register_execute_parent_kernel(Cast.id(), Map, CastExecuteAdaptor(Map));
    kernels.register_execute_parent_kernel(Dict.id(), Map, TakeExecuteAdaptor(Map));
    kernels.register_execute_parent_kernel(Filter.id(), Map, FilterExecuteAdaptor(Map));
    kernels.register_execute_parent_kernel(Mask.id(), Map, MaskExecuteAdaptor(Map));
    kernels.register_execute_parent_kernel(Slice.id(), Map, SliceExecuteAdaptor(Map));
}
