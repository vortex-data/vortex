// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_session::VortexSession;

use crate::ArrayVTable;
use crate::arrays::Chunked;
use crate::arrays::Dict;
use crate::arrays::Filter;
use crate::arrays::Slice;
use crate::arrays::dict::TakeExecuteAdaptor;
use crate::arrays::filter::FilterExecuteAdaptor;
use crate::arrays::slice::SliceExecuteAdaptor;
use crate::optimizer::kernels::ArrayKernelsExt;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::fns::mask::Mask;
use crate::scalar_fn::fns::mask::MaskExecuteAdaptor;
use crate::scalar_fn::fns::zip::Zip;
use crate::scalar_fn::fns::zip::ZipExecuteAdaptor;

pub(crate) fn initialize(session: &VortexSession) {
    let kernels = session.kernels();
    kernels.register_execute_parent_kernel(Filter.id(), Chunked, FilterExecuteAdaptor(Chunked));
    kernels.register_execute_parent_kernel(Mask.id(), Chunked, MaskExecuteAdaptor(Chunked));
    kernels.register_execute_parent_kernel(Slice.id(), Chunked, SliceExecuteAdaptor(Chunked));
    kernels.register_execute_parent_kernel(Dict.id(), Chunked, TakeExecuteAdaptor(Chunked));
    kernels.register_execute_parent_kernel(Zip.id(), Chunked, ZipExecuteAdaptor(Chunked));
}
