// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::ChunkedVTable;
use crate::arrays::FilterExecuteAdaptor;
use crate::arrays::SliceExecuteAdaptor;
use crate::arrays::TakeExecuteAdaptor;
use crate::kernel::ParentKernelSet;
use crate::scalar_fn::fns::mask::MaskExecuteAdaptor;
use crate::scalar_fn::fns::zip::ZipExecuteAdaptor;

pub(crate) static PARENT_KERNELS: ParentKernelSet<ChunkedVTable> = ParentKernelSet::new(&[
    ParentKernelSet::lift(&FilterExecuteAdaptor(ChunkedVTable)),
    ParentKernelSet::lift(&MaskExecuteAdaptor(ChunkedVTable)),
    ParentKernelSet::lift(&SliceExecuteAdaptor(ChunkedVTable)),
    ParentKernelSet::lift(&TakeExecuteAdaptor(ChunkedVTable)),
    ParentKernelSet::lift(&ZipExecuteAdaptor(ChunkedVTable)),
]);
