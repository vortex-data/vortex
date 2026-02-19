// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::ChunkedVTable;
use crate::arrays::FilterExecuteAdaptor;
use crate::arrays::SliceExecuteAdaptor;
use crate::arrays::TakeExecuteAdaptor;
use crate::expr::MaskExecuteAdaptor;
use crate::kernel::ParentKernelSet;

pub(crate) static PARENT_KERNELS: ParentKernelSet<ChunkedVTable> = ParentKernelSet::new(&[
    ParentKernelSet::lift(&FilterExecuteAdaptor(ChunkedVTable)),
    ParentKernelSet::lift(&MaskExecuteAdaptor(ChunkedVTable)),
    ParentKernelSet::lift(&SliceExecuteAdaptor(ChunkedVTable)),
    ParentKernelSet::lift(&TakeExecuteAdaptor(ChunkedVTable)),
]);
