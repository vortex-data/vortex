// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::ChunkedVTable;
use crate::arrays::SliceExecuteAdaptor;
use crate::kernel::ParentKernelSet;

pub(crate) static PARENT_KERNELS: ParentKernelSet<ChunkedVTable> =
    ParentKernelSet::new(&[ParentKernelSet::lift(&SliceExecuteAdaptor(ChunkedVTable))]);
