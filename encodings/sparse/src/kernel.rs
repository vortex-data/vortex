// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::FilterExecuteAdaptor;
use vortex_array::arrays::SliceExecuteAdaptor;
use vortex_array::kernel::ParentKernelSet;

use crate::SparseVTable;

pub(crate) static PARENT_KERNELS: ParentKernelSet<SparseVTable> = ParentKernelSet::new(&[
    ParentKernelSet::lift(&FilterExecuteAdaptor(SparseVTable)),
    ParentKernelSet::lift(&SliceExecuteAdaptor(SparseVTable)),
]);
