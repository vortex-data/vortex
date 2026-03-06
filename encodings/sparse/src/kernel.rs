// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::dict::TakeExecuteAdaptor;
use vortex_array::arrays::filter::FilterExecuteAdaptor;
use vortex_array::arrays::slice::SliceExecuteAdaptor;
use vortex_array::kernel::ParentKernelSet;

use crate::SparseVTable;

pub(crate) static PARENT_KERNELS: ParentKernelSet<SparseVTable> = ParentKernelSet::new(&[
    ParentKernelSet::lift(&FilterExecuteAdaptor(SparseVTable)),
    ParentKernelSet::lift(&SliceExecuteAdaptor(SparseVTable)),
    ParentKernelSet::lift(&TakeExecuteAdaptor(SparseVTable)),
]);
