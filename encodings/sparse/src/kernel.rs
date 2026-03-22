// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::dict::TakeExecuteAdaptor;
use vortex_array::arrays::filter::FilterExecuteAdaptor;
use vortex_array::arrays::slice::SliceExecuteAdaptor;
use vortex_array::kernel::ParentKernelSet;

use crate::Sparse;

pub(crate) static PARENT_KERNELS: ParentKernelSet<Sparse> = ParentKernelSet::new(&[
    ParentKernelSet::lift(&FilterExecuteAdaptor(Sparse)),
    ParentKernelSet::lift(&SliceExecuteAdaptor(Sparse)),
    ParentKernelSet::lift(&TakeExecuteAdaptor(Sparse)),
]);
