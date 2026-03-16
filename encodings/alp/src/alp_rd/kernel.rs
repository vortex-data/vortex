// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::dict::TakeExecuteAdaptor;
use vortex_array::arrays::filter::FilterExecuteAdaptor;
use vortex_array::arrays::slice::SliceExecuteAdaptor;
use vortex_array::kernel::ParentKernelSet;

use crate::alp_rd::ALPRD;

pub(crate) static PARENT_KERNELS: ParentKernelSet<ALPRD> = ParentKernelSet::new(&[
    ParentKernelSet::lift(&SliceExecuteAdaptor(ALPRD)),
    ParentKernelSet::lift(&FilterExecuteAdaptor(ALPRD)),
    ParentKernelSet::lift(&TakeExecuteAdaptor(ALPRD)),
]);
