// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::dict::TakeExecuteAdaptor;
use crate::arrays::filter::FilterExecuteAdaptor;
use crate::arrays::list::ListVTable;
use crate::kernel::ParentKernelSet;

pub(crate) const PARENT_KERNELS: ParentKernelSet<ListVTable> = ParentKernelSet::new(&[
    ParentKernelSet::lift(&FilterExecuteAdaptor(ListVTable)),
    ParentKernelSet::lift(&TakeExecuteAdaptor(ListVTable)),
]);
