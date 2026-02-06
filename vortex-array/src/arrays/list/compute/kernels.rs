// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::FilterExecuteAdaptor;
use crate::arrays::ListVTable;
use crate::arrays::TakeExecuteAdaptor;
use crate::kernel::ParentKernelSet;

pub(crate) const PARENT_KERNELS: ParentKernelSet<ListVTable> = ParentKernelSet::new(&[
    ParentKernelSet::lift(&FilterExecuteAdaptor(ListVTable)),
    ParentKernelSet::lift(&TakeExecuteAdaptor(ListVTable)),
]);
