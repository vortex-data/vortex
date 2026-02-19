// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::FilterExecuteAdaptor;
use vortex_array::arrays::TakeExecuteAdaptor;
use vortex_array::expr::CompareExecuteAdaptor;
use vortex_array::kernel::ParentKernelSet;

use crate::SequenceVTable;

pub(crate) const PARENT_KERNELS: ParentKernelSet<SequenceVTable> = ParentKernelSet::new(&[
    ParentKernelSet::lift(&CompareExecuteAdaptor(SequenceVTable)),
    ParentKernelSet::lift(&FilterExecuteAdaptor(SequenceVTable)),
    ParentKernelSet::lift(&TakeExecuteAdaptor(SequenceVTable)),
]);
