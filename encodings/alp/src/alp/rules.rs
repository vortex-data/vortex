// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::FilterExecuteAdaptor;
use vortex_array::arrays::SliceExecuteAdaptor;
use vortex_array::arrays::TakeExecuteAdaptor;
use vortex_array::kernel::ParentKernelSet;

use crate::ALPVTable;

pub(super) const PARENT_KERNELS: ParentKernelSet<ALPVTable> = ParentKernelSet::new(&[
    ParentKernelSet::lift(&FilterExecuteAdaptor(ALPVTable)),
    ParentKernelSet::lift(&SliceExecuteAdaptor(ALPVTable)),
    ParentKernelSet::lift(&TakeExecuteAdaptor(ALPVTable)),
]);
