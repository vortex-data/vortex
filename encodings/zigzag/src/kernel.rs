// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::TakeExecuteAdaptor;
use vortex_array::kernel::ParentKernelSet;

use crate::ZigZagVTable;

pub(crate) const PARENT_KERNELS: ParentKernelSet<ZigZagVTable> =
    ParentKernelSet::new(&[ParentKernelSet::lift(&TakeExecuteAdaptor(ZigZagVTable))]);
