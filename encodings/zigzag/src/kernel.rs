// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::dict::TakeExecuteAdaptor;
use vortex_array::kernel::ParentKernelSet;

use crate::ZigZag;

pub(crate) const PARENT_KERNELS: ParentKernelSet<ZigZag> =
    ParentKernelSet::new(&[ParentKernelSet::lift(&TakeExecuteAdaptor(ZigZag))]);
