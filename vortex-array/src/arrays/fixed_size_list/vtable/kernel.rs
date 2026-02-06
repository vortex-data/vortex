// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::FixedSizeListVTable;
use crate::arrays::TakeExecuteAdaptor;
use crate::kernel::ParentKernelSet;

pub(super) const PARENT_KERNELS: ParentKernelSet<FixedSizeListVTable> =
    ParentKernelSet::new(&[ParentKernelSet::lift(&TakeExecuteAdaptor(
        FixedSizeListVTable,
    ))]);
