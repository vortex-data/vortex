// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::dict::TakeExecuteAdaptor;
use crate::arrays::fixed_size_list::FixedSizeListVTable;
use crate::kernel::ParentKernelSet;

impl FixedSizeListVTable {
    pub(crate) const PARENT_KERNELS: ParentKernelSet<FixedSizeListVTable> =
        ParentKernelSet::new(&[ParentKernelSet::lift(&TakeExecuteAdaptor(
            FixedSizeListVTable,
        ))]);
}
