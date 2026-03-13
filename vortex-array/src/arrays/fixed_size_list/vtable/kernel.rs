// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::FixedSizeList;
use crate::arrays::dict::TakeExecuteAdaptor;
use crate::kernel::ParentKernelSet;

impl FixedSizeList {
    pub(crate) const PARENT_KERNELS: ParentKernelSet<FixedSizeList> =
        ParentKernelSet::new(&[ParentKernelSet::lift(&TakeExecuteAdaptor(FixedSizeList))]);
}
