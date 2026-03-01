// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::FilterExecuteAdaptor;
use vortex_array::arrays::TakeExecuteAdaptor;
use vortex_array::kernel::ParentKernelSet;

use crate::BitPackedVTable;

pub(crate) const PARENT_KERNELS: ParentKernelSet<BitPackedVTable> = ParentKernelSet::new(&[
    ParentKernelSet::lift(&FilterExecuteAdaptor(BitPackedVTable)),
    ParentKernelSet::lift(&TakeExecuteAdaptor(BitPackedVTable)),
]);
