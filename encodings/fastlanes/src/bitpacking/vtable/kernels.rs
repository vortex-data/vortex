// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::dict::TakeExecuteAdaptor;
use vortex_array::arrays::filter::FilterExecuteAdaptor;
use vortex_array::kernel::ParentKernelSet;

use crate::BitPacked;

pub(crate) const PARENT_KERNELS: ParentKernelSet<BitPacked> = ParentKernelSet::new(&[
    ParentKernelSet::lift(&FilterExecuteAdaptor(BitPacked)),
    ParentKernelSet::lift(&TakeExecuteAdaptor(BitPacked)),
]);
