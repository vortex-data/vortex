// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::dict::TakeExecuteAdaptor;
use vortex_array::arrays::filter::FilterExecuteAdaptor;
use vortex_array::arrays::slice::SliceExecuteAdaptor;
use vortex_array::kernel::ParentKernelSet;

use crate::BitPackedVTable;

pub(crate) const PARENT_KERNELS: ParentKernelSet<BitPackedVTable> = ParentKernelSet::new(&[
    ParentKernelSet::lift(&FilterExecuteAdaptor(BitPackedVTable)),
    ParentKernelSet::lift(&SliceExecuteAdaptor(BitPackedVTable)),
    ParentKernelSet::lift(&TakeExecuteAdaptor(BitPackedVTable)),
]);
