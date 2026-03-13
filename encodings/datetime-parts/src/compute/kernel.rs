// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::dict::TakeExecuteAdaptor;
use vortex_array::kernel::ParentKernelSet;
use vortex_array::scalar_fn::fns::binary::CompareExecuteAdaptor;

use crate::DateTimeParts;

pub(crate) const PARENT_KERNELS: ParentKernelSet<DateTimeParts> = ParentKernelSet::new(&[
    ParentKernelSet::lift(&CompareExecuteAdaptor(DateTimeParts)),
    ParentKernelSet::lift(&TakeExecuteAdaptor(DateTimeParts)),
]);
