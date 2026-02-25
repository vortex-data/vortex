// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::TakeExecuteAdaptor;
use vortex_array::kernel::ParentKernelSet;
use vortex_array::scalar_fn::CompareExecuteAdaptor;

use crate::DateTimePartsVTable;

pub(crate) const PARENT_KERNELS: ParentKernelSet<DateTimePartsVTable> = ParentKernelSet::new(&[
    ParentKernelSet::lift(&CompareExecuteAdaptor(DateTimePartsVTable)),
    ParentKernelSet::lift(&TakeExecuteAdaptor(DateTimePartsVTable)),
]);
