// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::TakeExecuteAdaptor;
use vortex_array::expr::CompareExecuteAdaptor;
use vortex_array::kernel::ParentKernelSet;

use crate::DecimalBytePartsVTable;

pub(crate) const PARENT_KERNELS: ParentKernelSet<DecimalBytePartsVTable> = ParentKernelSet::new(&[
    ParentKernelSet::lift(&CompareExecuteAdaptor(DecimalBytePartsVTable)),
    ParentKernelSet::lift(&TakeExecuteAdaptor(DecimalBytePartsVTable)),
]);
