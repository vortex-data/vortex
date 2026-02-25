// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::ExtensionVTable;
use crate::arrays::TakeExecuteAdaptor;
use crate::kernel::ParentKernelSet;
use crate::scalar_fn::CompareExecuteAdaptor;

pub(super) const PARENT_KERNELS: ParentKernelSet<ExtensionVTable> = ParentKernelSet::new(&[
    ParentKernelSet::lift(&CompareExecuteAdaptor(ExtensionVTable)),
    ParentKernelSet::lift(&TakeExecuteAdaptor(ExtensionVTable)),
]);
