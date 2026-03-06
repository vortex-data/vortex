// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::dict::TakeExecuteAdaptor;
use crate::arrays::extension::ExtensionVTable;
use crate::kernel::ParentKernelSet;
use crate::scalar_fn::fns::binary::CompareExecuteAdaptor;

pub(super) const PARENT_KERNELS: ParentKernelSet<ExtensionVTable> = ParentKernelSet::new(&[
    ParentKernelSet::lift(&CompareExecuteAdaptor(ExtensionVTable)),
    ParentKernelSet::lift(&TakeExecuteAdaptor(ExtensionVTable)),
]);
