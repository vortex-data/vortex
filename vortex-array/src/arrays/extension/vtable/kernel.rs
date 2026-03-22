// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::Extension;
use crate::arrays::dict::TakeExecuteAdaptor;
use crate::kernel::ParentKernelSet;
use crate::scalar_fn::fns::binary::CompareExecuteAdaptor;

pub(super) const PARENT_KERNELS: ParentKernelSet<Extension> = ParentKernelSet::new(&[
    ParentKernelSet::lift(&CompareExecuteAdaptor(Extension)),
    ParentKernelSet::lift(&TakeExecuteAdaptor(Extension)),
]);
