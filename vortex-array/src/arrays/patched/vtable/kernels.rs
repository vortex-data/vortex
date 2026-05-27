// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::Patched;
use crate::arrays::dict::TakeExecuteAdaptor;
use crate::arrays::filter::FilterExecuteAdaptor;
use crate::kernel::ParentKernelSet;
use crate::scalar_fn::fns::binary::CompareExecuteAdaptor;

pub(super) const PARENT_KERNELS: ParentKernelSet<Patched> = ParentKernelSet::new(&[
    ParentKernelSet::lift(&CompareExecuteAdaptor(Patched)),
    ParentKernelSet::lift(&FilterExecuteAdaptor(Patched)),
    ParentKernelSet::lift(&TakeExecuteAdaptor(Patched)),
]);
