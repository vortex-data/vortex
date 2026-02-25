// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::TakeExecuteAdaptor;
use crate::arrays::VarBinVTable;
use crate::arrays::filter::FilterExecuteAdaptor;
use crate::kernel::ParentKernelSet;
use crate::scalar_fn::CompareExecuteAdaptor;

pub(super) const PARENT_KERNELS: ParentKernelSet<VarBinVTable> = ParentKernelSet::new(&[
    ParentKernelSet::lift(&CompareExecuteAdaptor(VarBinVTable)),
    ParentKernelSet::lift(&FilterExecuteAdaptor(VarBinVTable)),
    ParentKernelSet::lift(&TakeExecuteAdaptor(VarBinVTable)),
]);
