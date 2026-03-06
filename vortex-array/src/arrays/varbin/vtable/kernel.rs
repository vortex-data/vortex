// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::dict::TakeExecuteAdaptor;
use crate::arrays::filter::FilterExecuteAdaptor;
use crate::arrays::varbin::VarBinVTable;
use crate::kernel::ParentKernelSet;
use crate::scalar_fn::fns::binary::CompareExecuteAdaptor;

pub(super) const PARENT_KERNELS: ParentKernelSet<VarBinVTable> = ParentKernelSet::new(&[
    ParentKernelSet::lift(&CompareExecuteAdaptor(VarBinVTable)),
    ParentKernelSet::lift(&FilterExecuteAdaptor(VarBinVTable)),
    ParentKernelSet::lift(&TakeExecuteAdaptor(VarBinVTable)),
]);
