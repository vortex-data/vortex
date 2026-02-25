// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::TakeExecuteAdaptor;
use crate::arrays::VarBinViewVTable;
use crate::kernel::ParentKernelSet;
use crate::scalar_fn::ZipExecuteAdaptor;

pub(super) const PARENT_KERNELS: ParentKernelSet<VarBinViewVTable> = ParentKernelSet::new(&[
    ParentKernelSet::lift(&TakeExecuteAdaptor(VarBinViewVTable)),
    ParentKernelSet::lift(&ZipExecuteAdaptor(VarBinViewVTable)),
]);
