// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::VarBinView;
use crate::arrays::dict::TakeExecuteAdaptor;
use crate::kernel::ParentKernelSet;
use crate::scalar_fn::fns::zip::ZipExecuteAdaptor;

pub(super) const PARENT_KERNELS: ParentKernelSet<VarBinView> = ParentKernelSet::new(&[
    ParentKernelSet::lift(&TakeExecuteAdaptor(VarBinView)),
    ParentKernelSet::lift(&ZipExecuteAdaptor(VarBinView)),
]);
