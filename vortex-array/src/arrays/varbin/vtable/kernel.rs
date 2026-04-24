// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::VarBin;
use crate::arrays::dict::TakeExecuteAdaptor;
use crate::arrays::filter::FilterExecuteAdaptor;
use crate::kernel::ParentKernelSet;
use crate::scalar_fn::fns::binary::CompareExecuteAdaptor;

pub(super) const PARENT_KERNELS: ParentKernelSet<VarBin> = ParentKernelSet::new(&[
    ParentKernelSet::lift(&CompareExecuteAdaptor(VarBin)),
    ParentKernelSet::lift(&FilterExecuteAdaptor(VarBin)),
    ParentKernelSet::lift(&TakeExecuteAdaptor(VarBin)),
]);
