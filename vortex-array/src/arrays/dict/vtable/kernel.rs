// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::DictVTable;
use crate::arrays::TakeExecuteAdaptor;
use crate::expr::FillNullExecuteAdaptor;
use crate::kernel::ParentKernelSet;

pub(super) const PARENT_KERNELS: ParentKernelSet<DictVTable> = ParentKernelSet::new(&[
    ParentKernelSet::lift(&TakeExecuteAdaptor(DictVTable)),
    ParentKernelSet::lift(&FillNullExecuteAdaptor(DictVTable)),
]);
