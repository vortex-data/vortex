// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::DictVTable;
use crate::arrays::TakeExecuteAdaptor;
use crate::kernel::ParentKernelSet;
use crate::scalar_fn::CompareExecuteAdaptor;
use crate::scalar_fn::FillNullExecuteAdaptor;

pub(super) const PARENT_KERNELS: ParentKernelSet<DictVTable> = ParentKernelSet::new(&[
    ParentKernelSet::lift(&CompareExecuteAdaptor(DictVTable)),
    ParentKernelSet::lift(&TakeExecuteAdaptor(DictVTable)),
    ParentKernelSet::lift(&FillNullExecuteAdaptor(DictVTable)),
]);
