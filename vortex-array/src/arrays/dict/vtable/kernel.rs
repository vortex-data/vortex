// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::dict::DictVTable;
use crate::arrays::dict::TakeExecuteAdaptor;
use crate::kernel::ParentKernelSet;
use crate::scalar_fn::fns::binary::CompareExecuteAdaptor;
use crate::scalar_fn::fns::fill_null::FillNullExecuteAdaptor;

pub(super) const PARENT_KERNELS: ParentKernelSet<DictVTable> = ParentKernelSet::new(&[
    ParentKernelSet::lift(&CompareExecuteAdaptor(DictVTable)),
    ParentKernelSet::lift(&TakeExecuteAdaptor(DictVTable)),
    ParentKernelSet::lift(&FillNullExecuteAdaptor(DictVTable)),
]);
