// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::BoolVTable;
use crate::arrays::TakeExecuteAdaptor;
use crate::kernel::ParentKernelSet;
use crate::scalar_fn::fns::fill_null::FillNullExecuteAdaptor;

pub(super) const PARENT_KERNELS: ParentKernelSet<BoolVTable> = ParentKernelSet::new(&[
    ParentKernelSet::lift(&FillNullExecuteAdaptor(BoolVTable)),
    ParentKernelSet::lift(&TakeExecuteAdaptor(BoolVTable)),
]);
