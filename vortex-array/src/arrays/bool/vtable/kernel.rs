// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::BoolVTable;
use crate::arrays::TakeExecuteAdaptor;
use crate::arrays::filter::FilterExecuteAdaptor;
use crate::expr::FillNullExecuteAdaptor;
use crate::kernel::ParentKernelSet;

pub(super) const PARENT_KERNELS: ParentKernelSet<BoolVTable> = ParentKernelSet::new(&[
    ParentKernelSet::lift(&FilterExecuteAdaptor(BoolVTable)),
    ParentKernelSet::lift(&FillNullExecuteAdaptor(BoolVTable)),
    ParentKernelSet::lift(&TakeExecuteAdaptor(BoolVTable)),
]);
