// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::PrimitiveVTable;
use crate::arrays::TakeExecuteAdaptor;
use crate::compute::CastExecuteAdaptor;
use crate::expr::BetweenExecuteAdaptor;
use crate::expr::FillNullExecuteAdaptor;
use crate::kernel::ParentKernelSet;

pub(super) const PARENT_KERNELS: ParentKernelSet<PrimitiveVTable> = ParentKernelSet::new(&[
    ParentKernelSet::lift(&BetweenExecuteAdaptor(PrimitiveVTable)),
    ParentKernelSet::lift(&CastExecuteAdaptor(PrimitiveVTable)),
    ParentKernelSet::lift(&FillNullExecuteAdaptor(PrimitiveVTable)),
    ParentKernelSet::lift(&TakeExecuteAdaptor(PrimitiveVTable)),
]);
