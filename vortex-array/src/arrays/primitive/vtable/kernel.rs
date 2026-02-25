// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::PrimitiveVTable;
use crate::arrays::TakeExecuteAdaptor;
use crate::kernel::ParentKernelSet;
use crate::scalar_fn::BetweenExecuteAdaptor;
use crate::scalar_fn::CastExecuteAdaptor;
use crate::scalar_fn::FillNullExecuteAdaptor;

pub(super) const PARENT_KERNELS: ParentKernelSet<PrimitiveVTable> = ParentKernelSet::new(&[
    ParentKernelSet::lift(&BetweenExecuteAdaptor(PrimitiveVTable)),
    ParentKernelSet::lift(&CastExecuteAdaptor(PrimitiveVTable)),
    ParentKernelSet::lift(&FillNullExecuteAdaptor(PrimitiveVTable)),
    ParentKernelSet::lift(&TakeExecuteAdaptor(PrimitiveVTable)),
]);
