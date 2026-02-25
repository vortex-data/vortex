// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::DecimalVTable;
use crate::arrays::TakeExecuteAdaptor;
use crate::kernel::ParentKernelSet;
use crate::scalar_fn::BetweenExecuteAdaptor;
use crate::scalar_fn::CastExecuteAdaptor;
use crate::scalar_fn::FillNullExecuteAdaptor;

pub(super) const PARENT_KERNELS: ParentKernelSet<DecimalVTable> = ParentKernelSet::new(&[
    ParentKernelSet::lift(&BetweenExecuteAdaptor(DecimalVTable)),
    ParentKernelSet::lift(&CastExecuteAdaptor(DecimalVTable)),
    ParentKernelSet::lift(&FillNullExecuteAdaptor(DecimalVTable)),
    ParentKernelSet::lift(&TakeExecuteAdaptor(DecimalVTable)),
]);
